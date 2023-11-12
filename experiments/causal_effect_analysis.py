import pickle

import numpy as np

from data import CausalDataset, MoralDataset, Example, AbstractDataset, JsonSerializable, Sentence, FactorUtils
from prompt import CausalFactorPrompt, MoralFactorPrompt, FactorPrompt

from typing import Union, List, Tuple, Dict
from tqdm import tqdm

from scipy.stats import norm, t
from copy import copy

import json

def bca_bootstrap(samples, B=100, alpha=0.05):
    n_users = len(samples)  # 89
    n_subsample = n_users  # n_val_patients

    wis_list = []
    for b in range(B):
        ids = np.random.choice(n_users, n_subsample)
        wis_list.append(np.mean(samples[ids]))

    # jackknife
    y = []
    for i in range(n_users):
        sample_copy = np.array(copy(samples))
        sam_copy = np.delete(sample_copy, i)
        y.append(np.mean(sam_copy))

    wis_list = np.array(wis_list)
    wis_list = np.sort(wis_list)
    y = np.array(y)

    avg = np.mean(samples)
    ql, qu = norm.ppf(alpha), norm.ppf(1 - alpha)

    # Acceleration factor
    num = np.sum((y.mean() - y) ** 3)
    den = 6 * np.sum((y.mean() - y) ** 2) ** 1.5
    ahat = num / den

    # Bias correction factor
    zhat = norm.ppf(np.mean(wis_list < avg))
    a1 = norm.cdf(zhat + (zhat + ql) / (1 - ahat * (zhat + ql)))
    a2 = norm.cdf(zhat + (zhat + qu) / (1 - ahat * (zhat + qu)))

    # print('Accel: %0.3f, bz: %0.3f, a1: %0.3f, a2: %0.3f' % (ahat, zhat, a1, a2))
    return np.quantile(wis_list, [a1, a2])


def bootstrap(samples1, samples2, B=100, alpha=0.05):
    samples1 = np.array(samples1)
    samples2 = np.array(samples2)

    score_list = []
    for b in range(B):
        ids_1 = np.random.choice(len(samples1), len(samples1))
        ids_2 = np.random.choice(len(samples2), len(samples2))
        score_list.append(np.mean(samples1[ids_1]) - np.mean(samples2[ids_2]))

    score_list = np.array(score_list)
    mean = np.mean(score_list)
    upper, lower = np.quantile(score_list, [alpha, 1 - alpha])
    return mean, lower, upper


def compute_amce(cd: AbstractDataset, cfp: Union[CausalFactorPrompt, MoralFactorPrompt], causal=False):
    amce = {}
    factor_to_index: Dict[str, Tuple[List, List]] = {}

    for factor in tqdm(cfp.factors):
        # for time, we only do early vs. late, ignore same time
        t_0, t_1 = [], []
        t_0_indx, t_1_indx = [], []

        for i, ex in enumerate(cd):
            factor_categories, factor_instances = cfp.apply(ex)
            if len(factor_categories) == 0:
                continue

            if factor not in factor_categories:
                continue

            group_idx = factor_instances[factor_categories.index(factor)].answer_index
            if factor == 'time':
                if group_idx == 1:
                    continue  # ignore same time cause
                elif group_idx == 2:  # late cause
                    group_idx = 1  # relabel into 1

            eval(f't_{group_idx}').append(ex)
            eval(f't_{group_idx}_indx').append(i)

        t_0_votes = [ex.individual_votes for ex in t_0]
        t_1_votes = [ex.individual_votes for ex in t_1]
        t_0_votes = [item for sublist in t_0_votes for item in sublist]
        t_1_votes = [item for sublist in t_1_votes for item in sublist]

        labels = eval(f"FactorUtils.{factor}_answers")
        if factor == 'time':
            labels = [labels[0], labels[2]]

        # decide directionality of the label
        if np.mean(t_0_votes) - np.mean(t_1_votes) < 0:
            # now we flip them; this is to always keep t_0 - t_1 to be positive
            processed_t_0_votes = t_1_votes
            processed_t_1_votes = t_0_votes
            processed_t_0_indices = t_1_indx
            processed_t_1_indices = t_0_indx
            label_str = f"{labels[1]} - {labels[0]}"
        else:
            processed_t_0_votes = t_0_votes
            processed_t_1_votes = t_1_votes
            processed_t_0_indices = t_0_indx
            processed_t_1_indices = t_1_indx
            label_str = f"{labels[0]} - {labels[1]}"

        mean, lower, upper = bootstrap(processed_t_0_votes, processed_t_1_votes)

        amce[f"{label_str}"] = mean, (lower, upper)
        factor_to_index[f"{label_str}"] = (processed_t_0_indices, processed_t_1_indices)

    if causal:
        t_0_0, t_0_1, t_1_0, t_1_1 = [], [], [], []
        t_0_0_indx, t_0_1_indx, t_1_0_indx, t_1_1_indx = [], [], [], []

        # we find stories that satisfies two factors at the same time
        # conjunctive_abnormal - conjunctive_normal
        # disjunctive_abnormal - disjunctive_normal

        # we create two more factors for the analysis purpose
        for i, ex in enumerate(cd):
            factor_categories, factor_instances = cfp.apply(ex)
            if len(factor_categories) == 0:
                continue

            if "causal_structure" in factor_categories and "event_normality" in factor_categories:
                # then we process the category
                # we should create a group idx
                cs_idx = factor_instances[factor_categories.index("causal_structure")].answer_index
                en_idx = factor_instances[factor_categories.index("event_normality")].answer_index

                eval(f't_{cs_idx}_{en_idx}').append(ex)
                eval(f't_{cs_idx}_{en_idx}_indx').append(i)
            else:
                continue

        t_0_0_votes = [ex.individual_votes for ex in t_0_0]
        t_0_1_votes = [ex.individual_votes for ex in t_0_1]
        t_1_0_votes = [ex.individual_votes for ex in t_1_0]
        t_1_1_votes = [ex.individual_votes for ex in t_1_1]
        labels = ['Conj Normal', 'Conj Abnormal']
        label_str = f"{labels[1]} - {labels[0]}"

        mean, lower, upper = bootstrap(t_0_1_votes, t_0_0_votes)
        amce[f"{label_str}"] = mean, (lower, upper)
        factor_to_index[f"{label_str}"] = (t_0_1_indx, t_0_0_indx)

        labels = ['Disj Normal', 'Disj Abnormal']
        label_str_2 = f"{labels[1]} - {labels[0]}"

        mean, lower, upper = bootstrap(t_1_1_votes, t_1_0_votes)
        amce[f"{label_str_2}"] = mean, (lower, upper)
        factor_to_index[f"{label_str_2}"] = (t_1_1_indx, t_1_0_indx)

    return amce, factor_to_index

def get_factor_to_story_index_mapping(cd: AbstractDataset, cfp: Union[CausalFactorPrompt, MoralFactorPrompt]):
    factor_to_index: Dict[str, Tuple[List, List]] = {}

    for factor in tqdm(cfp.factors):
        # for time, we only do early vs. late, ignore same time
        t_0, t_1 = [], []
        for i, ex in enumerate(cd):
            factor_categories, factor_instances = cfp.apply(ex)
            if len(factor_categories) == 0:
                continue

            if factor not in factor_categories:
                continue

            group_idx = factor_instances[factor_categories.index(factor)].answer_index

            if factor == 'time':
                if group_idx == 1:
                    continue  # ignore same time cause
                elif group_idx == 2:  # late cause
                    group_idx = 1  # relabel into 1

            eval(f't_{group_idx}').append(i)

        labels = eval(f"FactorUtils.{factor}_answers")
        if factor == 'time':
            labels = [labels[0], labels[2]]

        factor_to_index[f"{labels[0]} - {labels[1]}"] = (t_0, t_1)

    return factor_to_index


def compute_model_acme(cd: AbstractDataset, all_instances, all_choice_scores, all_label_indices, factor_to_story_idx_mapping):
    amce = {}

    for factor_labels, (t_0, t_1) in factor_to_story_idx_mapping.items():
        # choice_score = [P(Yes), P(No)], so we get P(Yes)

        # This is a "hack" to deal with we have 2 prmopts, so double the score
        t_0_scores = [all_choice_scores[idx][0] for idx in t_0 + (np.array(t_0) + len(cd)).tolist()]
        t_1_scores = [all_choice_scores[idx][0] for idx in t_1 + (np.array(t_1) + len(cd)).tolist()]

        mean, lower, upper = bootstrap(t_0_scores, t_1_scores)

        amce[factor_labels] = mean, (lower, upper)

    return amce

def compute_model_acme_with_training(cd: AbstractDataset, all_instances,
                                     all_choice_scores, all_label_indices, factor_to_story_idx_mapping,
                                     dataset='causal'):
    amce = {}

    for factor_labels, (t_0, t_1) in factor_to_story_idx_mapping.items():
        # choice_score = [P(Yes), P(No)], so we get P(Yes)

        if dataset == 'moral':
            training_indices= [61, 49, 28, 7, 0] + (np.array([61, 49, 28, 7, 0]) + len(cd)).tolist()
        elif dataset == 'causal':
            training_indices = [112, 29, 1, 23, 8] + (np.array([112, 29, 1, 23, 8]) + len(cd)).tolist()

        # we first recreate the full list
        full_all_choices_scores = []
        for _ in range(len(cd) * 2):
            full_all_choices_scores.append((0, 0))

        # then we put it back in, so that the order is correct
        inner_idx = 0
        for idx in range(len(cd) * 2):
            # we traverse the entire all_choice_scores, keep track of the index
            # don't update it if it's in the training indices
            if idx not in training_indices:
                full_all_choices_scores[idx] = all_choice_scores[inner_idx]
                inner_idx += 1

        # This is a "hack" to deal with we have 2 prmopts, so double the score
        t_0_scores = [full_all_choices_scores[idx][0] for idx in t_0 + (np.array(t_0) + len(cd)).tolist() if idx not in training_indices]
        t_1_scores = [full_all_choices_scores[idx][0] for idx in t_1 + (np.array(t_1) + len(cd)).tolist() if idx not in training_indices]

        mean, lower, upper = bootstrap(t_0_scores, t_1_scores)

        amce[factor_labels] = mean, (lower, upper)

    return amce

def marginal_causal_effect_analysis_table2():
    cd = CausalDataset()

    causal_amce, factor_to_index = compute_amce(cd, CausalFactorPrompt(anno_utils=cd.anno_utils), causal=True)
    result = {'Human': causal_amce}

    for engine in ["text-babbage-001", 'text-curie-001', 'text-davinci-002', 'text-davinci-003', 'alpaca_7b', 'claude-v1', 'gpt-3.5-turbo', 'gpt-4']:
        all_instances, all_choice_scores, all_label_indices = pickle.load(
            open(f'results/preds/exp1_{engine}_causal_preds.pkl', 'rb'))
        acme = compute_model_acme(cd, all_instances, all_choice_scores, all_label_indices, factor_to_index)
        result[engine] = acme

    with open('results/acme_causal_result.json', 'w') as f:
        json.dump(result, f)

def marginal_moral_effect_analysis_table2():
    md = MoralDataset()

    moral_amce, factor_to_index = compute_amce(md, MoralFactorPrompt(anno_utils=md.anno_utils), causal=False)
    result = {'Human': moral_amce}

    for engine in ["text-babbage-001", 'text-curie-001', 'text-davinci-002',
                   'text-davinci-003', 'alpaca_7b', 'claude-v1', 'gpt-3.5-turbo', 'gpt-4']:
        all_instances, all_choice_scores, all_label_indices = pickle.load(
            open(f'results/preds/exp1_{engine}_moral_preds.pkl', 'rb'))
        acme = compute_model_acme(md, all_instances, all_choice_scores, all_label_indices, factor_to_index)
        result[engine] = acme

    with open('results/acme_moral_result.json', 'w') as f:
        json.dump(result, f)

def marginal_causal_effect_analysis_methods_table3():
    # persona: average, persona: best, persona: worst

    category_to_pickle_map = {
        'single shot': "results/preds/exp1_text-davinci-003_causal_preds.pkl",
        'persona - average': "results/preds/exp5_persona_text-davinci-003_causal_preds.pkl",
        'persona - best': "results/preds/exp5_persona_text-davinci-003_causal_preds.pkl",
        "ape": "results/preds/exp6_ape_text-davinci-003_causal_preds.pkl"
    }

    cd = CausalDataset()

    causal_amce, factor_to_index = compute_amce(cd, CausalFactorPrompt(anno_utils=cd.anno_utils), causal=True)
    result = {'Human': causal_amce}

    for name, path in category_to_pickle_map.items():
        if name == 'persona - average':
            # we compute the average for choice-scores here
            # and just use the single-shot evaluation
            all_instances, all_label_indices = [], []
            all_choice_scores = np.zeros((len(cd) * 2, 2))
            cnt = 0
            res = pickle.load(open(path, 'rb'))
            for _, results in res.items():
                if len(all_instances) == 0:
                    all_instances = results[0]
                    all_label_indices = results[2]

                all_choice_scores += np.array(results[1])
                cnt += 1

            all_choice_scores /= cnt

            acme = compute_model_acme_with_training(cd, all_instances, all_choice_scores, all_label_indices, factor_to_index)
        elif name == 'persona - best':
            res = pickle.load(open(path, 'rb'))
            all_instances, all_choice_scores, all_label_indices = res['newDealAmerica_Nia Ellis']
            acme = compute_model_acme_with_training(cd, all_instances, all_choice_scores, all_label_indices, factor_to_index)
        else:
            # for fairness of comparison
            # APE used 5 examples as training data
            # but for single-shot, if we add those 5 examples, it would be very unfair
            all_instances, all_choice_scores, all_label_indices = pickle.load(open(path, 'rb'))
            acme = compute_model_acme_with_training(cd, all_instances, all_choice_scores, all_label_indices, factor_to_index)

        result[name] = acme

    with open('results/methods_acme_causal_result.json', 'w') as f:
        json.dump(result, f)

def marginal_moral_effect_analysis_methods_table3():
    category_to_pickle_map = {
        'single shot': "results/preds/exp1_text-davinci-002_moral_preds.pkl",
        'persona - average': "results/preds/exp5_persona_text-davinci-002_moral_preds.pkl",
        'persona - best': "results/preds/exp5_persona_text-davinci-002_moral_preds.pkl",
        "ape": "results/preds/exp6_ape_text-davinci-002_moral_preds.pkl"
    }

    md = MoralDataset()

    causal_amce, factor_to_index = compute_amce(md, MoralFactorPrompt(anno_utils=md.anno_utils), causal=False)
    result = {'Human': causal_amce}

    for name, path in category_to_pickle_map.items():
        if name == 'persona - average':
            # we compute the average for choice-scores here
            # and just use the single-shot evaluation
            all_instances, all_label_indices = [], []
            all_choice_scores = np.zeros((len(md) * 2, 2))
            cnt = 0
            res = pickle.load(open(path, 'rb'))
            for _, results in res.items():
                if len(all_instances) == 0:
                    all_instances = results[0]
                    all_label_indices = results[2]

                all_choice_scores += np.array(results[1])
                cnt += 1

            all_choice_scores /= cnt

            acme = compute_model_acme_with_training(md, all_instances, all_choice_scores, all_label_indices,
                                                    factor_to_index, dataset='moral')
        elif name == 'persona - best':
            res = pickle.load(open(path, 'rb'))
            all_instances, all_choice_scores, all_label_indices = res['nonPoliticalTwitter_Allen Lee']
            acme = compute_model_acme_with_training(md, all_instances, all_choice_scores, all_label_indices,
                                                    factor_to_index, dataset='moral')
        else:
            # APE, this method required training data
            all_instances, all_choice_scores, all_label_indices = pickle.load(open(path, 'rb'))
            acme = compute_model_acme_with_training(md, all_instances, all_choice_scores, all_label_indices,
                                                    factor_to_index, dataset='moral')

        result[name] = acme

    with open('results/methods_acme_moral_result.json', 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    ...

    # ===== Model Size =====
    marginal_causal_effect_analysis_table2()
    marginal_moral_effect_analysis_table2()

    # ===== Methods improvement Table =====
    marginal_causal_effect_analysis_methods_table3()
    marginal_moral_effect_analysis_methods_table3()