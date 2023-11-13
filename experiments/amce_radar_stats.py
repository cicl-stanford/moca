"""
This file is used to generate radar graph statistics for the website

We port this to Vega plotting
https://vega.github.io/vega/examples/radar-chart/
"""

import json
import pickle

from tqdm import tqdm

import numpy as np
from causal_effect_analysis import compute_amce, compute_model_acme, CausalDataset, MoralDataset, CausalFactorPrompt, \
    MoralFactorPrompt


def get_human_amce_stats(task="causal"):
    if task == 'causal':
        dataset = CausalDataset()
        factor_prompt = CausalFactorPrompt(anno_utils=dataset.anno_utils)
    elif task == 'moral':
        dataset = MoralDataset()
        factor_prompt = MoralFactorPrompt(anno_utils=dataset.anno_utils)
    else:
        raise NotImplementedError

    causal_amce, factor_to_index = compute_amce(dataset, factor_prompt, causal=task == "causal")
    return {'Human': causal_amce}, factor_to_index


def get_amce_stats(model_name, human_result, factor_to_index, task='causal'):
    if task == 'causal':
        dataset = CausalDataset()
        factor_prompt = CausalFactorPrompt(anno_utils=dataset.anno_utils)
    elif task == 'moral':
        dataset = MoralDataset()
        factor_prompt = MoralFactorPrompt(anno_utils=dataset.anno_utils)
    else:
        raise NotImplementedError

    result = {'Human': human_result['Human']}

    all_instances, all_choice_scores, all_label_indices = pickle.load(
        open(f'results/preds/exp1_{model_name}_{task}_preds.pkl', 'rb'))
    acme = compute_model_acme(dataset, all_instances, all_choice_scores, all_label_indices, factor_to_index)
    result[model_name] = acme

    return result


def compute_acme_radar(amce_stats, plot_cs_with_normality=False):
    for name, result in amce_stats.items():

        labels = []
        for label, (mean, (upper, lower)) in result.items():
            if label == 'Conjunctive - Disjunctive':
                continue
            if plot_cs_with_normality:
                # plot the interaction between causal structure and event normality
                if label == 'Normal - Abnormal':
                    continue
            else:
                # plot normal / abnormal like the figure in the paper
                if label == "Conj Abnormal - Conj Normal" or label == "Disj Abnormal - Disj Normal":
                    continue

            labels.append(label)

    left_label = [l.split(' - ')[1].strip() for l in labels]
    left_label = [ll if ll != 'Statistical Norm' else "Statistical" for ll in left_label]
    left_label = [ll if ll != 'Agent of harm' else "Agent" for ll in left_label]
    left_label = [ll if ll != 'Patient of harm' else "Patient" for ll in left_label]
    left_label = [ll if ll != 'Conj Normal' else "Conj. Normal" for ll in left_label]
    left_label = [ll if ll != 'Disj Normal' else "Disj. Normal" for ll in left_label]

    right_label = [l.split(' - ')[0].strip() for l in labels]
    right_label = [ll if ll != 'Prescriptive Norm' else "Prescriptive" for ll in right_label]
    right_label = [ll if ll != 'Agent of harm' else "Agent" for ll in right_label]
    right_label = [ll if ll != 'Patient of harm' else "Patient" for ll in right_label]
    right_label = [ll if ll != 'Conj Abnormal' else "Conj. Abnormal" for ll in right_label]
    right_label = [ll if ll != 'Disj Abnormal' else "Disj. Abnormal" for ll in right_label]

    values_groups = {}
    mean_point_groups = {}

    for name, result in amce_stats.items():
        left_values = []
        right_values = []
        mean_points_left, mean_points_right = [], []
        for label, (mean, (upper, lower)) in result.items():
            if label == 'Conjunctive - Disjunctive':
                continue
            if plot_cs_with_normality:
                # plot the interaction between causal structure and event normality
                if label == 'Normal - Abnormal':
                    continue
            else:
                # plot normal / abnormal like the figure in the paper
                if label == "Conj Abnormal - Conj Normal" or label == "Disj Abnormal - Disj Normal":
                    continue

            # put mean, upper, lower in the right value group
            # (lowever -> left, upper -> right)
            # Left means it's emphasizing on "Unaware"
            if lower >= 0 and upper >= 0:
                left_values.append(0.)
                right_values.append(upper)
            elif lower < 0 and upper > 0:
                left_values.append(abs(lower))
                right_values.append(upper)
            elif lower <= 0 and upper <= 0:
                right_values.append(0.)
                left_values.append(abs(lower))
            elif lower == 0 and upper == 0:
                right_values.append(0.)
                left_values.append(0.)
            else:
                print(lower, upper)
                raise Exception("not possible")

            if mean < 0:
                mean_points_left.append(mean)
                mean_points_right.append(0.)
            else:
                mean_points_right.append(mean)
                mean_points_left.append(0.)

        values_groups[name] = right_values + left_values
        mean_point_groups[name] = mean_points_right + mean_points_left

    labels = right_label + left_label

    vegas_values = []

    for model_name in amce_stats.keys():
        values = values_groups[model_name]
        np_values = np.concatenate((values, [values[0]]))
        mean_points = mean_point_groups[model_name]
        np_mean_points = np.concatenate((mean_points, [mean_points[0]]))

        category = 0 if model_name == 'Human' else 1

        for key, v in zip(labels, values):
            vegas_values.append({"key": key, "value": np.round(v * 100, decimals=1), "category": category})

    return vegas_values


model_names = [
    "gpt-4",
    "gpt-3.5-turbo",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-002",
    "text-davinci-003",
    "claude-v1",
    "llama-2-7b-chat",
    "llama-2-13b-chat",
    "llama-2-70b-chat",
    "vicuna-13b-v1.5-16k",
    "vicuna-13b-v1.5",
    "Qwen-7B-Chat",
    "WizardLM_70B_V1dot0",
    "mpt-30b-instruct",
    "Mistral_7B_Instruct_v0dot1",
    "falcon-7b-instruct",
    "Platypus2-70B-instruct",
]


def produce_causal_js_dic():
    dic = {}
    human_result, factor_to_index = get_human_amce_stats()
    for model_name in tqdm(model_names):
        if 'gpt-' not in model_name and 'text-' not in model_name and 'claude-' not in model_name:
            translated_model_name = model_name.replace(".", "dot").replace("-", "_")
        else:
            translated_model_name = model_name
        amce_stats = get_amce_stats(translated_model_name, human_result, factor_to_index)
        vegas_values = compute_acme_radar(amce_stats)

        dic[model_name] = vegas_values

    json.dump(dic, open("causal_dic.json", "w"), indent=4)

def produce_moral_js_dic():
    dic = {}
    human_result, factor_to_index = get_human_amce_stats(task="moral")
    for model_name in tqdm(model_names):
        if 'gpt-' not in model_name and 'text-' not in model_name and 'claude-' not in model_name:
            translated_model_name = model_name.replace(".", "dot").replace("-", "_")
        else:
            translated_model_name = model_name
        amce_stats = get_amce_stats(translated_model_name, human_result, factor_to_index, task="moral")
        vegas_values = compute_acme_radar(amce_stats)

        dic[model_name] = vegas_values

    json.dump(dic, open("moral_dic.json", "w"), indent=4)


if __name__ == '__main__':
    ...
    # produce_causal_js_dic()
    produce_moral_js_dic()