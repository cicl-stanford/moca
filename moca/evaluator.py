from typing import List, Tuple, Union, Dict
import numpy as np
import math

from dataclasses import dataclass

from moca.data import JsonSerializable

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, roc_curve, auc, mean_absolute_error, \
    precision_recall_fscore_support


# ==== Experiment Result Data classes ====

@dataclass
class FactorExperimentResult(JsonSerializable):
    acc: float
    conf_interval: tuple[float, float]
    prec: float
    rec: float
    f1: float


# ==== Evaluators ====

class BinaryCIEvaluator:
    def __init__(self, return_conf_interval: bool = True):
        self.return_conf_interval = return_conf_interval

    def get_conf_interval(self, acc: float, n: int) -> Tuple[float, float]:
        interval = 1.96 * math.sqrt((acc * (1 - acc)) / n)
        return acc - interval, acc + interval


class BoostrapCIEvaluator:
    def __init__(self, return_conf_interval: bool = True):
        self.return_conf_interval = return_conf_interval

    def get_conf_interval(self, accs: List[int]) -> Tuple[float, float]:
        # Bootstrap
        n_bootstraps = 1000
        n = len(accs)
        bootstrapped_scores = []
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = np.random.randint(0, n, n)
            if len(np.unique(indices)) < 2:
                # We need at least two samples to calculate variance
                continue
            sample = [accs[i] for i in indices]
            bootstrapped_scores.append(np.mean(sample))

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()

        # Return the 95% confidence interval
        return sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]


class AccuracyEvaluator(BoostrapCIEvaluator):
    def evaluate(
            self, choice_scores: List[List[float]], answer_idxs: List[int]
    ) -> Union[float, Tuple[float, Tuple]]:
        accs = []
        for choice_score, answer_index in zip(choice_scores, answer_idxs):
            if np.isnan(choice_score).any():
                choice_score = np.array([0.5, 0.5])
            pred_index = np.argmax(choice_score)
            accs.append(pred_index == answer_index)
        acc = float(np.mean(accs))
        if self.return_conf_interval:
            return acc, self.get_conf_interval(accs)
        else:
            return acc


class AccuracyEvaluatorWithAmbiguity(AccuracyEvaluator):

    def evaluate(
            self, choice_scores: List[List[float]], answer_probs: List[List[float]]
    ) -> Union[float, Tuple[float, Tuple]]:
        accs = []
        for choice_score, answer_prob in zip(choice_scores, answer_probs):
            if np.isnan(choice_score).any():
                choice_score = np.array([0.5, 0.5])
            assert (sum(choice_score) - 1) < 1e-2
            assert (sum(answer_prob) - 1) < 1e-2

            if max(choice_score) <= 0.6:
                choice_idx = 2
            else:
                choice_idx = np.argmax(choice_score)

            if max(answer_prob) <= 0.6:
                answer_idx = 2
            else:
                answer_idx = np.argmax(answer_prob)

            accs.append(choice_idx == answer_idx)
        acc = float(np.mean(accs))
        if self.return_conf_interval:
            return acc, self.get_conf_interval(accs)
        else:
            return acc


class AccuracyEvaluatorWithCategories(BoostrapCIEvaluator):

    def evaluate_within_category(
            self, choice_scores: List[List[float]], answer_idxs: List[int]
    ) -> Union[Tuple[float, float, float, float], Tuple[float, Tuple, float, float, float]]:
        accs = []
        y_true, y_pred = [], []
        for choice_score, answer_index in zip(choice_scores, answer_idxs):
            if np.isnan(choice_score).any():
                choice_score = np.array([0.5, 0.5])
            pred_index = np.argmax(choice_score)
            accs.append(pred_index == answer_index)
            y_true.append(answer_index)
            y_pred.append(pred_index)
        acc = float(np.mean(accs))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        if self.return_conf_interval:
            return acc, self.get_conf_interval(accs), float(prec), float(rec), float(f1)
        else:
            return acc, float(prec), float(rec), float(f1)

    def evaluate(
            self, choice_scores: List[List[float]], answer_idxs: List[int],
            categories: Dict[str, List[int]]
    ) -> Dict[str, FactorExperimentResult]:
        # return a dict of category -> (acc, (lower, upper))
        accs = {}
        for category, indices in categories.items():
            acc, ci, prec, rec, f1 = self.evaluate_within_category(
                [choice_scores[i] for i in indices],
                [answer_idxs[i] for i in indices]
            )
            accs[category] = FactorExperimentResult(acc, ci, prec, rec, f1)

        return accs


def closest_value(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]


class CorrelationEvaluator:
    def evaluate(
            self, choice_scores: List[List[float]], answer_probs: List[List[float]],
            discretize=False, unambiguous_only=False
    ) -> Union[float, Tuple[float, Tuple]]:
        p_correct, p_predicted_correct = [], []
        for choice_score, answer_prob in zip(choice_scores, answer_probs):
            if np.isnan(choice_score).any():
                choice_score = np.array([0.5, 0.5])
            assert (sum(choice_score) - 1) < 1e-2
            assert (sum(answer_prob) - 1) < 1e-2

            if unambiguous_only:
                if max(choice_score) <= 0.6 and max(answer_prob) <= 0.6:
                    continue

            p_correct.append(max(answer_prob))
            answer_index = np.argmax(answer_prob)
            if discretize:
                p_predicted_correct.append(
                    closest_value(list(range(0, 100, 4)), choice_score[answer_index] * 100) / 100)
            else:
                p_predicted_correct.append(choice_score[answer_index])

        r, pvalue = pearsonr(p_correct, p_predicted_correct)

        return r, pvalue


class RMSEEvaluator:
    def evaluate(
            self, choice_scores: List[List[float]], answer_probs: List[List[float]],
            discretize=False, unambiguous_only=False
    ) -> Union[float, Tuple[float, Tuple]]:
        p_correct, p_predicted_correct = [], []
        for choice_score, answer_prob in zip(choice_scores, answer_probs):
            if np.isnan(choice_score).any():
                choice_score = np.array([0.5, 0.5])
            assert (sum(choice_score) - 1) < 1e-2
            assert (sum(answer_prob) - 1) < 1e-2

            if unambiguous_only:
                # this means we only exclude a case if it's ambiguous and the model is correct
                # in inferring if it's ambiguous
                if max(choice_score) <= 0.6 and max(answer_prob) <= 0.6:
                    continue

            p_correct.append(max(answer_prob))
            answer_index = np.argmax(answer_prob)
            # (48%, 52%) , (48%, 52%)
            if discretize:
                p_predicted_correct.append(
                    closest_value(list(range(0, 100, 4)), choice_score[answer_index] * 100) / 100)
            else:
                p_predicted_correct.append(choice_score[answer_index])

        rmse = math.sqrt(mean_squared_error(p_correct, p_predicted_correct))

        return rmse


class MAEEvaluator:
    def evaluate(
            self, choice_scores: List[List[float]], answer_probs: List[List[float]],
            discretize=False, unambiguous_only=False
    ) -> Union[float, Tuple[float, Tuple]]:
        p_correct, p_predicted_correct = [], []
        for choice_score, answer_prob in zip(choice_scores, answer_probs):
            if np.isnan(choice_score).any():
                choice_score = np.array([0.5, 0.5])
            assert (sum(choice_score) - 1) < 1e-2
            assert (sum(answer_prob) - 1) < 1e-2

            if unambiguous_only:
                # this means we only exclude a case if it's ambiguous and the model is correct
                # in inferring if it's ambiguous
                if max(choice_score) <= 0.6 and max(answer_prob) <= 0.6:
                    continue

            p_correct.append(max(answer_prob))
            answer_index = np.argmax(answer_prob)
            if discretize:
                p_predicted_correct.append(
                    closest_value(list(range(0, 100, 4)), choice_score[answer_index] * 100) / 100)
            else:
                p_predicted_correct.append(choice_score[answer_index])

        mae = mean_absolute_error(p_correct, p_predicted_correct)

        return mae

def cross_entropy(predictions, targets):
    predictions = np.clip(predictions, 1e-3, None)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce
class CEEvaluator:
    def evaluate(
            self, choice_scores: List[List[float]], answer_probs: List[List[float]],
            discretize=False, unambiguous_only=False
    ) -> Union[float, Tuple[float, Tuple]]:
        p_correct, p_predicted_correct = [], []
        for choice_score, answer_prob in zip(choice_scores, answer_probs):
            if np.isnan(choice_score).any():
                choice_score = np.array([0.5, 0.5])

            assert (sum(choice_score) - 1) < 1e-2
            assert (sum(answer_prob) - 1) < 1e-2

            if unambiguous_only:
                if max(choice_score) <= 0.6 and max(answer_prob) <= 0.6:
                    continue

            p_correct.append(max(answer_prob))
            answer_index = np.argmax(answer_prob)
            # (48%, 52%) , (48%, 52%)
            if discretize:
                p_predicted_correct.append(
                    closest_value(list(range(0, 100, 4)), choice_score[answer_index] * 100) / 100)
            else:
                p_predicted_correct.append(choice_score[answer_index])

        ce = cross_entropy(np.array(p_predicted_correct), np.array(p_correct))

        return ce

class AuROCEvaluator:
    def evaluate(
            self, choice_scores: List[List[float]], answer_probs: List[List[float]],
            discretize=False, unambiguous_only=True
    ) -> Union[float, Tuple[float, Tuple]]:
        p_correct, p_predicted_correct = [], []
        for choice_score, answer_prob in zip(choice_scores, answer_probs):
            if np.isnan(choice_score).any():
                choice_score = np.array([0.5, 0.5])
            assert (sum(choice_score) - 1) < 1e-2
            assert (sum(answer_prob) - 1) < 1e-2

            if unambiguous_only:
                if max(answer_prob) <= 0.6:
                    continue

            # True binary labels. If labels are not either {-1, 1} or {0, 1}
            p_correct.append(np.argmax(answer_prob))
            # Target scores, can either be probability estimates of the positive class
            p_predicted_correct.append(choice_score[0])

        # label is ['yes', 'no'], we treat positive class as 0 (yes)
        fpr, tpr, thresholds = roc_curve(p_correct, p_predicted_correct, pos_label=0)
        roc_auc = auc(fpr, tpr)

        return roc_auc


if __name__ == "__main__":
    ...

    choice_scores = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.5, 0.5]]
    answer_probs = [[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.45, 0.55]]
    evaluator = CorrelationEvaluator()
    print(evaluator.evaluate(choice_scores, answer_probs))

    choice_scores = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.5, 0.5]]
    answer_probs = [[0.1, 0.9], [0.2, 0.8], [0.48, 0.52], [0.48, 0.52]]
    evaluator = RMSEEvaluator()
    print(evaluator.evaluate(choice_scores, answer_probs))

    choice_scores = [[0.9, 0.1], [0.3, 0.7], [0.5, 0.5], [0.5, 0.5]]
    answer_probs = [[0.9, 0.1], [0.2, 0.8], [0.48, 0.52], [0.48, 0.52]]
    evaluator = AuROCEvaluator()
    print(evaluator.evaluate(choice_scores, answer_probs))
