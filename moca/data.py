import json
import os
import string
from collections import defaultdict
from dataclasses import dataclass, asdict
from os.path import join as pjoin
from typing import List, NamedTuple, Optional, Dict

import numpy as np
import pandas as pd
import spacy


######################## MoCa Data ########################
@dataclass
class JsonSerializable:
    @property
    def __dict__(self):
        """
        get a python dictionary
        """
        return asdict(self)

    @property
    def json(self):
        """
        get the json formated string
        """
        return json.dumps(self.__dict__)


Annotation = NamedTuple("Annotation",
                        [("factor", str), ("value", str)])  # e.g., Annotation("Causal Structure", "Conjunctive")


@dataclass
class Sentence(JsonSerializable):
    text: str
    victim: Optional[str]  # recipient of action
    annotation: Annotation


@dataclass
class Example(JsonSerializable):
    story: str
    question: str
    answer: str
    answer_dist: List[float]  # (P(Yes), P(No))
    transcribed_answer: str
    annotated_sentences: List[Sentence]
    is_ambiguous: bool
    individual_votes: List[int]  # 0: No, 1: Yes


@dataclass
class AnnotationUtil(object):
    legend: Optional[Dict]
    reverse_legend: Optional[Dict]
    entity_to_seg_marking: Dict[str, List[str]]
    actual_hierarchy: Dict[str, List[str]]
    seg_marking_to_prompt_category: Dict

    @classmethod
    def load(cls, json_file: str):
        data = json.load(open(json_file))
        return cls(legend=data['legend'],
                   reverse_legend=data['reverse_legend'],
                   entity_to_seg_marking=data['entity_to_seg_marking'],
                   actual_hierarchy=data['actual_hierarchy'],
                   seg_marking_to_prompt_category=data['seg_marking_to_prompt_category'])

    def to_json(self):
        return {
            "legend": self.legend,
            "reverse_legend": self.reverse_legend,
            "entity_to_seg_marking": self.entity_to_seg_marking,
            "actual_hierarchy": self.actual_hierarchy,
            "seg_marking_to_prompt_category": self.seg_marking_to_prompt_category
        }


class InterfaceFactory:
    @staticmethod
    def to_exp1(example: Example) -> str:
        return example.story + ' ' + example.question

    @staticmethod
    def to_exp2(example: Example):
        return NotImplementedError

    @staticmethod
    def to_exp3(example: Example):
        return NotImplementedError


class JudgmentDatasetSchema:
    answer_choices: List[str] = ['Yes', 'No']
    answer_to_label: Dict[str, int] = {'No': 1, 'Yes': 0}


######################## Tool Objects ########################

class Tokenizer(object):
    def __init__(self):
        self.spacy_interface = None

    def __call__(self, text: str) -> spacy.language.Doc:
        # lazy init
        if self.spacy_interface is None:
            self.spacy_interface = spacy.load('en_core_web_sm')
        return self.spacy_interface(text)


######################## Usage for exp2 ########################

class FactorUtils:
    causal_structure_answers = ['Conjunctive', 'Disjunctive']
    causal_structure_answers_map = {'CS_1_Conjunctive': 'Conjunctive',
                                    'CS_1_Disjunctive': 'Disjunctive'}
    causal_structure_answers_map_reverse = {v.lower(): k for k, v in causal_structure_answers_map.items()}

    agent_awareness_answers = ['Aware', 'Unaware']
    agent_awareness_answers_map = {'AK_2_Agent_Unaware': 'Unaware',
                                   'AK_2_Agent_Aware': "Aware"}
    agent_awareness_answers_map_reverse = {v.lower(): k for k, v in agent_awareness_answers_map.items()}

    event_normality_answers = ['Normal', 'Abnormal']
    event_normality_answers_map = {'EC_4_Abnormal_Event': 'Abnormal',
                                   'EC_4_Normal_Event': 'Normal'}
    event_normality_answers_map_reverse = {v.lower(): k for k, v in event_normality_answers_map.items()}

    action_omission_answers = ['Action', 'Omission']
    action_omission_answers_map = {'EC_4_Omission': 'Omission', 'EC_4_Action': 'Action'}
    action_omission_answers_map_reverse = {v.lower(): k for k, v in action_omission_answers_map.items()}

    time_answers = ['Early Cause', 'Same Time Cause', 'Late Cause']
    time_answers_map = {'EC_4_Same_Time_Cause': 'Same Time Cause', 'EC_4_Early_Cause': 'Early Cause',
                        'EC_4_Late_Cause': 'Late Cause'}
    time_answers_map_reverse = {v.lower(): k for k, v in time_answers_map.items()}

    norm_type_answers = ['Prescriptive Norm', 'Statistical Norm']
    norm_type_answers_map = {'Norm_3_Prescriptive_Norm': 'Prescriptive Norm',
                             'Norm_3_Statistics_Norm': 'Statistical Norm'}
    norm_type_answers_map_reverse = {v.lower(): k for k, v in norm_type_answers_map.items()}

    personal_force_answers = ['Personal', 'Impersonal']
    personal_force_answers_map = {'personal': 'Personal',
                                  'impersonal': 'Impersonal'}  # "raw text": "label"
    personal_force_answers_map_reverse = {v.lower(): k for k, v in personal_force_answers_map.items()}

    # instrumental_accidental_answers = ['Instrumental', 'Accidental']  # ['Yes', 'No']
    # instrumental_accidental_answers_map = {'instrumental': 'Yes', 'accidental': 'No'}
    instrumental_accidental_answers = ['Means', 'Side Effect']
    instrumental_accidental_answers_map = {'instrumental': 'Means', 'accidental': 'Side Effect'}
    instrumental_accidental_answers_map_reverse = {v.lower(): k for k, v in instrumental_accidental_answers_map.items()}

    causal_role_answers = ['Means', 'Side Effect']
    causal_role_answers_map = {'instrumental': 'Means', 'accidental': 'Side Effect'}
    causal_role_answers_map_reverse = {v.lower(): k for k, v in causal_role_answers_map.items()}

    evitability_answers = ['Avoidable', 'Inevitable']
    evitability_answers_map = {'avoidable': 'Avoidable', 'inevitable': 'Inevitable'}
    evitability_answers_map_reverse = {v.lower(): k for k, v in evitability_answers_map.items()}

    beneficiary_answers = ['Self-beneficial', 'Other-beneficial']
    beneficiary_answers_map = {'self beneficial': "Self-beneficial", 'other beneficial': 'Other-beneficial'}
    beneficiary_answers_map_reverse = {v.lower(): k for k, v in beneficiary_answers_map.items()}

    locus_of_intervention_answers = ['Agent of harm', 'Patient of harm']
    locus_of_intervention_answers_map = {'influence agent': 'Agent of harm', 'influence patient': 'Patient of harm'}
    locus_of_intervention_answers_map_reverse = {v.lower(): k for k, v in locus_of_intervention_answers_map.items()}


######################## Data Class Objects ########################

class RawDataset(object):
    examples: List[Example]

    def process_raw_data(self, raw_data: Dict, crowd_src_label_file: str) -> List[Example]:
        raise NotImplementedError

    def to_json(self) -> List[str]:
        long_json = []
        for ex in self.examples:
            long_json.append(ex.json)
        return long_json

    def process_crowd_label(self, crowd_src_label_file: str) -> (List[float], List[bool]):
        # True (1) -> Yes, False (0) -> No
        crowd_src_labels = pd.read_csv(crowd_src_label_file)
        time_columns = [f'user{i} time' for i in range(1, 25 + 1)]
        res_columns = [f'user{i} res' for i in range(1, 25 + 1)]

        preds_probs = crowd_src_labels[res_columns].transpose().mean().to_numpy()
        preds = preds_probs >= 0.5

        individual_votes = crowd_src_labels[res_columns].to_numpy()

        return individual_votes.tolist(), preds_probs.tolist(), preds.tolist()


class RawCausalDataset(RawDataset):
    def __init__(self, json_file: str = "../data/causal_prompt_w_questions.json",
                 annotation_root_path: str = "../data/causal_annotations_raw",
                 crowd_src_label_file: str = "../data/prolific_labels/causal_questions.csv",
                 member='cactus'):
        """
        :param member: "cactus" is our concensus annotation.
                       "hd" is our second annotator to measure inter-annotater agreement.
        """
        raw_data = json.load(open(json_file))
        self.examples = self.process_raw_data(raw_data, crowd_src_label_file)

        self.anno_utils = self.prep_annotation_processing(annotation_root_path)
        self.load_annotations(annotation_root_path, member)

    def process_raw_data(self, raw_data: Dict, crowd_src_label_file: str) -> List[Example]:

        # Load crowd-sourced labels
        individual_votes, preds_probs, preds = self.process_crowd_label(crowd_src_label_file)

        examples: List[Example] = []
        for i, item in enumerate(raw_data["examples"]):
            trans_answer = "Yes" if item["target_scores"]["Yes"] == 1 else "No"
            crowd_answer = "Yes" if preds[i] == 1 else "No"
            is_ambiguous = max(preds_probs[i], 1 - preds_probs[i]) <= 0.6
            ex = Example(story=item["input"],
                         question=item['question'],
                         transcribed_answer=trans_answer,
                         answer=crowd_answer,
                         answer_dist=[preds_probs[i], 1 - preds_probs[i]],
                         annotated_sentences=[],
                         is_ambiguous=is_ambiguous,
                         individual_votes=individual_votes[i])
            examples.append(ex)
        return examples

    def prep_annotation_processing(self, annotation_root_path: str) -> AnnotationUtil:
        legend_path = pjoin(annotation_root_path, "Moca_public/annotations-legend.json")
        legend = json.load(open(legend_path))
        reverse_legend = {v: k for k, v in legend.items()}
        entity_to_seg_marking = {
            'Event_Cause': ['EC_4_Action', 'EC_4_Omission',
                            'EC_4_Normal_Event', 'EC_4_Abnormal_Event',
                            'EC_4_Early_Cause', 'EC_4_Same_Time_Cause', 'EC_4_Late_Cause'],
            'Norm': ['Norm_3_Prescriptive_Norm', 'Norm_3_Statistics_Norm'],
            'Causal_Struct': ['CS_1_Disjunctive', 'CS_1_Conjunctive'],
            'Agent_Knowledge': ['AK_2_Agent_Aware', 'AK_2_Agent_Unaware'],
            'Outcome': ['Outcome_5_negative', 'Outcome_5_positive']
        }

        actual_hierarchy = {
            'action_omission': ['EC_4_Action', 'EC_4_Omission'],
            'event_normality': ['EC_4_Normal_Event', 'EC_4_Abnormal_Event'],
            'time': ['EC_4_Early_Cause', 'EC_4_Same_Time_Cause', 'EC_4_Late_Cause'],
            'norm_type': ['Norm_3_Prescriptive_Norm', 'Norm_3_Statistics_Norm'],
            'agent_awareness': ['AK_2_Agent_Aware', 'AK_2_Agent_Unaware'],
            'causal_structure': ['CS_1_Conjunctive', 'CS_1_Disjunctive'],
            'outcome': ['Outcome_5_negative', 'Outcome_5_positive']
        }

        seg_marking_to_prompt_category = {}
        for k, v_list in actual_hierarchy.items():
            for v in v_list:
                seg_marking_to_prompt_category[v] = k

        anno_utils = AnnotationUtil(seg_marking_to_prompt_category=seg_marking_to_prompt_category,
                                    legend=legend,
                                    reverse_legend=reverse_legend,
                                    entity_to_seg_marking=entity_to_seg_marking,
                                    actual_hierarchy=actual_hierarchy)

        return anno_utils

    def load_annotations(self, annotation_root_path: str, member='cactus') -> None:
        assert member in ['cactus', 'hd']
        ignore_ids = [68, 69, 70, 71, 78, 79] + list(range(80, 87 + 1)) + list(range(90, 95 + 1)) + \
                     [96, 97] + [98, 99] + [14, 15]
        root_dir = pjoin(annotation_root_path, f"Moca_public/ann.json/members/{member}/pool/")

        story_id_to_data = {}
        for f_n in os.listdir(root_dir):
            story_id = int(f_n.split('-')[1].split('.')[0].split('_')[1])
            if story_id in ignore_ids:
                continue
            data = json.load(open(root_dir + f_n))
            story_id_to_data[story_id] = data

        story_id_to_segments = self.extract_annotations_to_tuples(story_id_to_data)

        for story_id, segs in story_id_to_segments.items():
            for seg in segs:
                factor, sent, value = seg
                self.examples[story_id].annotated_sentences.append(Sentence(text=sent,
                                                                            victim=None,
                                                                            annotation=Annotation(factor, value)))

    def extract_annotations_to_tuples(self, story_id_to_data, with_outcome=False) -> Dict:
        # story_id_to_data: {id(int): {data}}
        # with_outcome: not when we infer tags...but only in the end
        # example: [('norm_type', 'Claire uses it for schoolwork.', 'Norm_3_Statistics_Norm'), ...]
        story_id_to_segments = {}
        for story_id, data in story_id_to_data.items():
            tags_for_seg = set()
            entities = data['entities']
            combined_text = defaultdict(list)
            for ann_seg in entities:
                e = ann_seg['classId']
                text = ann_seg['offsets'][0]['text']
                combined_text[e].append(text)

            for e, text_list in combined_text.items():
                text = self.combine_and_normalize_list_of_text(text_list)
                seg_type = self.anno_utils.legend[e]
                if seg_type == 'Outcome' and not with_outcome:
                    continue
                for tag in self.anno_utils.entity_to_seg_marking[seg_type]:
                    # check if this tag is in metas; and if it's True
                    # print(tag, reverse_legend[tag])
                    exists = data['metas'].get(self.anno_utils.reverse_legend[tag], {'value': False})['value']
                    if exists:
                        tags_for_seg.add((self.anno_utils.seg_marking_to_prompt_category[tag], text, tag))

            story_id_to_segments[story_id] = list(tags_for_seg)

        return story_id_to_segments

    def combine_and_normalize_list_of_text(self, text_list: List) -> str:
        mod = []
        for t in text_list:
            if t[-1] not in string.punctuation:
                t = t + '.'
            mod.append(t.capitalize())
        return " ".join(mod)

    def return_label_statistics(self, story_id_to_segments) -> Dict[str, Dict[str, int]]:
        stats = {k: [0] * len(v) for k, v in self.anno_utils.actual_hierarchy.items()}
        for story_id, tups in story_id_to_segments.items():
            for tup in tups:
                hierarchy_tag = tup[0]
                stats[hierarchy_tag][self.anno_utils.actual_hierarchy[hierarchy_tag].index(tup[-1])] += 1

        stats_w_name = {}
        for k, v in self.anno_utils.actual_hierarchy.items():
            stats_w_name[k] = {}
            for i, num in enumerate(stats[k]):
                stats_w_name[k][v[i]] = num

        return stats_w_name

    def display_stats(self, stats_w_name: Dict[str, Dict[str, int]]):
        # use it with self.return_label_statistics
        for cate, names in stats_w_name.items():
            print(cate.capitalize(), sum(names.values()))
            for name, num in names.items():
                print("", "--", name + ":", num)


class RawMoralDataset(RawDataset):
    def __init__(self, json_file: str = "../data/moral_dilemma_w_questions.json",
                 annotation_root_path: str = "../data/moral_annotations_raw",
                 crowd_src_label_file: str = "../data/prolific_labels/moral_questions.csv",
                 member: str = 'AN'):
        raw_data = json.load(open(json_file))
        self.examples = self.process_raw_data(raw_data, crowd_src_label_file)

        self.anno_utils = self.prep_annotation_processing()

        self.load_annotations(annotation_root_path, member=member)

    def load_annotations(self, annotation_root_path: str, member: str = 'AN') -> None:
        assert member in ['AN', 'MK']
        df = pd.read_csv(pjoin(annotation_root_path, f'moral dilemma - Segment Annotations {member}.csv'))
        story_id_to_segments = self.extract_annotations_to_tuples(df)

        for story_id, segs in story_id_to_segments.items():
            for seg in segs:
                factor, sent, value, victim = seg
                self.examples[story_id].annotated_sentences.append(Sentence(text=sent,
                                                                            victim=victim,
                                                                            annotation=Annotation(factor, value)))

    def process_raw_data(self, raw_data: Dict, crowd_src_label_file: str) -> List[Example]:

        individual_votes, preds_probs, preds = self.process_crowd_label(crowd_src_label_file)

        examples: List[Example] = []
        for i, item in enumerate(raw_data["examples"]):
            trans_answer = "Yes" if item["target_scores"]["Yes"] == 1 else "No"
            crowd_answer = "Yes" if preds[i] == 1 else "No"
            is_ambiguous = max(preds_probs[i], 1 - preds_probs[i]) <= 0.6
            ex = Example(story=item["input"],
                         question=item['question'],
                         transcribed_answer=trans_answer,
                         answer=crowd_answer,
                         answer_dist=[preds_probs[i], 1 - preds_probs[i]],
                         annotated_sentences=[],
                         is_ambiguous=is_ambiguous,
                         individual_votes=individual_votes[i])
            examples.append(ex)
        return examples

    def prep_annotation_processing(self) -> AnnotationUtil:
        entity_to_seg_marking = {
            'Locus of Intervention': ['influence agent', 'influence patient'],
            'Personal Force': ['personal', 'impersonal'],
            'Counterfactual Evitability': ['avoidable', 'inevitable'],
            'Causal Role': ['instrumental', 'accidental'],
            'Beneficence': ['self beneficial', 'other beneficial']
        }

        actual_hierarchy = {
            'locus_of_intervention': ['influence agent', 'influence patient'],
            'personal_force': ['personal', 'impersonal'],
            'evitability': ['avoidable', 'inevitable'],
            'causal_role': ['instrumental', 'accidental'],
            'beneficiary': ['self beneficial', 'other beneficial']
        }

        seg_marking_to_prompt_category = {}
        for k, v_list in actual_hierarchy.items():
            for v in v_list:
                seg_marking_to_prompt_category[v] = k

        anno_utils = AnnotationUtil(seg_marking_to_prompt_category=seg_marking_to_prompt_category,
                                    legend=None,
                                    reverse_legend=None,
                                    entity_to_seg_marking=entity_to_seg_marking,
                                    actual_hierarchy=actual_hierarchy)

        return anno_utils

    def extract_annotations_to_tuples(self, df: pd.DataFrame) -> Dict:
        # story_id_to_data: {id(int): {data}}
        # with_outcome: not when we infer tags...but only in the end
        story_id_to_segments = {}
        story_id = 0

        skipped_ids = [12, 13, 14, 15]  # these 4 stories don't have annotations

        for _, row in df.iterrows():

            tags_for_seg = set()

            if pd.isna(row['Tag 2']):
                t = row['Tag 1']
                text = row['Segment 1']
                tags_for_seg.add((self.anno_utils.seg_marking_to_prompt_category[t], text, t, None))
            else:
                for tag_idx in range(1, 4 + 1):
                    if tag_idx != 2:
                        t = row[f'Tag {tag_idx}']
                        text = row[f'Segment {tag_idx}']
                        harm_recepient = row['Harm Recepient']
                        tags_for_seg.add((self.anno_utils.seg_marking_to_prompt_category[t], text, t, None))
                    else:
                        t = row[f'Tag {tag_idx}']
                        text = row[f'Segment {tag_idx}']
                        harm_recepient = row['Harm Recepient']
                        tags_for_seg.add(
                            (self.anno_utils.seg_marking_to_prompt_category[t], row['story'], t, harm_recepient))

            story_id_to_segments[story_id] = list(tags_for_seg)
            story_id += 1
            while story_id in skipped_ids:
                story_id += 1

        return story_id_to_segments

    def extract_stories(self, df: pd.DataFrame) -> Dict:
        story_id_to_examples = {}
        story_id = 0
        for _, row in df.iterrows():
            story_id_to_examples[story_id] = (row['comment'], row['story'], row['answer'], row['Harm Recepient'])
            story_id += 1
        return story_id_to_examples

    def rephrase_question(self, q: spacy.language.Doc) -> str:
        """
        We rephrase so that we can ask questions about the factors, without being influenced by the "question".

        :param q:
        :return:
        """
        if str(q[0]) == 'Should':
            insert_idx = 0
            changed_verb = ""
            for i, token in enumerate(q[1:]):
                if token.pos_ == 'VERB' or token.pos_ == 'AUX':
                    insert_idx = i + 1
                    changed_verb = token.lemma_
                    break
            re_arranged_tokens = [t.text for t in q[1:insert_idx]] + \
                                 [q[0].text.lower(), changed_verb] + [t.text for t in q[insert_idx + 1:len(q) - 1]]
            sent = " ".join(re_arranged_tokens).capitalize() + '.'
            return sent
        elif str(q[0]) == 'Was':
            re_arranged_tokens = [q[1].text, q[0].text.lower()] + [t.text for t in q[2:len(q) - 1]]
            sent = " ".join(re_arranged_tokens).capitalize() + '.'
            return sent
        elif str(q[0]) == 'Do':
            insert_idx = 0
            changed_verb = ""
            for i, token in enumerate(q[1:]):
                if token.pos_ == 'VERB' or token.pos_ == 'AUX':
                    insert_idx = i + 1
                    changed_verb = token.lemma_
                    break

            re_arranged_tokens = [t.text for t in q[1:insert_idx]] + [changed_verb] + [t.text for t in
                                                                                       q[insert_idx + 1:len(q) - 1]]
            sent = ''
            for t in re_arranged_tokens:
                if t not in string.punctuation and t != "n't":
                    sent += ' ' + t
                else:
                    sent += t
            sent = sent.strip().capitalize() + '.'
            return sent


class AutoAseembleTrait(object):
    def auto_assemble(self, data: List[str]) -> List[Example]:
        examples = []
        for ex_raw in data:
            ex_raw = json.loads(ex_raw)
            annotated_sentences_raw = ex_raw['annotated_sentences']
            annotated_sentences = []
            for sent in annotated_sentences_raw:
                annotated_sentences.append(Sentence(text=sent['text'],
                                                    victim=sent['victim'],
                                                    annotation=Annotation(*sent['annotation'])))

            is_ambiguous = max(ex_raw['answer_dist'][0], ex_raw['answer_dist'][1]) <= 0.6
            ex = Example(story=ex_raw['story'],
                         question=ex_raw['question'],
                         transcribed_answer=ex_raw['transcribed_answer'],
                         answer=ex_raw['answer'],
                         answer_dist=ex_raw['answer_dist'],
                         annotated_sentences=annotated_sentences,
                         is_ambiguous=is_ambiguous,
                         individual_votes=ex_raw['individual_votes'])
            examples.append(ex)

        return examples


class AutoIndex(object):
    examples: List[Example]

    def __getitem__(self, key):
        return self.examples[key]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self) -> Example:
        if self.i < len(self.examples):
            ex = self.examples[self.i]
            self.i += 1
            return ex
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self.examples)


class AbstractDataset(AutoIndex, AutoAseembleTrait):
    examples: List[Example]


class CausalDataset(AbstractDataset):
    def __init__(self, json_file: str = "../data/causal_dataset_v1.json",
                 anno_json_file: str = '../data/causal_dataset_anno_utils.json'):
        data = json.load(open(json_file))
        self.examples = self.auto_assemble(data)
        self.anno_utils = AnnotationUtil.load(anno_json_file)

        self.ignore_ids = [68, 69, 70, 71, 78, 79] + list(range(80, 87 + 1)) + list(range(90, 95 + 1)) + \
                          [96, 97] + [98, 99] + [14, 15]  # examples with no annotations


class MoralDataset(AbstractDataset):
    def __init__(self, json_file: str = "../data/moral_dataset_v1.json",
                 anno_json_file: str = '../data/moral_dataset_anno_utils.json'):
        data = json.load(open(json_file))
        self.examples = self.auto_assemble(data)
        self.anno_utils = AnnotationUtil.load(anno_json_file)
        self.ignore_ids = [12, 13, 14, 15]  # examples with no annotations


######################## Interact with LLM ########################
@dataclass
class Token:
    text: str
    log_prob: float


@dataclass
class Interface:
    request: str
    response: Optional[List[Token]]


class Adapter:
    def __init__(self, model: str):
        self.model = self.init_model(model)

    def init_model(self, model: str):
        raise NotImplementedError

    def adapt(self, interface: Interface) -> Interface:
        interface.response = self.model(interface.request)
        return interface


class EvaluationFactory:
    @staticmethod
    def accuracy(example: Example, interface: Interface) -> float:
        raise NotImplementedError


######################## Usage for exp1 ########################
def create_examples(filename: str):
    raise NotImplementedError


if __name__ == "__main__":
    pass
