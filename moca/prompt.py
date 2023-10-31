from moca.data import CausalDataset, MoralDataset, Example, JudgmentDatasetSchema, Sentence, AnnotationUtil
from promptsource.templates import DatasetTemplates, Template
from moca.thought_as_text_translator import AbstractExample

import string
import os
from os.path import join as pjoin
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from moca.data import FactorUtils

def rephrase_victim(victim):
    if victim is None:
        return victim

    words = victim.split()
    if words[0] in ['the']:
        if words[1] in ['one', 'two', 'five', 'third']:
            return ' '.join(words[1:])
        else:
            return 'a ' + ' '.join(words[1:])
    return victim

def remove_victm_quant(victim):
    if victim is None:
        return victim

    words = victim.split()
    if words[0] in ['one', 'two', 'five', 'third', 'a']:
        return ' '.join(words[1:])
    else:
        return victim

@dataclass
class FormattedPrompt:
    prompt: str
    choices: List[str]
    answer: str
    answer_index: int

def example_to_promptsource_dict(example: Example, name: string, description: string) -> Dict:
    return {
        'story': example.story,
        'question': example.question,
        'label': JudgmentDatasetSchema.answer_to_label[example.answer],
        'name': name,
        'description': description
    }

class JudgmentPrompt(object):
    def __init__(self, exp1_prompt):
        txt = open(exp1_prompt).read()
        self.prompt = Template(name="exp1_prompt", jinja=txt, reference="",
                                answer_choices="Yes|||No")

    def apply(self, example: Example, name: string='', description: string='') -> FormattedPrompt:
        # (prompt, answer)
        prompt, answer = self.prompt.apply(example_to_promptsource_dict(example, name, description))
        return FormattedPrompt(prompt=prompt, choices=JudgmentDatasetSchema.answer_choices,
                               answer=answer, answer_index=JudgmentDatasetSchema.answer_to_label[answer])

class AbstractJudgmentPrompt(object):
    def __init__(self, exp1_prompt):
        txt = open(exp1_prompt).read()
        self.prompt = Template(name="exp1_prompt", jinja=txt, reference="",
                                answer_choices="Yes|||No")

    def apply(self, ex: AbstractExample) -> FormattedPrompt:
        # (prompt, answer)
        data = {'story': ex.story, 'question': ex.question, 'label': JudgmentDatasetSchema.answer_to_label[ex.answer]}
        prompt, answer = self.prompt.apply(data)
        return FormattedPrompt(prompt=prompt, choices=JudgmentDatasetSchema.answer_choices,
                               answer=answer, answer_index=JudgmentDatasetSchema.answer_to_label[answer])

    def apply2(self, story: str, question: str, answer: str) -> FormattedPrompt:
        # (prompt, answer)
        data = {'story': story, 'question': question, 'label': JudgmentDatasetSchema.answer_to_label[answer]}
        prompt, answer = self.prompt.apply(data)
        return FormattedPrompt(prompt=prompt, choices=JudgmentDatasetSchema.answer_choices,
                               answer=answer, answer_index=JudgmentDatasetSchema.answer_to_label[answer])

class CausalJudgmentPrompt(JudgmentPrompt):
    def __init__(self, exp1_prompt='./prompts/exp1_causal_prompt.jinja'):
        super().__init__(exp1_prompt)

class MoralJudgmentPrompt(JudgmentPrompt):
    def __init__(self, exp1_prompt='./prompts/exp1_moral_prompt.jinja'):
        super().__init__(exp1_prompt)

class CausalAbstractJudgmentPrompt(AbstractJudgmentPrompt):
    def __init__(self, exp1_prompt='./prompts/exp1_causal_prompt.jinja'):
        super().__init__(exp1_prompt)

class MoralAbstractJudgmentPrompt(AbstractJudgmentPrompt):
    def __init__(self, exp1_prompt='./prompts/exp1_moral_prompt.jinja'):
        super().__init__(exp1_prompt)

def factor_to_promptsource_dict(sent: Sentence, label: int, norm: str="") -> Dict:
    return {
        'text': sent.text,
        'victim': remove_victm_quant(rephrase_victim(sent.victim)),
        'norm': norm,
        'factor': sent.annotation.factor,
        'label': label
    }

class FactorPrompt(object):
    def __init__(self, name, prompt_file, answer_choices):
        txt = open(prompt_file).read()
        self.prompt = Template(name=name, jinja=txt, reference="",
                               answer_choices=answer_choices)

        assert FactorUtils

    def apply(self, example: Sentence, label: int, norm: str="") -> FormattedPrompt:
        # (prompt, answer)
        prompt, answer = self.prompt.apply(factor_to_promptsource_dict(example, label, norm))
        return FormattedPrompt(prompt=prompt, answer=answer, choices=eval(f"FactorUtils.{example.annotation.factor}_answers"),
                               answer_index=label)

class FactorPromptLoader(object):
    factors: List[str]
    anno_utils: AnnotationUtil

    def load_prompts(self, prompt_folder: str) -> Dict[str, FactorPrompt]:
        prompts = {}
        for key in self.factors:
            file = pjoin(prompt_folder, key + '_prompt.jinja')
            answers = [eval(f"FactorUtils.{key}_answers_map")[a] for a in self.anno_utils.actual_hierarchy[key]]
            prompts[key] = FactorPrompt(key, file, "|||".join(answers))

        return prompts

class CausalFactorPrompt(FactorPromptLoader):
    def __init__(self, prompt_folder: str="./prompts/exp3_prompts/causal/",
                 anno_utils: Optional[AnnotationUtil]=None):
        self.factors = ['causal_structure', 'agent_awareness', 'event_normality', 'action_omission',
                        'time', 'norm_type']
        self.anno_utils = anno_utils
        self.prompts = self.load_prompts(prompt_folder)

    def apply(self, example: Example) -> Tuple[List[str], List[FormattedPrompt]]:
        """
        This has to apply on an example level, because "norm" needs to be collected within the story.
        :return:
        """
        factor_categories, sent_prompts = [], []
        for sent in example.annotated_sentences:
            factor = sent.annotation.factor
            factor_answer_map = eval(f"FactorUtils.{factor}_answers_map")
            factor_answers = eval(f"FactorUtils.{factor}_answers")

            label = factor_answers.index(factor_answer_map[sent.annotation.value])

            if factor != "event_normality":
                sent_prompts.append(self.prompts[factor].apply(sent, norm="", label=label))
            else:
                norms = [s.text for s in example.annotated_sentences if s.annotation.factor == 'norm_type']
                norm = ' '.join(norms).capitalize()
                if norm == '':
                    norm = 'The usual standard societal norms, cultural practices and morality.'
                elif norm[-1] not in string.punctuation:
                    norm = norm + '.'
                sent_prompts.append(self.prompts[factor].apply(sent, norm=norm, label=label))

            factor_categories.append(factor)

        return factor_categories, sent_prompts

class MoralFactorPrompt(FactorPromptLoader):
    def __init__(self, prompt_folder: str="./prompts/exp3_prompts/moral/",
                 anno_utils: Optional[AnnotationUtil]=None):
        self.factors = ['personal_force', 'causal_role', 'evitability', 'beneficiary',
                        'locus_of_intervention']
        self.anno_utils = anno_utils
        self.prompts = self.load_prompts(prompt_folder)

    def apply(self, example: Example) -> Tuple[List[str], List[FormattedPrompt]]:
        """
        This has to apply on an example level, because "norm" needs to be collected within the story.
        :return:
        """
        factor_categories, sent_prompts = [], []
        for sent in example.annotated_sentences:
            factor = sent.annotation.factor
            factor_answer_map = eval(f"FactorUtils.{factor}_answers_map")
            factor_answers = eval(f"FactorUtils.{factor}_answers")

            label = factor_answers.index(factor_answer_map[sent.annotation.value])

            sent_prompts.append(self.prompts[factor].apply(sent, label=label))

            factor_categories.append(factor)

        return factor_categories, sent_prompts


if __name__ == '__main__':
    ...
    # cd = CausalDataset()
    # md = MoralDataset()

    # print(cd[0])

    # cjp = CausalJudgmentPrompt(exp1_prompt='./prompts/exp1_causal_prompt.jinja')
    # print(cjp.apply(cd[0]).prompt)

    # cfp = CausalFactorPrompt(anno_utils=cd.anno_utils)
    # cfp.load_prompts("./prompts/exp3_prompts/causal/")
    #
    # factor_categories, factor_instances = cfp.apply(cd[140])
    # for p in factor_instances:
    #     print(p)

    # mfp = MoralFactorPrompt(anno_utils=md.anno_utils)
    # mfp.load_prompts('./prompts/exp3_prompts/moral/')
    #
    # factor_categories, factor_instances = mfp.apply(md[20])
    # for p in factor_instances:
    #     print(p)

    # for p in cfp.apply(cd[2]):
    #     print(p.prompt)
