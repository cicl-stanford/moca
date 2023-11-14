import json
from jinja2 import Environment, FileSystemLoader
from moca.data import FactorUtils

# Create a template environment with the folder containing the template
env = Environment(loader=FileSystemLoader('./templates'))


def generate_causal_story_pages():
    template = env.get_template('causal_story.html')

    examples = json.load(open("../data/causal_dataset_v1.json"))
    for i, ex in enumerate(examples):
        d = json.loads(ex)

        factors = []
        for anno in d['annotated_sentences']:
            normalized_factor = " ".join([w.capitalize() for w in anno['annotation'][0].split("_")])
            factor_attr = eval(f"FactorUtils.{anno['annotation'][0]}_answers_map")[anno['annotation'][1]]
            factors.append(f"{normalized_factor}: {factor_attr}")
        factors = sorted(factors)

        data = {
            "prev_story_id": max(1, i),
            "next_story_id": min(len(examples), i + 2),
            "story_id": i + 1,
            "story_text": d['story'],
            "story_question": d['question'],
            "answer": d['answer'],
            "p_yes": "P(Yes): " + "{:.2f}".format(d['answer_dist'][0]),
            "p_no": "P(No): " + "{:.2f}".format(d['answer_dist'][1]),
            "ambiguous": d['is_ambiguous'],
            "factors": factors
        }

        rendered_html = template.render(data)
        with open(f"./causal_output/{i + 1}.md", 'w') as f:
            f.write(rendered_html)


def generate_moral_story_pages():
    template = env.get_template('moral_story.html')
    examples = json.load(open("../data/moral_dataset_v1.json"))
    for i, ex in enumerate(examples):
        d = json.loads(ex)

        factors = []
        for anno in d['annotated_sentences']:
            normalized_factor = " ".join([w.capitalize() for w in anno['annotation'][0].split("_")])
            factor_attr = eval(f"FactorUtils.{anno['annotation'][0]}_answers_map")[anno['annotation'][1]]
            factors.append(f"{normalized_factor}: {factor_attr}")
        factors = sorted(factors)

        data = {
            "prev_story_id": max(1, i),
            "next_story_id": min(len(examples), i + 2),
            "story_id": i + 1,
            "story_text": d['story'],
            "story_question": d['question'],
            "answer": d['answer'],
            "p_yes": "P(Yes): " + "{:.2f}".format(d['answer_dist'][0]),
            "p_no": "P(No): " + "{:.2f}".format(d['answer_dist'][1]),
            "ambiguous": d['is_ambiguous'],
            "factors": factors
        }

        rendered_html = template.render(data)
        with open(f"./moral_output/{i + 1}.md", 'w') as f:
            f.write(rendered_html)


if __name__ == '__main__':
    # generate_causal_story_pages()
    generate_moral_story_pages()