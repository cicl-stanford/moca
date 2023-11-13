# MoCa: Measuring Human-Language Model Alignment on Causal and Moral Judgment Tasks

Paper link: https://arxiv.org/abs/2310.19677

We provide the dataset and code that can reproduce the table/experiment in the paper.

Please contact us if you have questions about usage or make a pull request! 

Important note: it only runs on Python 3.9+ (because we use extensive class annotations).

There are two intended usage of our codebase:
- Dataset Loading for LLM Evaluations
- Generate Average Marginal Causal Effect (AMCE) Analytics and Plotting

For this goal, we provide data loading code and analytics code. We also provide a simple interface to produce 
API calls to the LLMs. Please note that the LLM interface code will not be maintained (as we expect better tooling libraries
will appear over time), but the data loader code and analytics code will be maintained.

## Setup

```bash
pip install -r requirements.txt
```

Then add the following to your `PYTHONPATH`:
```bash
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)" >> ~/.bashrc
```

## Dataset Loading

We provide very easy-to-use data loading code.

```python
from moca.data import CausalDataset, MoralDataset

cd = CausalDataset()
print(cd[0])

"""
Example(story="Lauren and Jane work for the same company.....", 
question='Did Jane cause the computer to crash?', answer='Yes', 
answer_dist=[0.92, 0.07999999999999996], transcribed_answer='Yes', 
annotated_sentences=[...], 
is_ambiguous=False, 
individual_votes=[True, True, True, True,...])
"""

md = MoralDataset()
print(md[0])
```

## Running LLMs

We provide a simple interface to do LLM API calls. The script does cache saving to a temporary file, which can be disabled.

The LLM code we provide covers major chat-based and completion-based models.

```python
from moca.adapter import GPT3Adapter
from moca.data import CausalDataset, MoralDataset
from moca.evaluator import AccuracyEvaluatorWithAmbiguity, AuROCEvaluator
from moca.prompt import CausalJudgmentPrompt, MoralJudgmentPrompt

engine = "text-davinci-003"
evaluator = AccuracyEvaluatorWithAmbiguity()
auroc_evaluator = AuROCEvaluator()

adapter = GPT3Adapter(engine=engine)

cd = CausalDataset()

prompt_file = "../moca/prompts/exp1_moral_prompt.jinja"

jp = CausalJudgmentPrompt(prompt_file)

all_instances, all_choice_scores, all_label_dist = [], [], []
for ex in cd:
    instance = jp.apply(ex)
    choice_scores = adapter.adapt(instance, method="yesno")
    all_choice_scores.append(choice_scores)
    all_label_dist.append(ex.answer_dist)
    all_instances.append(instance)

acc, conf_interval = evaluator.evaluate(all_choice_scores, all_label_dist)
auroc = auroc_evaluator.evaluate(all_choice_scores, all_label_dist)

print("AuROC: {:.3f}".format(auroc))
print("Accuracy: {:.3f} ({:.3f}, {:.3f})".format(acc, conf_interval[0], conf_interval[1]))
```

More details are shown in `experiments/r1_experiments.py` file.

We recommend reporting multiple metrics, as they can be quite different.

We propose to use `AuROC` as the main metric, and `Accuracy` as a secondary metric to compare between models.

## AMCE Analysis 

### Bar Plot

<img height="570" width="690" src="https://github.com/cicl-stanford/moca/blob/main/experiments/acme_moral_fig.png?raw=true"/>

To generate a figure like this, run the following:

**Step 1**: We first calculate the actual statistics from the experiment.

Assume we ran the experiment in `experiments/r1_experiments.py`, note that in order to run AMCE, we need to access
low level information saved in `.pkl` files.

If `r1_experiments.py` was run as is, then pickle files of each model is saved under:
`results/preds/exp1_{model}_causal_preds.pkl`.

Run `causal_effect_analysis.py` file to generate the AMCE statistics, including computing 95% CI
using Bootstrap.
Note that in the end function, the pickle files are provided as:
```python
category_to_pickle_map = {
        'single shot': "results/preds/exp1_text-davinci-002_moral_preds.pkl",
        'persona - average': "results/preds/exp5_persona_text-davinci-002_moral_preds.pkl",
        'persona - best': "results/preds/exp5_persona_text-davinci-002_moral_preds.pkl",
        "ape": "results/preds/exp6_ape_text-davinci-002_moral_preds.pkl"
    }
```
You can modify this to add your own models.

After running this, a result file that contains high-level statistics will be saved under a file like
`results/acme_causal_result.json`.

**Step 2**: We then generate the plot using the result file.

Run `experiments/amce_plot.py` file to generate the plot.

This file runs the following method:
```python
visualize_causal_acme('results/methods_acme_causal_result.json', name_mapping=method_name_mapping,
                          save_fig_name="acme_causal_methods.png")
```

### Radar Plot

## LLM Credentials

In order to run OpenAI models, you need to first provide your credential as `credential.txt`.
We expect `credential.txt` to contain two lines:
```text
secrete key: xxxx
organization key: xxxx
```

For Anthropic models, we expect a credential file `anthro_credential.txt` that is formatted the same as above.

Note: Due to changes in Anthropic Claude APIs, our code (which relies on access to `claude-v1`) is no longer runnable.

## Note on Cost

Running through the entire pipeline (without Social Simulacra) would cost roughly $1000 as of 2023-10-01.
This is because the stories are relatively long, and we average performance over 2 prompt templates.

Running Social Simulacra would significantly increase the cost, since we use 25 personas on 2 models, with 2 prompt templates,
which would be 100 runs of the entire dataset.

## Bugs

Please report bugs as issues on GitHub.

Contact: anie@stanford.edu