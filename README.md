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

(Coming soon)

## LLM Credentials

In order to run OpenAI models, you need to first provide your credential as `credential.txt`.
We expect `credential.txt` to contain two lines:
```text
secrete key: xxxx
organization key: xxxx
```

For Anthropic models, we expect a credential file `anthro_credential.txt` that is formatted the same as above.

Note: Due to changes in Anthropic Claude APIs, our code (which relies on access to `claude-v1`) is no longer runnable.

## Bugs

Please report bugs as issues on GitHub.

Contact: anie@stanford.edu