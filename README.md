# MoCa: Measuring Human-Language Model Alignment on Causal and Moral Judgment Tasks

We provide the dataset and code that can reproduce the table/experiment in the paper.

Please contact us if you have questions about usage or make a pull request! 

Important note: it only runs on Python 3.9+ (because we use extensive class annotations).

There are two intended usage of our codebase:
- Dataset Loading for LLM Evaluations
- Generate Average Marginal Causal Effect (AMCE) Analytics and Plotting

For this goal, we provide data loading code and analytics code. We also provide a simple interface to produce 
API calls to the LLMs. Please note that the LLM interface code will not be maintained (as we expect better tooling libraries
will appear over time), but the data loader code and analytics code will be maintained.

## Dataset Loading

## Analytics 

## Running LLMs

We provide a simple interface to do LLM API calls. The script does cache saving to a local file, which can be disabled. 

## LLM Credentials

In order to run OpenAI models, you need to first provide your credential as `credential.txt`.
We expect `credential.txt` to contain two lines:
```text
secrete key: xxxx
organization key: xxxx
```

For Anthropic models, we expect a credential file `anthro_credential.txt` that is formatted the same as above.

Note: Due to changes in Anthropic Claude APIs, our code (which relies on access to `claude-v1`) is no longer runnable.
