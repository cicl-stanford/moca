from setuptools import setup

setup(
    name='moca',
    version='0.1.0',
    description="Dataset to evaluate LLM-Human Alignment on Causal and Moral Tasks",
    author="windweller",
    packages=["moca"],
    install_requires=[
        "openai~=0.18.0",
        "spacy~=3.2.4",
        "requests~=2.25.1",
        "pandas~=1.3.2",
        "tqdm~=4.61.2",
        "sklearn~=0.0",
        "scikit-learn~=1.0.2",
        "torch~=1.9.0",
        "myers~=1.0.1",
        "numpy~=1.19.5",
        "scipy~=1.7.1",
        "transformers~=4.18.0",
        "termcolor~=1.1.0",
        "ipython~=8.1.1",
        "stanza",
        "promptsource",
        "nltk",
        "anthropic",
        "together"
    ],
    python_requires='>=3.9',
)