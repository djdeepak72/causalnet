## causalnet

`causalnet` is a python package that uses the dragonnet architecture( adapted from Shi, et. al. : https://github.com/claudiashi57/dragonnet) in the backend to build causal inference models for your structured data.

### What makes this package easy to use?
* Helps with data preparation by creating a custom data object that is universal across all functions within this package.
* Creates embeddings out of your categorical variables automatically within the function that builds & trains the causal inference models. The packae also lets the user build their embeddings separately if needed, offering the users to change the embedding structure as required
* Dragonnet is a 2 stage neural network & this package lets the user train the entire network in one go, or stage by stage if the users needs more flexibility in evaluating their results between stages.
* Loads of causal inference metrics are specified in the package to evaluate your models(look at `causal_inference_metrics.py` file)
* Finally when all your models are trained, the package can help you interpret your results using the `treatment_effects()` function that estimates in percent that the one treatment group performs better than the other.

### Installation
From a command terminal run:

`pip install git+https://github.com/djdeepak72/causalnet.git`

OR just clone the repo into your local drive , cd into it & then run:

`pip install .`

For further instructions and examples about how to use please see the [tutorial](tutorials) in the tutorials/ folder.