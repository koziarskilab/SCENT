# Using acquisition functions from PSALM

For proxies models that are Bayesian or approximate Bayesian, you can use the acquisition functions from the `psalm` package to produce a scalar proxy score using the Bayesian posterior predictive distribution. This is useful for training RGFN/SCENT models that need to balance exploration and exploitation during sampling.

## Setup RGFN environment

See [README.md](README.md).

## Install PSALM source code

```bash
pip install git+https://github.com/koziarskilab/psalm.git@unify/rgfn
```

## Usage

For a model to use the acquisition functions from `psalm`, it must implement the [`BayesianProxy`](rgfn/shared/proxies/bayesian_proxy.py) interface. This interface accepts a [`BayesianModel`](rgfn/shared/proxies/bayesian_proxy.py) and an `AcquisitionFn` from `psalm`. The idea is that the `BayesianModel` can output a posterior predictive distribution over the function space, and the `AcquisitionFn` can be used to compute a scalar proxy value from this distribution.

See the [`BayesianClppProxy`](rgfn/gfns/reaction_gfn/proxies/clp_proxy.py) for an example of how to use and implement this interface. The `BayesianClppProxy` uses the `UpperConfidenceBound` acquisition function from `psalm` to compute the proxy value based on the posterior predictive distribution of an `ApproximateTanimotoGP` model.

TODO: Adapt GNEprop proxy to use this interface.
