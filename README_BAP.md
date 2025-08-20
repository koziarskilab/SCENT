# Using Proxies in the Binding Affinity Pipeline

## Setup RGFN environment

See [README.md](README.md).

## Install binding affinity pipeline (BAP) source code

```bash
pip install git+https://github.com/koziarskilab/BindingAffinityPipeline.git@feature/rgfn_proxy
```

## Test random scoring proxy

To run a simple test that the BAP source code is installed properly, you can run the following test run:

```bash
python train.py --cfg configs/scent_test.gin
```

This will use the [`RandomScorer`](https://github.com/koziarskilab/BindingAffinityPipeline/blob/feature/rgfn_proxy/bap/scoring/random_scorer.py) proxy inside the `bap` package to score molecules for the training job. If you see that the training job is running and the scores are being generated, then the BAP source code is installed properly.

## Test Chai docker proxy

I've also added support for the Chai docking proxy via the `bap` package, but I haven't tested if it works yet. To do so, you will need to run the following command (after making sure that all Chai dependencies are installed):

```bash
python train.py --cfg configs/scent_chai_proxy.gin
```

If you take a look at this configuration file, you will see that it no longer uses the `ChaiDockerProxy` from the `rgfn` package, but instead uses the `ChaiDocker` class from the `bap` package. The new configuration for `bap` proxies can be found under `configs/proxies/binding_affinity/...`. This directory contains config files for all dockers, scorers, preparators, and receptors that are available in the `bap` package. I have simply copy/pasted the gin files from `bap` to `rgfn` and made some minor modifications to remove any dataset references.

@Evan, please can you also take a look at the implementation of the `ChaiDocker` class inside this [PR](https://github.com/koziarskilab/BindingAffinityPipeline/pull/83/commits/015db3b173a96effb13268778784e55516a2b9f5)? I am not 100% sure that it is up-to-date with all the fixes and modifications you made the the `ChaiDockerProxy` class in the `rgfn` package (which we will want to deprecate in the future).

## Test binding affinity pipeline with decentralized inference server

All the proxies in the `bap` package that implement the [`ProxyBase` class](https://github.com/koziarskilab/BindingAffinityPipeline/blob/feature/rgfn_proxy/bap/proxy/proxy_base.py) can be used with the decentralised inference server. This means that you can run the `bap` proxies in the same way as the decentralized `rgfn` proxies, by using the `rgfn/shared/proxies/server_proxy/manager.py` script. The only difference is that you will need to use the `configs/proxies/binding_affinity/...` configuration files instead of the `configs/proxies/target/...` files.

To test this, @Evan, please follow the steps below:

1. Modify the `configs/proxies/binding_affinity/proxy_base.gin` file to set `Proxy.use_proxy_server = True` on line 26. This will tell the `bap` proxies to call the decentralized inference server rather than running locally on the same node.

2. Launch the decentralised inference server with the following command:

```bash
conda activate rgfn
python rgfn/shared/proxies/server/manager.py stop # Stop any previous server instances
python rgfn/shared/proxies/server/manager.py start \
  --proxy-cfg configs/proxies/binding_affinity/chai_docker.gin \
  --slurm-cfg gpu_template.sh \
  --num-servers 2 \
  --base-port 5555

# Make sure that the Chai proxy servers are running properly
python rgfn/shared/proxies/server/manager.py test
```

3. Launch the SCENT training job with the following command:

```bash
# Launch the RGFN training job with Chai proxy
sbatch run.sh --cfg configs/scent_chai_proxy.gin
```
