# Decentralised Proxy Server

For some of the proxies such as Chai which are GPU heavy and require a lot of compute to process a single molecule, we use a decentralised proxy server to offload the reward computation to a separate set of nodes. To use this service, you need a proxy that implements the [`ServerProxy`](api.py) interface such as our [`ChaiDockerProxy`](../../../gfns/reaction_gfn/proxies/chai_proxy/chai_docker.py) module.


## Setup

To set up the decentralised proxy server, you need to install additional dependencies. You can do this by running the following:

```bash
```

Further, to use the Chai docking proxy, you need to setup a separate conda environment. Make sure to match the environment name in `configs/proxies/chai_docker.gin`.

```bash
conda create -n chai python=3.12
conda activate chai
pip install chai_lab==0.5.2
```

Finally, make sure that the environment name in line 28 of `external/chai/run_chai.py` matches the name of the main `rgfn` conda environment (or what you named it).


## Usage

### Setting up the Proxy Server

We assume that you are using SLURM to manage your compute nodes. The first step is to modify the proxy server config to choose which proxy you want to use. For this, modify the `server.gin` file at `configs/proxies/server.gin` by replacing the `include ...` line to point to the gin configuration file for the proxy you want to use. By default, it is set to use `ChaiDockerProxy`.

Next, you should modify the `slurm_template.sh` file at `rgfn/shared/proxies/server_proxy/slurm_template.sh` to set the correct parameters for each proxy server node that will be launched. Crucially, you want to make sure that the allocated resources (e.g., GPUs, CPUs, memory) and time limits are appropriate for the proxy and training job that you will run. The default template is setup for a single GPU and CPU per node with 8GB of process memory and a time limit of 1 week. You must adjust these parameters according to your use case.

Finally, you need to start the server nodes which will run the proxy and listen for requests. To do this, you can run the manager script provided in `rgfn/shared/proxies/server_proxy`.

```bash
python rgfn/shared/proxies/server_proxy/manager.py start \
    --num-servers 4 \
    --base-port 5555 \
    --cfg configs/proxies/server.gin
```

If you encounter an error at this step, you might need to stop any existing server processes that are running on the same port. You can stop nodes launched using the manager script with:

```bash
python rgfn/shared/proxies/server_proxy/manager.py stop
```
You can also check the status of the servers with:

```bash
python rgfn/shared/proxies/server_proxy/manager.py status
```

You can use `squeue --user $USER` to check the status of the SLURM jobs once the servers are started. The output should show the jobs running on their own nodes.


### Training RGFN with a Proxy Client

Now that the server is running, we want to launch an RGFN training run that interacts with the proxy server to compute rewards in a distributed manner on the individual nodes. To do this, all you need to do is launch a regular RGFN training run using the same proxy configuration that you `include`d in the server config. For example, if you are using the `ChaiDockerProxy`, you should also use the `ChaiDockerProxy` in your RGFN training config.
