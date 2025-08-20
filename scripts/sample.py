# Description

import argparse
from pathlib import Path

import gin
from tqdm import tqdm
from utils import *

from gin_config import get_time_stamp
from rgfn.api.sampler_base import SamplerBase
from rgfn.gfns.reaction_gfn.api.reaction_api import *
from rgfn.shared.samplers.parallel_sampler import (
    BacktrackStrategy,
    ParallelSampler,
    SelectionStrategy,
)
from rgfn.trainer.trainer import Trainer
from rgfn.utils.helpers import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="samples.csv")
    parser.add_argument(
        "--n_backtrack_steps", type=int, default=2, help="Number of backtracking steps"
    )
    parser.add_argument(
        "--n_mixtures", type=int, default=100, help="Number of trajectories to sample"
    )
    parser.add_argument(
        "--mixture_size", type=int, default=12, help="Number of refined samples per trajectory"
    )
    parser.add_argument(
        "--backtrack_strategy", type=str, default="LSD", choices=[s.name for s in BacktrackStrategy]
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="BEAM",
        choices=[s.name for s in SelectionStrategy],
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--top_k", type=float, default=0.7, help="Top-k sampling threshold")
    args = parser.parse_args()
    seed = args.seed

    # NOTE: Hyperparameters for the backtracking and forward sampling
    n_backtrack_steps = args.n_backtrack_steps
    n_trajectories = args.n_mixtures
    n_refined_samples_per_traj = args.mixture_size
    backtrack_strategy = BacktrackStrategy[args.backtrack_strategy]
    selection_strategy = SelectionStrategy[args.selection_strategy]
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k

    # NOTE: Validate hyperparameters
    assert n_backtrack_steps > 1, "n_backtrack_steps must be greater than 1"
    assert n_trajectories > 0, "n_trajectories must be greater than 0"
    assert n_refined_samples_per_traj > 0, "n_refined_samples_per_traj must be greater than 0"

    train_dir = (
        "/Users/stephenlu/Documents/ml/rgfn/experiments/scent_qed_proxy/2025-07-24_18-07-51/"
    )
    args.cfg = train_dir + "logs/config.txt"
    args.checkpoint_path = train_dir + "train/checkpoints/last_gfn.pt"

    seed_everything(seed)

    config_name = Path(args.cfg).stem
    run_name = f"{config_name}/{get_time_stamp()}"
    gin.parse_config_files_and_bindings([args.cfg], bindings=[f'run_name="{run_name}"'])
    print(f"Loading run {run_name} from config file {args.cfg}")

    trainer = Trainer(resume_path=args.checkpoint_path)
    env = trainer.train_forward_sampler.env
    policy = trainer.train_forward_sampler.policy
    reward = trainer.train_forward_sampler.reward

    forward_sampler = ParallelSampler(
        policy=policy,
        env=env,
        mixture_reward=reward,
        max_mixture_size=n_refined_samples_per_traj,
        n_backtrack_steps=n_backtrack_steps,
        backtrack_strategy=backtrack_strategy,
        selection_strategy=selection_strategy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    assert forward_sampler.env.is_reversed == False, "Forward sampler should not be reversed"

    # NOTE: Sample mixtures from initial states
    source_states = forward_sampler.env.sample_source_states(n_trajectories)

    trajectories = SamplerBase.sample_trajectories_from_sources(forward_sampler, source_states)
    backtracked_states, _ = forward_sampler.backtrack_trajectories(trajectories)
    backtracked_states = [
        st for st in backtracked_states if not isinstance(st, ReactionStateEarlyTerminal)
    ]
    backtracked_proxy = (
        reward.compute_reward_output(backtracked_states).proxy.detach().cpu().numpy()
    )
    mixture_states: List[List[ReactionState]] = [
        forward_sampler.forward_from_source_to_mixture[forward_sampler.strategy](state)
        for state in tqdm(backtracked_states)
    ]

    mixture_proxy = [
        reward.compute_reward_output(list(proposal_states)).proxy.detach().cpu().numpy()
        for proposal_states in mixture_states
    ]

    print(f"Maximum final mixture size: {np.nanmax([len(ms) for ms in mixture_states])}")
    print(f"Average final mixture size: {np.mean([len(ms) for ms in mixture_states])}")
    print(f"Maximum final mixture proxy: {np.nanmax([mp.mean() for mp in mixture_proxy])}")
    print(f"Average final mixture proxy: {np.mean([mp.mean() for mp in mixture_proxy])}")

    # Save the results to csv file
    data = []
    for i, backtracked_state in enumerate(backtracked_states):
        scaffold_smi = backtracked_state.molecule.smiles
        mixture_smis = [st.molecule.smiles for st in mixture_states[i] if hasattr(st, "molecule")]
        scaffold_prox = backtracked_proxy[i]
        mixture_prox = mixture_proxy[i]
        traj_id = f"traj_{i}"

        data.append(
            {
                "tid": traj_id,
                "smiles": scaffold_smi,
                "proxy": scaffold_prox,
                "is_scaffold": True,
            }
        )
        data.extend(
            [
                {
                    "tid": traj_id,
                    "smiles": smi,
                    "proxy": prox,
                    "is_scaffold": False,
                }
                for smi, prox in zip(mixture_smis, mixture_prox)
            ]
        )

    df = pd.DataFrame(data)
    df.to_csv(args.output_path)
    # breakpoint()
