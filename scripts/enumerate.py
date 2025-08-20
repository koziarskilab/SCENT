import argparse
from collections import defaultdict
from typing import Dict, Set

import pandas as pd
from tqdm import tqdm

from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction0,
    ReactionActionA,
    ReactionActionB,
    ReactionActionC,
    ReactionStateA,
    ReactionStateTerminal,
)
from rgfn.gfns.reaction_gfn.reaction_env import ReactionDataFactory, ReactionEnv
from rgfn.shared.policies.uniform_policy import UniformPolicy
from rgfn.shared.samplers.random_sampler import RandomSampler
from scripts.utils import parallel_search_from_source


def save(_pathway_to_mols: Dict[str, Set[str]]):
    all_smiles = []
    all_pathways = []
    for pathway, molecules in _pathway_to_mols.items():
        for molecule in molecules:
            all_smiles.append(molecule)
            all_pathways.append(pathway)
    assert len(all_smiles) == len(all_pathways)
    pd.DataFrame(
        {
            "SMILES": all_smiles,
            "Pathway": all_pathways,
        }
    ).to_csv(f"RGFN_{args.r}_{args.i}.csv", index=False)


def encode_actions(action_list):
    action_strings = []
    for action in action_list:
        if isinstance(action, ReactionAction0) or isinstance(action, ReactionActionB):
            action_strings.append(f"F_{action.fragment.idx}")
        if isinstance(action, ReactionActionA) and action.anchored_reaction is not None:
            action_strings.append(f"R_{action.anchored_reaction.idx}")
        if isinstance(action, ReactionActionC):
            action_strings.append(f"P_{action.output_molecule.smiles}")
    return ";".join(action_strings)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", type=int, required=False, default=2
    )  # number of reactions (3 is a good default)
    parser.add_argument(
        "-n",
        type=int,
        required=False,
        default=110000,
    )  # number of generated molecules
    parser.add_argument(
        "-m",
        type=int,
        required=False,
        default=1000,
    )  # maximum number of molecules per pool
    parser.add_argument(
        "-i", type=int, required=False, default=0
    )  # file suffix (if you want to run this in parallel)
    args = parser.parse_args()

    data_factory = ReactionDataFactory(
        reaction_path="/hpf/projects/mkoziarski/slu/RGFN/data/synflow/templates.txt",
        fragment_path="/hpf/projects/mkoziarski/slu/RGFN/data/synflow/fragment_64k.txt",
        # reaction_path="data/rgfn_new_filtered/templates.txt",
        # fragment_path="data/rgfn_new_filtered/fragments.txt",
        docking=False,
    )
    policy = UniformPolicy()

    rows = []
    for fragment in data_factory.fragments:
        rows.append([f"F_{fragment.idx}", fragment.smiles])
    for reaction in data_factory.reactions:
        rows.append([f"R_{reaction.idx}", reaction.reaction])

    molecules = set()
    pathway_to_mols = defaultdict(set)
    # pd.DataFrame(rows, columns=["Key", "Value"]).to_csv("mapping.csv", index=False)

    env = ReactionEnv(data_factory, max_num_reactions=args.r)
    sampler = RandomSampler(policy, env, None)
    pbar = tqdm(total=args.n)

    while len(molecules) < args.n:
        # NOTE: Sample the trajectories in batches of 100
        trajectories = sampler.sample_trajectories(100)
        last_states = trajectories.get_last_states_flat()

        terminal_mask = env.get_terminal_mask(last_states)
        assert all(terminal_mask), "Some trajectories are not terminated"

        # NOTE: Check if the trajectories are long enough for backtracking
        traj_lengths = trajectories.trajectory_lengths()
        branching_state_idxs, mask = [], []
        for traj in trajectories._states_list:
            loc_branching_state_idxs = []
            for i, s in enumerate(traj):
                if isinstance(s, ReactionStateA):
                    loc_branching_state_idxs.append(i + 1)

            if len(loc_branching_state_idxs) < 2:
                mask.append(False)
            else:
                mask.append(True)
                branching_state_idxs.append(loc_branching_state_idxs)

        trajectories = trajectories.masked_select(mask)
        print(
            f"Removed {100 - sum(mask)} trajectories that are not long enough for last step diversification"
        )
        assert len(trajectories) == len(branching_state_idxs)

        # NOTE: Backtrack the trajectories
        backtrack_idx = [x[-2] for x in branching_state_idxs]
        removed_trajectories = trajectories.backtrack_to_state_idx(backtrack_idx)

        # NOTE: Get the backtracked states which we will use to refine the trajectories
        branching_states = trajectories.get_last_states_flat()
        branching_state_types = set([type(s) for s in branching_states])
        print(f"We have {len(branching_states)} branching states of type: {branching_state_types}")

        # NOTE: Run parallel random search from the branching state to find a set of different products
        pool_sizes = []

        for i, branching_state in enumerate(branching_states):
            # Encode the parent trajectory as a SMARTS pattern
            pathway = encode_actions(removed_trajectories[i]._actions_list[0])

            if pathway in pathway_to_mols.keys():
                print("Duplicate pathway generated, skipping...")
                continue

            # Find products from the branching state, up to args.m of them per pool
            proposal_states = parallel_search_from_source(
                branching_state,
                env,
                n=args.m,
            )

            # Collect the SMILES of the terminal molecules and check for duplicates
            smiles = [
                state.molecule.smiles
                for state in proposal_states
                if type(state) == ReactionStateTerminal
            ]
            smiles = set(smiles)

            # Add pool to the pathway indexer and individual molecules to the counter
            pbar.update(len(smiles.difference(molecules)))
            pathway_to_mols[pathway] = smiles
            molecules = molecules.union(smiles)
            pool_sizes.append(len(smiles))

        total_pool_size = sum(pool_sizes)
        print(f"Mean pool size: {total_pool_size/len(pool_sizes):.3f}")
        print()

    pbar.close()
    save(pathway_to_mols)
