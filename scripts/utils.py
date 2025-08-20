import math
import random
from collections import defaultdict
from copy import deepcopy
from itertools import compress
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw

from rgfn.api.type_variables import TAction, TState
from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionState
from rgfn.gfns.reaction_gfn.reaction_env import ReactionEnv
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace, UniformPolicy
from rgfn.shared.proxies.cached_proxy import THashableState
from rgfn.shared.samplers.parallel_sampler import BeamSearchCandidate
from rgfn.shared.samplers.random_sampler import RandomSampler


def plot_molecules(smiles: List[str], mols_per_row: int = 5):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def compute_tanimoto_similarity_to_ref(smiles: List[str], ref_smi: str) -> np.array:
    ref_mol = Chem.MolFromSmiles(ref_smi)
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)

    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]

    similarities = np.zeros(len(smiles))
    for i in range(len(smiles)):
        similarities[i] = DataStructs.TanimotoSimilarity(ref_fp, fps[i])

    return similarities


def compute_tanimoto_similarity_to_ref_molecule(
    molecules: List[Molecule], ref_molecule: Molecule
) -> np.array:
    ref_mol = ref_molecule.rdkit_mol
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)

    mols = [mol.rdkit_mol for mol in molecules]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]

    similarities = np.zeros(len(molecules))
    for i in range(len(molecules)):
        similarities[i] = DataStructs.TanimotoSimilarity(ref_fp, fps[i])

    return similarities


def compute_tanimoto_similarity_between_fragments(smiles: List[str]) -> np.array:
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]

    similarities = np.zeros((len(smiles), len(smiles)))
    for i in range(len(smiles)):
        for j in range(i + 1, len(smiles)):
            similarities[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities[j, i] = similarities[i, j]

    return similarities


def save_final_states_to_csv(
    current_states: List[Tuple[TState, int]], forward_sampler: RandomSampler, verbose: bool = False
) -> None:
    # NOTE: Separate the final products by their original trajectory
    products_by_traj = defaultdict(set)
    for state, traj_id in current_states:
        products_by_traj[traj_id].add(state)

    print()
    print("Final products by trajectory\n")

    df = pd.DataFrame()
    for traj_id, products in products_by_traj.items():
        print(f"Trajectory {traj_id} has {len(products)} final products")
        reward_outputs = forward_sampler.reward.compute_reward_output(list(products))
        df_loc = pd.DataFrame(
            {
                "Trajectory": traj_id,
                "Reward": reward_output,
                "Product": product.molecule.smiles,
            }
            for product, reward_output in zip(products, reward_outputs.proxy.cpu().numpy())
        )
        df = pd.concat([df, df_loc], axis=0)

        if verbose:
            print(pd.DataFrame(reward_outputs.proxy.cpu().numpy()).describe())
            print()

    # Save molecules and rewards to a pandas file for further analysis
    df.to_csv("final_products.csv", index=False)
    return df


def get_action_logits_for_state(
    states: List[THashableState],
    action_spaces: List[TIndexedActionSpace],
    policy: FewPhasePolicyBase | UniformPolicy,
    pad_value: float = float("-inf"),
    pad=True,
) -> List[torch.Tensor]:
    if type(policy) == UniformPolicy:
        logits = policy._get_action_logits(states, action_spaces)
    else:
        shared_embeddings = policy.get_shared_embeddings(states, action_spaces)
        action_logits, action_to_state_idx = [], []

        for action_space_type, forward_fn in policy.action_space_to_forward_fn.items():
            phase_indices = [
                idx
                for idx, action_space in enumerate(action_spaces)
                if isinstance(action_space, action_space_type)
            ]
            if len(phase_indices) == 0:
                continue
            phase_states = [states[idx] for idx in phase_indices]
            phase_action_spaces = [action_spaces[idx] for idx in phase_indices]
            logits, _ = forward_fn(phase_states, phase_action_spaces, shared_embeddings)
            action_logits.extend(logits)
            action_to_state_idx.extend(phase_indices)

        state_to_action_idx = [0] * len(states)
        for action_idx, state_idx in enumerate(action_to_state_idx):
            state_to_action_idx[state_idx] = action_idx

        logits = [action_logits[state_to_action_idx[state_idx]] for state_idx in range(len(states))]

    if pad and pad_value is not None:
        # Pad the logits to the same length such that we can stack them with shape (n_states, max_n_actions)
        max_len = max([len(x) for x in logits])
        logits = [
            F.pad(logits[i], (0, max_len - len(logits[i])), value=pad_value)
            for i in range(len(logits))
        ]

    assert len(set([len(x) for x in logits])) == 1, "All logits should have the same length"
    return logits


def sample_k_actions_from_logits(
    logits: torch.Tensor,
    logits_mask: torch.Tensor,
    action_space: TIndexedActionSpace,
    method: str = "boltzmann",
    with_replace: bool = False,
    temperature: float = 1.0,
    guidance_strength: float = 1.0,
    max_k: int = 48,
    fragment_similarities: np.ndarray = None,
) -> List[TAction]:
    probs = torch.softmax(logits / temperature, dim=-1)
    k = min(max_k, logits_mask.sum().item())
    if method == "greedy":
        _, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        action_indices = sorted_indices[:k]
    elif method == "boltzmann":
        action_indices = torch.multinomial(probs, k, replacement=with_replace)
    elif method == "random":
        action_indices = torch.multinomial(logits_mask.float(), k, replacement=with_replace)
    elif method == "tanimoto_kriging":
        action_indices = torch.zeros(k, dtype=torch.long)
        possible_fragments = [
            action_space.get_action_at_idx(i) for i in action_space.get_possible_actions_indices()
        ]
        possible_fragment_indices = [
            frag.idx for frag in possible_fragments
        ]  # (n_possible_actions,)
        possible_fragment_similarities = fragment_similarities[possible_fragment_indices][
            :, possible_fragment_indices
        ]
        for i in range(k):
            max_action_idx = torch.argmax(probs).item()
            action_indices[i] = max_action_idx
            fragment_similarities_to_selected = possible_fragment_similarities[
                max_action_idx
            ]  # (n_possible_actions,)
            fragment_similarities_to_selected = torch.tensor(
                fragment_similarities_to_selected, dtype=torch.float32, device=logits.device
            )
            logits[max_action_idx] = float("-inf")
            logits[logits_mask] -= guidance_strength * fragment_similarities_to_selected
            probs = torch.softmax(logits / temperature, dim=-1)

    return [action_space.get_action_at_idx(idx.item()) for idx in action_indices]


def sample_k_actions_from_batched_logits(
    states: List[THashableState],  # (n_states,)
    logits: torch.Tensor,  # (n_states, n_actions)
    logits_mask: torch.Tensor,  # (n_states, n_actions)
    action_spaces: List[TIndexedActionSpace],  # (n_states,)
    method: str = "boltzmann",
    with_replace: bool = False,
    temperature: float = 1.0,
    guidance_strength: float = 1.0,
    max_k: int = 96,
) -> List[Tuple[THashableState, TAction]]:
    probs = F.softmax(logits.view(-1) / temperature, 0).view_as(logits)
    k = min(max_k, logits_mask.sum().item())

    if method == "boltzmann":
        # Sample k actions overall in the entire probs tensor
        top_k_idx = torch.multinomial(probs.view(-1), k, replacement=with_replace)
        is_top_k = torch.zeros_like(probs, dtype=torch.bool)
        is_top_k.view(-1)[top_k_idx] = True
    elif method == "greedy":
        # Get the top k actions overall in the entire probs tensor
        kth_value = torch.topk(probs.view(-1), k).values.min()
        is_top_k = probs >= kth_value
    elif method == "random":
        # Randomly sample k actions overall in the entire probs tensor
        top_k_idx = torch.multinomial(logits_mask.view(-1).float(), k, replacement=with_replace)
        is_top_k = torch.zeros_like(probs, dtype=torch.bool)
        is_top_k.view(-1)[top_k_idx] = True
    elif method == "tanimoto_kriging":
        top_k_idx = torch.zeros(k, dtype=torch.long)
        all_products = [
            action_space.get_action_at_idx(i)
            for action_space in action_spaces
            for i in action_space.get_possible_actions_indices()
        ]
        all_products = [product.output_molecule for product in all_products]
        for i in range(k):
            max_action_idx = torch.argmax(probs.view(-1)).item()
            row, col = max_action_idx // logits.size(1), max_action_idx % logits.size(1)
            chosen_product = action_spaces[row].get_action_at_idx(col).output_molecule
            product_similarity_to_chosen = compute_tanimoto_similarity_to_ref_molecule(
                all_products, ref_molecule=chosen_product
            )
            product_similarity_to_chosen = torch.tensor(
                product_similarity_to_chosen, dtype=torch.float32, device=logits.device
            )
            logits.view(-1)[max_action_idx] = float("-inf")
            logits[logits_mask] -= guidance_strength * product_similarity_to_chosen
            probs = F.softmax(logits.view(-1) / temperature, 0).view_as(logits)
            top_k_idx[i] = max_action_idx
        is_top_k = torch.zeros_like(probs, dtype=torch.bool)
        is_top_k.view(-1)[top_k_idx] = True

    # Finally, we get the state-action pairs that correspond to the True values in is_top_k
    return [
        (state, action_space.get_action_at_idx(action_idx))
        for state, top_k_action_mask, action_space in zip(states, is_top_k, action_spaces)
        for action_idx, is_top in enumerate(top_k_action_mask)
        if is_top
    ]


def beam_search_from_source(
    source_state: ReactionState,
    policy: FewPhasePolicyBase,
    env: ReactionEnv,
    temperature: float,
    n: int,
) -> List[ReactionState]:
    """Uses beam search to sample a n terminal states from a source state.

    Args:
        source_state (TState): Source state to start the trajectory from.
        n (int): Width of the beam search.

    Returns:
        List[TState]: List of terminal states sampled from the source state.
    """
    assert isinstance(policy, FewPhasePolicyBase)
    non_terminal_candidates = [BeamSearchCandidate(state=source_state)]
    terminal_candidates = []

    while True:
        current_states = [c.states[-1] for c in non_terminal_candidates]
        terminal_mask = env.get_terminal_mask(current_states)
        terminal_candidates.extend(compress(non_terminal_candidates, terminal_mask))

        non_terminal_mask = [not is_terminal for is_terminal in terminal_mask]
        non_terminal_candidates = list(compress(non_terminal_candidates, non_terminal_mask))
        non_terminal_candidates = non_terminal_candidates[:n]

        non_terminal_states = [c.states[-1] for c in non_terminal_candidates]
        if len(non_terminal_states) == 0 or len(terminal_candidates) >= n:
            break

        forward_action_spaces = env.get_forward_action_spaces(non_terminal_states)
        list_of_logits = policy._get_action_logits(non_terminal_states, forward_action_spaces)

        assert len(non_terminal_states) == len(list_of_logits)
        assert len(forward_action_spaces) == len(list_of_logits)

        new_candidate_states = []
        for i in range(len(non_terminal_states)):
            scaled_logits = list_of_logits[i] / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            top_probs, top_ids = probs.topk(min(n, len(probs)), dim=-1)
            top_actions = [
                forward_action_spaces[i].get_action_at_idx(idx.item())
                for idx, prob in zip(top_ids, top_probs)
                if prob > 0.0
            ]
            log_top_probs = torch.log(top_probs[top_probs > 0.0]).detach().cpu().numpy()
            source_states = [non_terminal_states[i]] * len(top_actions)
            new_states = env.apply_forward_actions(source_states, top_actions)

            for new_state, log_prob in zip(new_states, log_top_probs):
                new_candidate = deepcopy(non_terminal_candidates[i])
                new_candidate.add_state(new_state, log_prob)
                new_candidate_states.append(new_candidate)

        # Sort the new candidate states by their normalized score
        new_candidate_states.sort(reverse=True)
        non_terminal_candidates = new_candidate_states

    # Sort the terminal candidates by their normalized score
    terminal_candidates.sort(reverse=True)
    terminal_candidates = terminal_candidates[:n]
    return [c.states[-1] for c in terminal_candidates]


def parallel_search_from_source(
    source_state: ReactionState,
    env: ReactionEnv,
    n: int,
) -> List[ReactionState]:
    """Uses parallel search to sample up to n terminal states from a source state.

    Args:
        source_state (TState): Source state to start the trajectory from.
        n (int): Maximum number of terminal states to sample

    Returns:
        List[TState]: List of terminal states sampled from the source state.
    """
    non_terminal_candidates = [BeamSearchCandidate(state=source_state)]
    terminal_candidates: List[BeamSearchCandidate] = []

    while True:
        current_states = [c.states[-1] for c in non_terminal_candidates]
        terminal_mask = env.get_terminal_mask(current_states)
        terminal_candidates.extend(compress(non_terminal_candidates, terminal_mask))

        non_terminal_mask = [not is_terminal for is_terminal in terminal_mask]
        non_terminal_candidates = list(compress(non_terminal_candidates, non_terminal_mask))
        non_terminal_candidates = non_terminal_candidates[:n]

        non_terminal_states = [c.states[-1] for c in non_terminal_candidates]
        if len(non_terminal_states) == 0 or len(terminal_candidates) >= n:
            break

        forward_action_spaces = env.get_forward_action_spaces(non_terminal_states)
        new_candidate_states = []

        for i in range(len(non_terminal_states)):
            possible_actions_indices = forward_action_spaces[i].get_possible_actions_indices()
            random.shuffle(possible_actions_indices)
            width_per_candidate = math.ceil(n / len(non_terminal_states))
            top_actions = [
                forward_action_spaces[i].get_action_at_idx(idx)
                for idx in possible_actions_indices[:width_per_candidate]
            ]
            source_states = [non_terminal_states[i]] * len(top_actions)
            new_states = env.apply_forward_actions(source_states, top_actions)

            for new_state in new_states:
                new_candidate = deepcopy(non_terminal_candidates[i])
                new_candidate.add_state(new_state, 1)
                new_candidate_states.append(new_candidate)

        random.shuffle(new_candidate_states)
        non_terminal_candidates = new_candidate_states

    # Sort the terminal candidates by their normalized score
    random.shuffle(terminal_candidates)
    terminal_candidates = terminal_candidates[:n]
    return [c.states[-1] for c in terminal_candidates]
