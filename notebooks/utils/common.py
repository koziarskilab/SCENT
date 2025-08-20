from typing import Any, Dict, Tuple

import numpy as np
from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
from tqdm import tqdm

from rgfn import ReactionDataFactory
from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
from rgfn.gfns.reaction_gfn.proxies.path_cost_proxy import PathCostProxy


def _get_scaffold(smiles: str, generic: bool = False):
    if not isinstance(smiles, str):
        return None
    mol = MolFromSmiles(smiles)
    scaffold_mol = GetScaffoldForMol(mol)
    if generic:
        scaffold_mol = MakeScaffoldGeneric(scaffold_mol)
    return MolToSmiles(scaffold_mol)


def _get_fp(smiles: str, fp_type: str, n_bits: int = 2048):
    if "morgan" in fp_type:
        radius = int(fp_type[-1])
        mol = MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(
            MolFromSmiles(smiles),
            radius=radius,
            nBits=n_bits,
            useFeatures=False,
            useChirality=False,
        )
    else:
        raise ValueError(f"Unknown fp_type: {fp_type}")


def get_path_cost_proxy(templates_name: str, sanitized: bool | str) -> PathCostProxy:
    if "synflow" in templates_name:
        k = int(templates_name.split("_")[-1])
        cost_path = f"../data/synflow/fragment_{k}k_to_real_cost.json"
        if sanitized is True:
            cost_path = f"../data/synflow/fragment_{k}k_to_real_cost_sanitized.json"
        elif sanitized == "rxnflow":
            cost_path = f"../data/synflow/fragment_{k}k_to_real_cost_rxnflow.json"
        return PathCostProxy(
            data_factory=ReactionDataFactory(
                reaction_path=f"../data/synflow/templates.txt",
                fragment_path=f"../data/synflow/fragment_{k}k.txt",
                cost_path=cost_path,
            ),
            yield_value=0.75,
        )
    elif templates_name == "rgfn_new_filtered":
        cost_path = f"../data/rgfn_new_filtered/fragment_to_real_cost.json"
        if sanitized is True:
            cost_path = f"../data/rgfn_new_filtered/fragment_to_real_cost_sanitized.json"
        elif sanitized == "rxnflow":
            cost_path = f"../data/rgfn_new_filtered/fragment_to_real_cost_rxnflow.json"
        return PathCostProxy(
            data_factory=ReactionDataFactory(
                reaction_path="../data/rgfn_new_filtered/templates.txt",
                fragment_path="../data/rgfn_new_filtered/fragments.txt",
                cost_path=cost_path,
                yield_path="../data/rgfn_new_filtered/templates_yields.csv",
            )
        )
    else:
        raise ValueError(f"Unknown templates_name: {templates_name}")


def get_path_costs(
    result: Any,
    actual_path_cost_proxy: PathCostProxy,
    molecule_to_cheapest_cost: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    costs = []
    molecule_to_cheapest_cost = molecule_to_cheapest_cost or {}
    for path in tqdm(result.paths, desc="path_costs"):
        current_molecule = path[0]
        current_cost = get_fragment_cost(
            current_molecule, actual_path_cost_proxy, molecule_to_cheapest_cost
        )
        reaction = None
        input_molecules = None
        fragment_costs = None
        counter = 0
        for molecule_or_reaction in path[1:]:
            if isinstance(molecule_or_reaction, str):
                yield_value = actual_path_cost_proxy.compute_yield_raw(
                    input_smiles_list=input_molecules,
                    output_smiles=molecule_or_reaction,
                    reaction=reaction,
                )
                current_cost = fragment_costs * yield_value**-1
                current_molecule = molecule_or_reaction
                molecule_to_cheapest_cost[molecule_or_reaction] = min(
                    molecule_to_cheapest_cost.get(molecule_or_reaction, np.inf), current_cost
                )
                counter += 1
            else:
                reaction = molecule_or_reaction[0]
                fragments_smiles = [m for m in molecule_or_reaction[1:] if isinstance(m, str)]
                input_molecules = [current_molecule] + fragments_smiles
                fragment_costs = current_cost + sum(
                    get_fragment_cost(m, actual_path_cost_proxy, molecule_to_cheapest_cost)
                    for m in fragments_smiles
                )
        costs.append(current_cost)

    return np.array(costs), molecule_to_cheapest_cost


def get_fragment_cost(
    molecule: str,
    actual_path_cost_proxy: PathCostProxy,
    molecule_to_cheapest_cost: Dict[str, float],
):
    if molecule in actual_path_cost_proxy.fragment_smiles_to_cost:
        return actual_path_cost_proxy.fragment_smiles_to_cost[molecule]
    elif molecule in molecule_to_cheapest_cost:
        return molecule_to_cheapest_cost[molecule]
    else:
        cost = decompose_and_get_cost(molecule, actual_path_cost_proxy, molecule_to_cheapest_cost)
        assert cost != np.inf, f"Cost for {molecule} is np.inf"
        return cost


def decompose_and_get_cost(
    molecule: str,
    actual_path_cost_proxy: PathCostProxy,
    molecule_to_cheapest_cost: Dict[str, float],
):
    if molecule in actual_path_cost_proxy.fragment_smiles_to_cost:
        return actual_path_cost_proxy.fragment_smiles_to_cost[molecule]
    elif molecule in molecule_to_cheapest_cost:
        return molecule_to_cheapest_cost[molecule]
    molecule_to_cheapest_cost[molecule] = np.inf

    mol = Molecule(molecule)
    if not mol.valid:
        return np.inf

    reactions = actual_path_cost_proxy.data_factory.get_reactions()
    disconnections = actual_path_cost_proxy.data_factory.get_disconnections()

    possible_costs = []
    for disconnection, reaction in zip(disconnections, reactions):
        reactants_list = disconnection.rdkit_rxn.RunReactants((mol.rdkit_mol,))
        for reactants in reactants_list:
            reactant_smiles = [Molecule(r).smiles for r in reactants]
            reactant_costs = [
                decompose_and_get_cost(r, actual_path_cost_proxy, molecule_to_cheapest_cost)
                for r in reactant_smiles
            ]
            reaction_yield = actual_path_cost_proxy.compute_yield_raw(
                input_smiles_list=reactant_smiles,
                output_smiles=molecule,
                reaction=reaction.reaction,
            )
            cost = sum(reactant_costs) * reaction_yield**-1
            possible_costs.append(cost)

    if not possible_costs:
        return np.inf
    molecule_to_cheapest_cost[molecule] = min(possible_costs)
    return molecule_to_cheapest_cost[molecule]
