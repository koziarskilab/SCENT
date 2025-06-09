from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Set, Tuple

import gin
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from rgfn.api.trajectories import TrajectoriesContainer
from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction,
    ReactionAction0,
    ReactionActionB,
    ReactionActionC,
    ReactionActionSpace0,
    ReactionActionSpace0orCBackward,
    ReactionActionSpaceA,
    ReactionActionSpaceB,
    ReactionActionSpaceC,
    ReactionState0Invalid,
    ReactionStateEarlyTerminal,
    ReactionStateTerminal,
)
from rgfn.gfns.reaction_gfn.objectives.rgfn_trajectory_filter import (
    RGFNTrajectoryFilter,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase
from rgfn.trainer.metrics.metric_base import MetricsBase, MetricsList


@gin.configurable()
class BackwardDecomposeLogProbs(MetricsBase):
    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        trajectories = trajectories_container.objective_trajectories
        if trajectories is None:
            return {}
        backward_action_spaces = trajectories.get_backward_action_spaces_flat()
        backward_log_probs = trajectories.get_backward_log_probs_flat()
        log_probs_list = []
        number_of_actions_list = []
        for action_space, log_probs in zip(backward_action_spaces, backward_log_probs):
            if isinstance(action_space, ReactionActionSpace0orCBackward):
                log_probs_list.append(log_probs.detach().cpu().numpy())
                number_of_actions_list.append(len(action_space))
        return {
            "backward_log_probs": np.mean(log_probs_list),
            "backward_num_actions": np.mean(number_of_actions_list),
        }


@gin.configurable()
class ForwardLogProbs(MetricsBase):
    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        trajectories = trajectories_container.objective_trajectories
        if trajectories is None:
            return {}
        forward_action_spaces = trajectories.get_forward_action_spaces_flat()
        forward_log_probs = trajectories.get_forward_log_probs_flat()
        log_probs_dict = defaultdict(list)
        for action_space, log_probs in zip(forward_action_spaces, forward_log_probs):
            log_probs = log_probs.detach().cpu().numpy()
            if isinstance(action_space, ReactionActionSpace0):
                log_probs_dict["forward_log_probs_0"].append(log_probs)
            elif isinstance(action_space, ReactionActionSpaceA):
                log_probs_dict["forward_log_probs_A"].append(log_probs)
            elif isinstance(action_space, ReactionActionSpaceB):
                log_probs_dict["forward_log_probs_B"].append(log_probs)
            elif isinstance(action_space, ReactionActionSpaceC):
                log_probs_dict["forward_log_probs_C"].append(log_probs)

        return {k: np.mean(v) for k, v in log_probs_dict.items()}


@gin.configurable()
class NewBuildingBlocksUsage(MetricsBase):
    def __init__(self, threshold: float, last_n_discovered: int = 100, last_n_any: int = 1000):
        self.high_reward_molecules: Set[str] = set()
        self.new_building_blocks: Set[Molecule] = set()
        self.forward_discovered: List[bool] = [False]
        self.forward_as_any: List[bool] = [False]
        self.forward_as_shortcut: List[bool] = [False]
        self.forward_as_fragment: List[bool] = [False]

        self.replay_as_any: List[bool] = [False]
        self.replay_as_shortcut: List[bool] = [False]
        self.replay_as_fragment: List[bool] = [False]

        self.threshold = threshold
        self.last_n_discovered = last_n_discovered
        self.last_n_any = last_n_any

    def _uses_new_building_block_as_fragment(self, actions: List[ReactionAction]) -> bool:
        return any(
            isinstance(action, ReactionActionB) and action.fragment in self.new_building_blocks
            for action in actions
        )

    def _uses_new_building_block_as_shortcut(self, actions: List[ReactionAction]) -> bool:
        return (
            isinstance(actions[0], ReactionAction0)
            and actions[0].fragment in self.new_building_blocks
        )

    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, Any]:
        forward_trajectories = trajectories_container.forward_trajectories
        for states, actions, reward in zip(
            forward_trajectories.get_all_states_grouped(),
            forward_trajectories.get_all_actions_grouped(),
            forward_trajectories.get_reward_outputs().proxy,
        ):
            if reward.item() > self.threshold:
                smiles = states[-1].molecule.smiles
                as_shortcut = self._uses_new_building_block_as_shortcut(actions)
                as_fragment = self._uses_new_building_block_as_fragment(actions)
                as_any = as_shortcut or as_fragment

                self.forward_as_any.append(as_any)
                self.forward_as_shortcut.append(as_shortcut)
                self.forward_as_fragment.append(as_fragment)
                if smiles not in self.high_reward_molecules:
                    self.forward_discovered.append(as_any)
                    self.high_reward_molecules.add(smiles)

        replay_trajectories = trajectories_container.replay_trajectories
        if trajectories_container.replay_trajectories is not None and len(replay_trajectories) > 0:
            for states, actions, reward in zip(
                replay_trajectories.get_all_states_grouped(),
                replay_trajectories.get_all_actions_grouped(),
                replay_trajectories.get_reward_outputs().proxy,
            ):
                as_shortcut = self._uses_new_building_block_as_shortcut(actions)
                as_fragment = self._uses_new_building_block_as_fragment(actions)
                as_any = as_shortcut or as_fragment
                self.replay_as_any.append(as_any)
                self.replay_as_shortcut.append(as_shortcut)
                self.replay_as_fragment.append(as_fragment)

        self.forward_discovered = self.forward_discovered[-self.last_n_discovered :]
        self.forward_as_any = self.forward_as_any[-self.last_n_any :]
        self.forward_as_shortcut = self.forward_as_shortcut[-self.last_n_any :]
        self.forward_as_fragment = self.forward_as_fragment[-self.last_n_any :]
        self.replay_as_any = self.replay_as_any[-self.last_n_any :]
        self.replay_as_shortcut = self.replay_as_shortcut[-self.last_n_any :]
        self.replay_as_fragment = self.replay_as_fragment[-self.last_n_any :]
        return {
            "discovered_with_new_bb": np.mean(self.forward_discovered),
            "forward_bb_used_as_shortcut": np.mean(self.forward_as_shortcut),
            "forward_bb_used_as_fragment": np.mean(self.forward_as_fragment),
            "forward_bb_used": np.mean(self.forward_as_any),
            "replay_bb_used_as_shortcut": np.mean(self.replay_as_shortcut),
            "replay_bb_used_as_fragment": np.mean(self.replay_as_fragment),
            "replay_bb_used": np.mean(self.replay_as_any),
        }

    def on_update_fragments_library(
        self,
        iteration_idx: int,
        fragments: List[Molecule],
        costs: List[float],
        recursive: bool = True,
    ) -> Dict[str, Any]:
        self.new_building_blocks.update(fragments)
        return {}


@gin.configurable()
class ActionSpaceSize(MetricsBase):
    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        trajectories = trajectories_container.get_all_trajectories()
        action_spaces = trajectories.get_backward_action_spaces_flat()
        action_spaces_sizes = [
            len(a) for a in action_spaces if isinstance(a, ReactionActionSpace0orCBackward)
        ]
        return {"backward_c_mean_actions": np.mean(action_spaces_sizes)}


@gin.configurable()
class ScaffoldCost(MetricsBase):
    def __init__(
        self,
        threshold: float,
        proxy_component_name: str | None = None,
        n_cheapest_list: List[int] = (100,),
        forward_only: bool = False,
    ):
        super().__init__()
        self.threshold = threshold
        self.proxy_component_name = proxy_component_name
        self.scaffold_to_mean_cost: Dict[str, float] = defaultdict(lambda: 0)
        self.scaffold_to_count: Dict[str, int] = defaultdict(lambda: 0)
        self.scaffold_to_min_cost: Dict[str, float] = defaultdict(lambda: float("inf"))
        self.n_cheapest_list = n_cheapest_list
        self.forward_only = forward_only
        self.trajectory_filter = RGFNTrajectoryFilter()

    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        trajectories = (
            trajectories_container.forward_trajectories
            if self.forward_only
            else trajectories_container.get_all_non_backward_trajectories()
        )
        trajectories = self.trajectory_filter(trajectories)
        terminal_states = trajectories.get_last_states_flat()
        reward_outputs = trajectories.get_reward_outputs()
        proxy_values = (
            reward_outputs.proxy
            if self.proxy_component_name is None
            else reward_outputs.proxy_components[self.proxy_component_name]
        )
        costs = trajectories.get_costs()
        for state, proxy, cost in zip(terminal_states, proxy_values, costs):
            if proxy.item() > self.threshold:
                scaffold = MurckoScaffoldSmiles(state.molecule.smiles)
                current_mean = self.scaffold_to_mean_cost[scaffold]
                current_count = self.scaffold_to_count[scaffold]
                new_mean = current_mean * (current_count / (current_count + 1)) + cost / (
                    current_count + 1
                )
                self.scaffold_to_mean_cost[scaffold] = new_mean
                self.scaffold_to_min_cost[scaffold] = min(self.scaffold_to_min_cost[scaffold], cost)
                self.scaffold_to_count[scaffold] += 1

        results = {}
        suffix = "_forward" if self.forward_only else ""
        mean_sorted_values = sorted(self.scaffold_to_mean_cost.values())
        min_sorted_values = sorted(self.scaffold_to_min_cost.values())
        for n in self.n_cheapest_list:
            results[f"cost_{n}_cheapest_mean_{self.threshold}{suffix}"] = np.mean(
                mean_sorted_values[:n]
            )
            results[f"cost_{n}_cheapest_min_{self.threshold}{suffix}"] = np.mean(
                min_sorted_values[:n]
            )

        results[f"num_scaffolds_{self.threshold}"] = len(self.scaffold_to_mean_cost)
        return results


@gin.configurable()
class TrajectoryCost(MetricsBase):
    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        result = {}
        result["forward_mean_cost"] = np.mean(
            trajectories_container.forward_trajectories.get_costs()
        ).item()
        if trajectories_container.replay_trajectories is not None:
            costs = np.array(trajectories_container.replay_trajectories.get_costs())
            result["replay_mean_cost"] = np.mean(costs[~np.isinf(costs)]).item()
        if trajectories_container.backward_trajectories is not None:
            costs = np.array(trajectories_container.backward_trajectories.get_costs())
            result["backward_mean_cost"] = np.mean(costs[~np.isinf(costs)]).item()

        return result


@gin.configurable()
class ScaffoldCostsList(MetricsBase):
    def __init__(
        self,
        proxy_value_threshold_list: List[float],
        proxy_component_name: str | None = None,
        n_cheapest_list: List = (100,),
    ):
        super().__init__()
        self.metrics = MetricsList(
            [
                ScaffoldCost(
                    threshold=threshold,
                    proxy_component_name=proxy_component_name,
                    n_cheapest_list=n_cheapest_list,
                    forward_only=forward_only,
                )
                for threshold in proxy_value_threshold_list
                for forward_only in [False, True]
            ]
        )

    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        return self.metrics.compute_metrics(trajectories_container)


@gin.configurable()
class QED(MetricsBase):
    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        terminal_states = trajectories_container.forward_trajectories.get_last_states_flat()
        qed_scores_list = []
        for state in terminal_states:
            if isinstance(state, ReactionStateTerminal):
                qed_score = qed(state.molecule.rdkit_mol)
                qed_scores_list.append(qed_score)
        return {"qed": np.mean(qed_scores_list)}


@gin.configurable()
class NumScaffoldsFound(MetricsBase):
    def __init__(
        self,
        proxy_value_threshold_list: List[float],
        proxy_component_name: str | None,
        proxy_higher_better: bool = True,
    ):
        super().__init__()
        self.proxy_value_threshold_list = proxy_value_threshold_list
        self.proxy_higher_better = proxy_higher_better
        self.threshold_to_set: Dict[float, Set[str]] = {
            threshold: set() for threshold in proxy_value_threshold_list
        }
        self.proxy_component_name = proxy_component_name

    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        reward_outputs = trajectories_container.forward_trajectories.get_reward_outputs()
        terminal_states = trajectories_container.forward_trajectories.get_last_states_flat()
        values = (
            reward_outputs.proxy
            if self.proxy_component_name is None
            else reward_outputs.proxy_components[self.proxy_component_name]
        )
        for state, proxy_value in zip(terminal_states, values):
            for threshold in self.proxy_value_threshold_list:
                if isinstance(state, ReactionStateTerminal) and (
                    (self.proxy_higher_better and proxy_value.item() > threshold)
                    or (not self.proxy_higher_better and proxy_value.item() < threshold)
                ):
                    self.threshold_to_set[threshold].add(
                        MurckoScaffoldSmiles(state.molecule.smiles)
                    )

        return {
            f"num_scaffolds_{threshold}": len(self.threshold_to_set[threshold])
            for threshold in self.proxy_value_threshold_list
        }


@gin.configurable()
class SaveSynthesisPaths(MetricsBase):
    def __init__(
        self, run_dir: str, proxy_component_name: str | None, file_name: str = "paths.csv"
    ):
        super().__init__()
        self.path = Path(run_dir) / file_name
        self.unique_molecules = set()
        with open(self.path, "w") as f:
            f.write("iteration,path,proxy\n")
        self.to_be_added: List[Tuple[int, List, float]] = []
        self.trajectories_counter = 0
        self.iterations_counter = 0
        self.proxy_component_name = proxy_component_name

    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        trajectories = trajectories_container.forward_trajectories
        reward_outputs = trajectories.get_reward_outputs()
        values = (
            reward_outputs.proxy
            if self.proxy_component_name is None
            else reward_outputs.proxy_components[self.proxy_component_name]
        )
        states_grouped = trajectories.get_all_states_grouped()
        actions_grouped = trajectories.get_all_actions_grouped()
        to_be_added = []
        for i, (states, actions, score) in enumerate(zip(states_grouped, actions_grouped, values)):
            current_trajectory = [actions[0].fragment.smiles]
            for action, state in zip(actions[1:], states[2:]):
                if isinstance(action, ReactionActionC):
                    if len(action.input_fragments) == 0:
                        value = (action.input_reaction.reaction, None)
                    else:
                        value = (action.input_reaction.reaction,) + tuple(
                            f.smiles for f in action.input_fragments
                        )
                    current_trajectory.append(value)
                    assert state.molecule == action.output_molecule
                    current_trajectory.append(state.molecule.smiles)

            if isinstance(states[-1], ReactionStateTerminal):
                self.unique_molecules.add(states[-1].molecule.smiles)
                score = score.item()
            else:
                score = 0.0
            to_be_added.append((self.trajectories_counter + i, current_trajectory, score))

        with open(self.path, "a") as f:
            for iteration, path, score in to_be_added:
                f.write(f'{iteration},"{path}",{score}\n')

        self.trajectories_counter += len(trajectories)
        return {
            "num_unique_molecules": len(self.unique_molecules),
            "num_visited_molecules": self.trajectories_counter,
        }


@gin.configurable()
class TanimotoSimilarityModes(MetricsBase):
    def __init__(
        self,
        run_dir: str,
        proxy: CachedProxyBase,
        term_name: str = "value",
        proxy_term_threshold: float = -np.inf,
        similarity_threshold: float = 0.7,
        max_modes: int | None = 5000,
        compute_every_n: int = 1,
        dump_to_disk: bool = False,
    ):
        super().__init__()
        self.proxy = proxy
        self.term_name = term_name if term_name is not None else "value"
        self.proxy_term_threshold = proxy_term_threshold
        self.similarity_threshold = similarity_threshold
        self.max_modes = max_modes
        self.compute_every_n = compute_every_n
        self.iterations = 0
        self.dump_path = Path(run_dir) / "modes"
        self.dump_path.mkdir(exist_ok=True, parents=True)
        self.xlsx_path = None
        self.dump_to_disk = dump_to_disk

    def _extract_top_sorted_smiles(self) -> Dict[str, float | Dict[str, float]]:
        """
        Fetches SMILES from proxy cache, extracts the ones with reward above thresholds,
        and sorts them by reward.
        """
        if isinstance(next(iter(self.proxy.cache.values())), float):
            cache = {k: {"value": v} for k, v in self.proxy.cache.items()}
        else:
            cache = self.proxy.cache

        d = {}
        for state, scores in cache.items():
            if (
                isinstance(state, ReactionStateTerminal)
                and scores[self.term_name] >= self.proxy_term_threshold
            ):
                d[state.molecule.smiles] = scores
        d = dict(sorted(d.items(), key=lambda item: item[1][self.term_name], reverse=True))

        return d

    def _extract_modes(self) -> Dict[str, float | Dict[str, float]]:
        d = self._extract_top_sorted_smiles()
        mols = [Chem.MolFromSmiles(x) for x in d.keys()]
        ecfps = [
            AllChem.GetMorganFingerprintAsBitVect(
                m, radius=3, nBits=2048, useFeatures=False, useChirality=False
            )
            for m in mols
        ]
        modes = []
        for mol, ecfp, r, smiles in zip(mols, ecfps, d.values(), d.keys()):
            if len(modes) >= self.max_modes:
                break
            is_mode = True
            for mode in modes:
                if DataStructs.TanimotoSimilarity(ecfp, mode[1]) > self.similarity_threshold:
                    is_mode = False
                    break
            if is_mode:
                modes.append((mol, ecfp, r, smiles))
        return {m[3]: m[2] for m in modes}

    @staticmethod
    def _modes_to_df(modes: Dict[str, float | Dict[str, float]]) -> pd.DataFrame:
        reward_terms = [k for k in next(iter(modes.values())).keys() if k != "value"]

        rows = []
        for smiles, scores in modes.items():
            reward = scores["value"]
            mol = Chem.MolFromSmiles(smiles)
            heavy_atoms = mol.GetNumHeavyAtoms()
            efficiency = reward / heavy_atoms
            row = (
                [
                    "",
                    f"{np.round(reward, 2):.2f}",
                ]
                + [f"{np.round(scores[term], 2):.2f}" for term in reward_terms]
                + [
                    f"{np.round(Descriptors.ExactMolWt(mol), 2):.2f}",
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    heavy_atoms,
                    f"{np.round(Descriptors.MolLogP(mol), 3):.3f}",
                    Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
                    f"{np.round(efficiency, 4):.4f}",
                    smiles,
                ]
            )
            rows.append(row)

        columns = (
            [
                "Molecule",
                "Reward",
            ]
            + [f"Reward ({term})" for term in reward_terms]
            + [
                "MW",
                "H-bond donors",
                "H-bond acceptors",
                "Heavy atoms",
                "cLogP",
                "Rotatable bonds",
                "Ligand efficiency",
                "SMILES",
            ]
        )

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _save_modes_xlsx(df: pd.DataFrame, file_path: Path | str):
        writer = pd.ExcelWriter(file_path, engine="xlsxwriter")
        df.to_excel(writer, sheet_name="Molecules", index=False)
        worksheet = writer.sheets["Molecules"]
        worksheet.set_column(0, 0, 21)
        worksheet.set_column(1, len(df.columns), 15)

        directory = TemporaryDirectory()

        for i, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            image_path = Path(directory.name) / f"molecule_{i}.png"
            Draw.MolToFile(mol, filename=image_path, size=(150, 150))

            worksheet.set_row(i + 1, 120)
            worksheet.insert_image(i + 1, 0, image_path)

        writer.book.close()
        directory.cleanup()

    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        if self.iterations % self.compute_every_n == 0:
            modes = self._extract_modes()

            if self.dump_to_disk and len(modes) > 0:
                df = self._modes_to_df(modes)
                self.xlsx_path = self.dump_path / f"modes_{self.iterations}.xlsx"
                self._save_modes_xlsx(df, self.xlsx_path)

            self.iterations += 1
            return {"num_modes": len(modes)}
        else:
            self.iterations += 1

            return {}

    def collect_files(self) -> List[Path | str]:
        if self.xlsx_path is None:
            return []
        else:
            result = [self.xlsx_path]
            self.xlsx_path = None
            return result


@gin.configurable()
class FractionEarlyTerminate(MetricsBase):
    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        terminal_states = trajectories_container.forward_trajectories.get_last_states_flat()
        num_early_terminate = sum(
            [1 for state in terminal_states if isinstance(state, ReactionStateEarlyTerminal)]
        )
        result = {"fraction_forward_invalid": num_early_terminate / max(len(terminal_states), 1)}

        source_states = (
            trajectories_container.get_all_non_forward_trajectories().get_source_states_flat()
        )
        num_invalid = sum(
            [1 for state in source_states if isinstance(state, ReactionState0Invalid)]
        )
        result["fraction_backward_invalid"] = num_invalid / max(len(source_states), 1)
        return result


@gin.configurable()
class NumReactions(MetricsBase):
    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        terminal_states = trajectories_container.forward_trajectories.get_last_states_flat()
        num_reactions = np.mean(
            [
                state.num_reactions
                for state in terminal_states
                if isinstance(state, ReactionStateTerminal)
            ]
        )
        return {"num_reactions": num_reactions}
