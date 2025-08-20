import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from more_itertools import chunked
from rdkit import DataStructs
from tqdm import tqdm

from notebooks.utils.common import _get_fp, _get_scaffold


@dataclass
class SamplingResult:
    model_name: str
    seed: int
    templates_name: str
    task_name: str
    threshold: float
    include_paths: bool = True
    load_from_cache: bool = True
    n_molecules: int = 1000
    fp_type: str = "morgan_3"
    results_dir: Path = Path("results")

    base_dir: Path = field(init=False, repr=False)
    molecules: List[str] = field(init=False, repr=False)
    paths: List[str | Tuple[str, ...]] = field(init=False, repr=False)
    rewards: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.base_dir = (
            self.results_dir / self.templates_name / self.task_name / self.model_name / "sampling"
        )
        if self.include_paths:
            file_path = self.base_dir / f"paths_{self.seed}.csv"
        else:
            file_path = self.base_dir / f"molecules_{self.seed}.csv"

        if not file_path.exists():
            self.results_dir = Path("/Volumes/External Disk/RGFN/notebooks/results/")
            self.base_dir = (
                self.results_dir
                / self.templates_name
                / self.task_name
                / self.model_name
                / "sampling"
            )
            if self.include_paths:
                file_path = self.base_dir / f"paths_{self.seed}.csv"
            else:
                file_path = self.base_dir / f"molecules_{self.seed}.csv"
            raise FileNotFoundError(f"File {file_path} not found.")

        df = pd.read_csv(file_path).iloc[: self.n_molecules]
        if self.include_paths:
            self.paths = df["path"].apply(eval).tolist()
            self.molecules = [p[-1] for p in self.paths]
        else:
            column = df.columns[-2]
            self.molecules = df[column].tolist()
        if " proxy" in df.columns:
            self.rewards = np.array(df[" proxy"].tolist())
        else:
            self.rewards = np.array(df["proxy"].tolist())

        if self.task_name.lower() != "seh" and self.rewards.max() > 1.1:
            self.rewards = self.rewards / 8.0

        self.all_rewards = self.rewards.copy()
        self.all_molecules = self.molecules.copy()
        self.all_molecules_sorted = [
            x for _, x in sorted(zip(self.all_rewards, self.all_molecules), reverse=True)
        ]
        if self.threshold > 0.0:
            indices = np.where(self.rewards > self.threshold)[0]
            if self.include_paths:
                self.paths = [self.paths[i] for i in indices]
            self.molecules = [self.molecules[i] for i in indices]
            self.rewards = self.rewards[indices]

        if self.load_from_cache:
            self.load()

    def cache_path(self):
        return (
            self.base_dir
            / f"results_{self.seed}_{self.n_molecules}_{self.fp_type}_{self.threshold}.pkl"
        )

    average_reward: float = None
    average_tanimoto_similarity: float = None
    num_scaffolds: int = None
    average_cost: float = None
    diversity: float = None
    novelty: float = None

    def save(self):
        results_dict = {
            "average_reward": self.average_reward,
            "average_tanimoto_similarity": self.average_tanimoto_similarity,
            "num_scaffolds": self.num_scaffolds,
            "average_cost": self.average_cost,
            "diversity": self.diversity,
            "novelty": self.novelty,
        }
        pickle.dump(results_dict, open(self.cache_path(), "wb"))

    def load(self):
        result_path = self.cache_path()
        if not result_path.exists():
            return
        results_dict = pickle.load(open(result_path, "rb"))
        for key, value in results_dict.items():
            setattr(self, key, value)


def get_num_unique_molecules(result: SamplingResult):
    return len(set(result.molecules))


def get_average_reward(result: SamplingResult):
    return result.all_rewards.mean()


def get_average_tanimoto_similarity(result: SamplingResult):
    all_ecfp_list = [_get_fp(mol, "morgan_3") for mol in result.molecules]
    tanimoto_similarities = []
    for i in tqdm(range(len(all_ecfp_list)), desc="tanimoto_similarities"):
        similarities = DataStructs.BulkTanimotoSimilarity(
            all_ecfp_list[i], all_ecfp_list[:i] + all_ecfp_list[i + 1 :]
        )
        tanimoto_similarities.extend(similarities)
    return np.array(tanimoto_similarities).mean()


def get_novelty(result: SamplingResult, all_chembl_ecfps, top_n=1000):
    similarities_list = []
    for smiles in tqdm(result.all_molecules_sorted[:top_n], desc="novelty"):
        best_similarity = 0.0
        for batch in chunked(all_chembl_ecfps, 1000):
            similarities = DataStructs.BulkTanimotoSimilarity(
                _get_fp(smiles, "morgan_2", n_bits=1024), batch
            )
            best_similarity = max(best_similarity, np.max(similarities))
        similarities_list.append(best_similarity)
    return 1 - np.array(similarities_list).mean()


def get_diversity(result: SamplingResult, top_n=1000):
    all_ecfp_list = [_get_fp(mol, "morgan_3") for mol in result.all_molecules_sorted[:top_n]]
    similarities_list = []
    for i in tqdm(range(len(all_ecfp_list)), desc="diversity"):
        similarities = DataStructs.BulkTanimotoSimilarity(
            all_ecfp_list[i], all_ecfp_list[:i] + all_ecfp_list[i + 1 :]
        )
        similarities_list.extend(similarities)
    return 1 - np.array(similarities_list).mean()


def get_num_scaffolds(result: SamplingResult):
    return len(set([_get_scaffold(mol) for mol in result.molecules]))


class SamplingResultsList:
    def __init__(self, results: List[SamplingResult]):
        self.results = results
        self.model_name = results[0].model_name

    def get_reward_mean_std(self):
        return np.mean([r.average_reward for r in self.results]), np.std(
            [r.average_reward for r in self.results]
        )

    def get_num_unique_molecules_mean_std(self):
        return np.mean([get_num_unique_molecules(r) for r in self.results]), np.std(
            [get_num_unique_molecules(r) for r in self.results]
        )

    def get_tanimoto_similarity_mean_std(self):
        return np.mean([r.average_tanimoto_similarity for r in self.results]), np.std(
            [r.average_tanimoto_similarity for r in self.results]
        )

    def get_num_scaffolds_mean_std(self):
        return np.mean([r.num_scaffolds for r in self.results]), np.std(
            [r.num_scaffolds for r in self.results]
        )

    def get_average_cost_mean_std(self):
        return np.mean([r.average_cost for r in self.results]), np.std(
            [r.average_cost for r in self.results]
        )

    def get_diversity_mean_std(self):
        return np.mean([r.diversity for r in self.results]), np.std(
            [r.diversity for r in self.results]
        )

    def get_novelty_mean_std(self):
        return np.mean([r.novelty for r in self.results]), np.std([r.novelty for r in self.results])
