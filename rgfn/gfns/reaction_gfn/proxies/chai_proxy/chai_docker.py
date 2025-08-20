import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gin
import numpy as np

from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
    ReactionStateTerminal,
)
from rgfn.gfns.reaction_gfn.proxies.chai_proxy.docker_base import DockerBase
from rgfn.gfns.reaction_gfn.proxies.chai_proxy.utils import (
    chai_hash,
    convert_pdbqt_file_to_pdb_str,
    fastas_from_pdb,
    get_sequence_from_pdb,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase
from rgfn.shared.proxies.normalised_proxy import NormalisedProxy
from rgfn.shared.proxies.server_proxy import ServerProxy


@gin.configurable()
class ChaiDockerProxy(
    ServerProxy[ReactionState],
    NormalisedProxy[ReactionState],
    CachedProxyBase[ReactionState],
    DockerBase,
):
    """
    Predict binding poses with Chai
    https://github.com/chaidiscovery/chai-lab
    """

    def __init__(
        self,
        n_workers=0,
        conda_env="chai",
        chai_path="external/chai/run_chai_parallel.py",
        num_diffn_timesteps=200,
        seed=42,
        device="cuda",
        use_proxy_server=True,
        use_esm_embeddings=False,
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
        use_msa_cache=True,
        pocket_conditioning=False,
        ## Receptor arguments
        n_conformers: int = 5,
        conf_score_agg_method: str = "mean",
        receptor_file_path: str = "data/receptors/ClpP.pdbqt",
        receptor_metadata_file_path: str | None = "data/receptors/metadata/ClpP.json",
        template_m8_file: str | None = "data/receptors/metadata/6bba.m8",
        kalign_dir: str
        | None = "/hpf/projects/mkoziarski/zdeng/chai_sandbox/kalign-3.4.0/build/install/bin",
        *args,
        **kwargs,
    ):
        if use_msa_cache:
            if use_msa_server or msa_server_url:
                logging.warning("MSA cache is enabled. MSA server will not be used.")

        super().__init__(*args, **kwargs)
        self.n_workers = n_workers
        self.conda_env = conda_env
        self.chai_path = chai_path
        self.num_diffn_timesteps = num_diffn_timesteps
        self.seed = seed
        self.device = device
        self.use_proxy_server = use_proxy_server
        self.use_esm_embeddings = use_esm_embeddings
        self.use_msa_server = use_msa_server
        self.msa_server_url = msa_server_url
        self.use_msa_cache = use_msa_cache
        self.pocket_conditioning = pocket_conditioning
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}
        self.conf_score_agg_method = conf_score_agg_method
        self.template_m8_file = template_m8_file
        self.kalign_dir = kalign_dir
        self.env = os.environ.copy()
        self.env["PATH"] = self.env.get("PATH", "") + ":" + self.kalign_dir
        self.prepare_docker(
            n_conformers=n_conformers,
            receptor_file_path=receptor_file_path,
            receptor_metadata_file_path=receptor_metadata_file_path,
        )

    def get_important_params_values(self) -> List[Any]:
        return [
            self.num_diffn_timesteps,
            self.seed,
            self.use_esm_embeddings,
            self.use_msa_server,
            self.use_msa_cache,
            self.pocket_conditioning,
        ]

    def prepare_docker(self, receptor_file_path: str, **kwargs) -> None:
        super().prepare_docker(receptor_file_path, **kwargs)
        if Path(receptor_file_path).suffix == ".pdbqt":
            pdb_str = convert_pdbqt_file_to_pdb_str(receptor_file_path)
        elif Path(receptor_file_path).suffix == ".pdb":
            pdb_str = "\n".join(open(receptor_file_path, "r").readlines())

        sequence = get_sequence_from_pdb(pdb_str)

        # Hash the sequence to get the MSA file name. Chai looks for files hashed this way for the MSA.
        self.msa_dir = Path(__file__).parents[5] / "data" / "receptors" / "metadata"
        msa_file = self.msa_dir / chai_hash(sequence) if self.use_msa_cache else None
        if self.use_msa_cache and not msa_file.exists():
            raise ValueError(f"MSA cache enabled but cached MSA not found at {msa_file}")
        else:
            logging.info(f"Cached MSA found at {msa_file}")

    @property
    def name(self) -> str:
        return "chai"

    @property
    def returns_single_result(self) -> bool:
        return True

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(self, smiles_list: List[str]) -> List[Dict[str, float]] | List[float]:
        # logging.info(f"start processing {smiles_list}")
        # from subprocess import run
        # logging.info(f"GPU Status:\n")
        # _=run(['nvidia-smi']).stderr
        final_scores, _ = self.dock_smiles(smiles_list)
        assert all(
            isinstance(score, (float, int)) for score in final_scores
        ), "All scores must be numeric."
        assert all(
            not np.isnan(score) for score in final_scores
        ), "Scores must not contain NaN values."
        return final_scores

    def _compute_proxy_output(self, states: List[ReactionState]):
        if self.use_proxy_server:
            return self.min_max_normalise(self.distributed_compute_proxy_output(states))
        return self.min_max_normalise(self.__call__([state.molecule.smiles for state in states]))

    def dock_smiles(self, smiles_list: list[str]) -> Tuple[List[float], Dict[str, List[float]]]:
        """
        Docks a list of SMILES strings.

        Args:
            smiles_list: A list of length N of SMILES strings to dock.

        Returns:
            A list of length N containing the docking results.
        """
        # get the receptor file path and convert it to fasta
        rec_fastas = fastas_from_pdb(self.receptor_file_path)
        lig_fastas = [f">ligand|name=LIG_{i}\n{smiles}\n" for i, smiles in enumerate(smiles_list)]
        temp_npy_out = tempfile.NamedTemporaryFile(mode="w", suffix=".npy", delete=False)

        # Make temporary fasta file for receptor
        temp_rec_fasta = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta")
        temp_rec_fasta.write(rec_fastas)
        temp_rec_fasta.flush()

        # Make temporary fasta file for ligands
        temp_lig_fasta = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta")
        temp_lig_fasta.write("".join(lig_fastas))
        temp_lig_fasta.flush()

        # Run inference for the SMILES using the chai script
        command = [
            f"conda run -n {self.conda_env} --live-stream python {self.chai_path}",
            f" --fasta-rec {temp_rec_fasta.name}",
            f" --fasta-lig {temp_lig_fasta.name}",
            f" --template {self.template_m8_file}",
            f" --out {temp_npy_out.name}",
            f" --seed {self.seed}",
            f" --num_processes {self.n_workers}",
            f" --num_diffn_timesteps {self.num_diffn_timesteps}",
            f" --num_diffn_samples {self.n_conformers}",
            f" --device {self.device}",
            f" --use_esm_embeddings {self.use_esm_embeddings}",
        ]

        if self.use_msa_cache:
            command.append(f" --msa_directory {self.msa_dir}")
        else:
            command.extend(
                [
                    f" --use_msa_server {self.use_msa_server}",
                    f" --msa_server_url {self.msa_server_url}",
                ]
            )

        if self.pocket_conditioning:
            # get pocket residues in the format following format: "{chain_id}{res_id}", e.g. "A18"
            if not self.receptor_pocket_residues:
                raise ValueError(
                    "Pocket residues are not set. Please set them in the receptor metadata to use pocket conditioning."
                )
            pocket_residues = [
                "".join(map(str, residue)) for residue in self.receptor_pocket_residues
            ]
            command.append(f" --pocket_residues {' '.join(pocket_residues)}")
        logging.info("chai cmd:" + "\n".join(command))
        # TODO: only print these log under debugging mode.

        def stream_reader(pipe, target):
            for line in iter(pipe.readline, ""):
                target.write(line)
                target.flush()
            pipe.close()

        process = subprocess.Popen(
            "".join(command).split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        threading.Thread(target=stream_reader, args=(process.stdout, sys.stdout)).start()
        threading.Thread(target=stream_reader, args=(process.stderr, sys.stderr)).start()
        process.wait()
        _, stderr = process.communicate()
        # from pathlib import Path
        # from shutil import copy
        # for i in [temp_rec_fasta,temp_lig_fasta,temp_npy_out]:
        #     copy(i.name,f'/hpf/projects/mkoziarski/zdeng/RGFN/data/{Path(i.name).name}')

        # logging.info(f'chai finished. returncode: {process.returncode}; stderr: {stderr}')
        # check if the process finished successfully
        metrics = {}
        if process.returncode == 0:
            scores_arr: np.ndarray = np.load(temp_npy_out.name, allow_pickle=True)
            scores = self.agg_score(scores_arr)
            metrics["variance"] = np.nanvar(scores_arr, axis=1).tolist()
            metrics["max"] = np.nanmax(scores_arr, axis=1).tolist()
            metrics["min"] = np.nanmin(scores_arr, axis=1).tolist()
        else:
            logging.warning(f"Failed to dock {smiles_list} with error: {stderr}")
            scores = [float("nan")] * len(smiles_list)

        assert len(scores) == len(smiles_list)
        logging.info(f"metrics: {metrics}")
        temp_npy_out.close()
        temp_rec_fasta.close()
        temp_lig_fasta.close()

        return scores, metrics

    def agg_score(self, scores: np.ndarray) -> List[float]:
        """
        Aggregate scores based on the configured aggregation method.
        """
        if self.conf_score_agg_method == "mean":
            return np.nanmean(scores, axis=1).tolist()
        elif self.conf_score_agg_method == "median":
            return np.nanmedian(scores, axis=1).tolist()
        elif self.conf_score_agg_method == "max":
            return np.nanmax(scores, axis=1).tolist()
        elif self.conf_score_agg_method == "min":
            return np.nanmin(scores, axis=1).tolist()
        else:
            raise ValueError(f"Unknown aggregation method: {self.conf_score_agg_method}")


if __name__ == "__main__":
    # Example usage
    import timeit

    chai_docker = ChaiDockerProxy(
        n_workers=2,
        use_esm_embeddings=True,
        use_proxy_server=False,
        use_msa_cache=False,
        use_msa_server=False,
    )
    smiles_list = [
        "CC(C)(C)OC(=O)N1CCN(CC1)C(=O)C(CC2CCCO2)Nc2cc(N)cc(N)c2",
        "CN1CCN(CC1)C(=O)C(CC2CCCO2)Nc2ccc(F)cc2",
    ]
    states: List[ReactionState] = [
        ReactionStateTerminal(molecule=Molecule(x), num_reactions=1) for x in smiles_list
    ]
    start_time = timeit.default_timer()
    result = chai_docker.compute_proxy_output(states)
    end_time = timeit.default_timer()
    print(f"Time taken to dock {len(smiles_list)} SMILES: {end_time - start_time:.2f} seconds")

    for smiles, res in zip(smiles_list, result.value):
        print(f"SMILES: {smiles}")
        print(f"Score: {res}")
        print()
