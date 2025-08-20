"""
Run CHAI-1 on a protein-ligand complex and extract the predicted ligand pose from the output mmCIF file.

Example usage:
    python run_chai.py --fasta protein-ligand.fasta --template receptor.pdb --restraints pocket.restraints

The inference process can be seeded by passing a custom --seed argument (default 42).
"""

import argparse
import json
import subprocess
import tempfile

# suppress Biopython warnings
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from chai_lab.chai1 import run_inference

warnings.simplefilter("ignore", BiopythonWarning)


def get_one_letter_aa_code(residue: str) -> str:
    aa3to1 = {
        "ALA": "A",
        "VAL": "V",
        "PHE": "F",
        "PRO": "P",
        "MET": "M",
        "ILE": "I",
        "LEU": "L",
        "ASP": "D",
        "GLU": "E",
        "LYS": "K",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "TYR": "Y",
        "HIS": "H",
        "CYS": "C",
        "ASN": "N",
        "GLN": "Q",
        "TRP": "W",
        "GLY": "G",
        "MSE": "M",
    }
    return aa3to1[residue]


def get_resnames_for_resids(pdb_path: str, resids: List[str], one_letter: bool = True) -> List[str]:
    """
    Get the residue names for a list of residue indices in the format {chain_id}{res_id}
    Args:
        pdb_path: Path to the PDB file of the receptor
        resids: List of residue indices in the format {chain_id}{res_id}
        one_letter: Whether to return the one-letter or three-letter amino acid codes
    Returns:
        list: list of residue names
    """

    # Load the receptor structure with biopython
    parser = PDBParser(PERMISSIVE=1, is_pqr=True)
    structure = parser.get_structure("receptor", pdb_path)[0]  # assuming single model in PDB

    # Extract the binding pocket residue names and chain IDs (ex. A18 -> chain A, residue 18)
    res_ids = [int(residue[1:]) for residue in resids]
    chain_ids = [residue[0] for residue in resids]

    # List all chains in the structure
    chains = [chain.get_id() for chain in structure.get_chains()]

    # Get the residue names for the residues of interest
    resnames = []
    for chain, res in zip(chain_ids, res_ids):
        if chain not in chains:
            raise ValueError(f"Chain {chain} not found in the receptor structure")
        resname = structure[chain][res].get_resname()
        resnames.append(get_one_letter_aa_code(resname) if one_letter else resname)

    return resnames


def prepare_pocket_restraints_file(
    receptor_file: str, residues: list[str], min_distance=0.0, max_distance=10.0
) -> str:
    """
    Generate a restraints file for the residues in the binding pocket of a receptor structure.
    Args:
        receptor_file: Path to the receptor PDBPRQ file
        residues: List of residues to use for pocket conditioning in {chain_id}{res_id} format (e.g. A18 is residue 18 in chain A)
        min_distance: Minimum distance in Angstroms (ligand to residue)
        max_distance: Maximum distance in Angstroms (ligand to residue)
    Returns:
        str: Restraints file in the CHAI-1 format

    The restraints file is a CSV file with the following columns:
    - restraint_id: Unique identifier for the restraint (e.g. "restraint1")
    - chainA: Chain ID of the ligand (assume ligand is the last chain in the input fasta)
    - res_idxA: Residue index (empty for ligand)
    - chainB: Chain ID of the receptor residue (e.g. "A")
    - res_idxB: Residue index and one-letter amino acid code (e.g. "19S")
    - connection_type: Type of connection ("pocket" for pocket conditioning)
    - confidence: Confidence score (not implemented in CHAI-1 yet)
    - min_distance_angstrom: Minimum distance in Angstroms (ligand to residue)
    - max_distance_angstrom: Maximum distance in Angstroms (ligand to residue)
    - comment: Optional comment (empty for now)
    """

    map_integer_to_alphabet = (
        lambda x: ""
        if x == 0
        else map_integer_to_alphabet((x - 1) // 26) + chr((x - 1) % 26 + ord("A"))
    )

    # Load the receptor structure with biopython
    parser = PDBParser(PERMISSIVE=1, is_pqr=True)
    structure = parser.get_structure("receptor", receptor_file)[0]  # assuming single model in PDB

    # list all chains in the structure
    chain_ids = [chain.get_id() for chain in structure.get_chains()]
    # map the chain IDs to CHAI-1 chain IDs (A, B, C, ...)
    map_chain_ids_to_alphabet = {
        chain_id: map_integer_to_alphabet(i + 1) for i, chain_id in enumerate(chain_ids)
    }

    # Get the names of binding pocket residues and chain IDs
    pocket_resn_one_letter = get_resnames_for_resids(receptor_file, residues)
    pocket_resid = [int(residue[1:]) for residue in residues]
    pocket_chain_ids = [residue[0] for residue in residues]

    # Format the pocket residues as a restraints file
    df = pd.DataFrame(
        {
            "restraint_id": [f"restraint{i+1}" for i in range(len(pocket_resid))],
            # assume that the ligand is the last chain!
            "chainA": [map_integer_to_alphabet(len(chain_ids) + 1)] * len(pocket_resid),
            "res_idxA": [""] * len(pocket_resid),
            "chainB": [map_chain_ids_to_alphabet[chain_id] for chain_id in pocket_chain_ids],
            "res_idxB": [
                aa_code + str(resid) for resid, aa_code in zip(pocket_resid, pocket_resn_one_letter)
            ],
            "connection_type": ["pocket"] * len(pocket_resid),
            "confidence": [1.0] * len(pocket_resid),  # not implemented in CHAI-1 yet
            "min_distance_angstrom": [float(min_distance)] * len(pocket_resid),
            "max_distance_angstrom": [float(max_distance)] * len(pocket_resid),
            "comment": ["empty"] * len(pocket_resid),
        }
    )

    return df.to_csv(index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, help="Path to the fasta file")
    parser.add_argument("--template", type=str, help="Path to the receptor PDB")
    parser.add_argument("--pdb_out", type=str, help="Path to the output file for storing PDBs")
    parser.add_argument("--scores_out", type=str, help="Path to the output file for storing scores")
    parser.add_argument(
        "--num_diffn_timesteps", type=int, default=200, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--num_diffn_samples", type=int, default=5, help="Number of diffusion samples"
    )
    parser.add_argument("--device", type=str, help="Device to run the model on")
    parser.add_argument("--use_esm_embeddings", help="Use ESM embeddings", type=bool)
    parser.add_argument("--use_msa_server", help="Use MSA server", type=bool)
    parser.add_argument(
        "--msa_server_url",
        type=str,
        default="https://api.colabfold.com",
        help="MSA server URL",
    )
    parser.add_argument(
        "--msa_directory",
        type=Path,
        help="Path to the directory containing the MSA files",
    )
    parser.add_argument(
        "--pocket_residues",
        type=str,
        help="List of residues to use for pocket conditioning, in {chain_id}{res_id} format (e.g. A18 is residue 18 in chain A)",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--obabel_env",
        type=str,
        help="Name of the conda environment with Open Babel python bindings installed",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Check if the template file exists
    if not Path(args.template).exists():
        raise FileNotFoundError(f"Template file {args.template} not found")

    # make a temporary directory for the outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)

        # if restraints are provided, prepare the restraints file
        if args.pocket_residues:
            restraints = prepare_pocket_restraints_file(args.template, args.pocket_residues)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_restraints:
                temp_restraints.write(restraints)

        # run CHAI-1 on the input fasta file
        # TODO: Add conditioning on experimental structure of the receptor
        candidates = run_inference(
            fasta_file=Path(args.fasta),
            output_dir=out_dir,
            seed=args.seed,
            num_diffn_timesteps=args.num_diffn_timesteps,
            num_diffn_samples=args.num_diffn_samples,
            device=args.device,
            use_esm_embeddings=args.use_esm_embeddings,
            use_msa_server=args.use_msa_server,
            msa_server_url=args.msa_server_url,
            msa_directory=args.msa_directory,
            constraint_path=temp_restraints.name if args.pocket_residues else None,
        )

        # parse the scores from the output files
        scores_dict = {
            "aggregate_score": [],
            "ptm": [],
            "iptm": [],
            "ligand_ptm": [],
            "per_chain_ptm": [],
            "per_chain_pair_iptm": [],
            "has_inter_chain_clashes": [],
            "chain_chain_clashes": [],
        }

        for i in range(len(candidates.ranking_data)):
            scores_npz = np.load(out_dir.joinpath(f"scores.model_idx_{i}.npz"))
            # Extract numeric scores and append to the dictionary
            for score_name in ["aggregate_score", "ptm", "iptm"]:
                scores_dict[score_name].append(float(scores_npz[score_name][0]))
            # Extract per-chain scores and append to the dictionary
            for score_name in [
                "per_chain_ptm",
                "per_chain_pair_iptm",
                "has_inter_chain_clashes",
                "chain_chain_clashes",
            ]:
                scores_dict[score_name].append(scores_npz[score_name][0].tolist())
            # Append the ligand pTM score (last element in per_chain_ptm, assumed to be the ligand)
            scores_dict["ligand_ptm"].append(float(scores_npz["per_chain_ptm"][0, -1]))

        # convert output cif files to pdbs with openbabel
        pdb_paths = [cif_path.with_suffix(".pdb") for cif_path in candidates.cif_paths]
        for in_cif_path, out_pdb_path in zip(candidates.cif_paths, pdb_paths):
            cmd = f"conda run -n {args.obabel_env} obabel -i mmcif {in_cif_path} -O {out_pdb_path}"
            subprocess.run(cmd, shell=True, check=True)

        # read the pdb files into a list of strings
        pdb_str_list = []
        for pdb_path in pdb_paths:
            with open(pdb_path, "r") as f:
                pdb_str_list.append(f.read())

        # write the pdb strings to a single output file
        with open(args.pdb_out, "w") as f:
            f.write("\n".join(pdb_str_list) + "\n")

        # write the scores to a JSON file
        with open(args.scores_out, "w") as f:
            json.dump(scores_dict, f)
