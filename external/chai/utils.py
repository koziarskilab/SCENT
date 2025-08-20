# suppress Biopython warnings
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import openbabel as ob
import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from tqdm import tqdm

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
    parser = PDBParser(PERMISSIVE=True, is_pqr=True)
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


def ligands_fasta_to_dict(fasta_path: Path) -> Dict[str, str]:
    """
    Load the ligand fasta file into a dictionary where the key is the fasta id and the value is the sequence of the ligand.
    Args:
        fasta_path: Path to the ligand fasta file
    Returns:
        Dict[str, str]: Dictionary with fasta id as key and sequence as value
    """
    lig_fasta_dict = {}
    with open(fasta_path, "r") as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:]  # remove '>' character
            else:
                if current_id is not None:
                    lig_fasta_dict[current_id] = line
    return lig_fasta_dict


def mmcif_to_pdb_single(cif_file, output_dir=None):
    """Convert a single file"""
    output_dir = Path(output_dir) if output_dir else Path(cif_file).parent
    try:
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("mmcif", "pdb")
        mol = ob.OBMol()
        pdb_file = output_dir / f"{cif_file.stem}.pdb"
        if conv.ReadFile(mol, str(cif_file)):
            if conv.WriteFile(mol, str(pdb_file)):
                return f"✓ {cif_file.name}"
            else:
                return f"✗ Write failed: {cif_file.name}"
        else:
            return f"✗ Read failed: {cif_file.name}"
    except Exception as e:
        return f"✗ Error {cif_file.name}: {e}"


def mmcif_to_pdb_parallel(cif_files: List[Path], output_dir=None, max_workers=4):
    if not cif_files:
        print("No files found")
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(mmcif_to_pdb_single, cif_file, output_dir): cif_file
            for cif_file in cif_files
        }
        for future in tqdm(as_completed(futures), total=len(cif_files)):
            result = future.result()
            print(result)


def mmcif_and_scores_to_pdb_files(
    cif_paths: List[Path],
    scores: List[float],
    out_pdb_file: Path,
    BINDING_AFFINITY_CONDA_ENV: str = "rgfn",
):
    # convert output cif files to pdbs
    pdb_paths = [cif_path.with_suffix(".pdb") for cif_path in cif_paths]
    for in_cif_path, out_pdb_path in zip(cif_paths, pdb_paths):
        # TODO: Make this more robust by not using a specific conda environment!!!
        cmd = f"conda run -n {BINDING_AFFINITY_CONDA_ENV} obabel -i mmcif {in_cif_path} -O {out_pdb_path}"
        subprocess.run(cmd, shell=True, check=True)

    # read the pdb files into a list of strings
    pdb_str_list = []
    for pdb_path in pdb_paths:
        with open(pdb_path, "r") as f:
            pdb_str_list.append(f.read())

    # write the pdb strings and scores to the output file
    with open(out_pdb_file, "w") as f:
        f.write("\n".join(pdb_str_list) + "\n")
        f.write("\n".join(map(str, scores)) + "\n")
