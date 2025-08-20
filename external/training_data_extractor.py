import gzip
import re
import tempfile
from datetime import date
from glob import glob
from pathlib import Path
from typing import Set

import pandas as pd
from bap.data.utils import get_canonical_smiles, rdkit_conformers_from_sdf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def save_leakage_information(leaked_smiles_list: Set[str], our_datasets_dir: str | Path):
    """
    Appends a column containing a True/False leakage flag to all datasets.
    The flag is set to True if the row's SMILES is subject to leakage, False otherwise.
    """
    our_datasets_list = glob(f"{our_datasets_dir}/*.csv")
    if "ifpan" not in our_datasets_dir:
        our_datasets_list.remove(f"{our_datasets_dir}/test.csv")
    for dataset in our_datasets_list:
        df = pd.read_csv(dataset)
        df["leaked"] = df["smiles"].isin(leaked_smiles_list)
        df.to_csv(dataset, index=False)


def isolate_leaked_gnina(training_data_dir: str | Path, our_datasets_dir: str | Path):
    """
    Isolate SMILES that GNINA 1.0 was trained on for all datasets and save them to data/datasets/leaked/gnina
    Input:
        training_data_dir - the directory where the crossdocked_all_data, redocking_all_data folders are located
        our_datasets_dir - the directory where the pipeline's ligand datasets are kept in .csv format
    Output:
        ligand_smiles_list - list containing SMILES which were included in GNINA's training data
    """
    training_data_dir += "/gnina"

    # Load all ligand sdf file paths from Wierbowski et al. dataset into a list
    ligand_smiles_list = []

    ligand_sdf_file_list = glob(
        f"{training_data_dir}/crossdocked_all_data/wierbowski_cd/**/*.sdf",
        recursive=True,
    )

    # Convert those ligands into SMILES
    for ligand_sdf_path in ligand_sdf_file_list:
        try:
            molecule = rdkit_conformers_from_sdf(ligand_sdf_path)
            ligand_smiles_list.append(Chem.MolToSmiles(molecule))
        except Exception:
            pass

    # Load all ligand sdf file paths from PDBbind 2019 refined dataset into a list
    ligand_sdf_file_list = glob(
        f"{training_data_dir}/redocking_all_data/PDBbind_refined_2019/**/*.sdf.gz",
        recursive=True,
    )

    # Uncompress with gzip and convert those ligands into SMILES
    for ligand_sdf_gz_path in ligand_sdf_file_list:
        with tempfile.NamedTemporaryFile(mode="w") as fw, gzip.open(ligand_sdf_gz_path, "rt") as fr:
            gz_content = fr.read()
            fw.write(gz_content)
            try:
                molecule = rdkit_conformers_from_sdf(fw.name)
            except Exception:
                pass
        ligand_smiles_list.append(Chem.MolToSmiles(molecule))

    # Validate the canonical SMILES with our function
    ligand_smiles_list = [
        get_canonical_smiles(ligand_smiles) for ligand_smiles in ligand_smiles_list
    ]

    return ligand_smiles_list


def isolate_leaked_chai_boltz(
    training_data_dir: str | Path, our_datasets_dir: str | Path, model: str = "boltz"
):
    """
    Isolate SMILES that Chai-1/ Boltz-1 was trained on for all datasets and save them to data/datasets/leaked/[boltz|chai]
    Input:
        training_data_dir - the directory where the Components-pub.cif file from RCSB LigandExpo is located
        our_datasets_dir - the directory where the pipeline's ligand datasets are kept in .csv format
    Output:
        ligand_smiles_list - list containing SMILES which were included in Chai/Boltz' training data
    """

    # Set cutoff date based on the model
    if model == "boltz":
        cutoff_date = date.fromisoformat("2021-09-30")
    elif model == "chai":
        cutoff_date = date.fromisoformat("2021-01-12")
    else:
        raise ValueError(
            "Function isolate_leaked_chai_boltz run with 'model' parameter other than 'chai' or 'boltz'"
        )

    # For us Chai and Boltz use the same data (with differing cutoff)
    training_data_dir += "/boltz"
    # Regex for separating mmCIF entries in a file which contains many of them
    mmcif_delimitation_pattern = r"(#\n+data_.*?\n+# .*?)(?=\n+#\n+data_|\Z)"
    # Components-pub.cif is the CCD in mmCIF format, where all ion/ligand/small-molecule mmCIFs are in one file
    with open(f"{training_data_dir}/Components-pub.cif", "r") as f:
        # Separate the CCD entries
        combined_mmcif_text = f.read()
        ligand_mmcif_list = re.findall(mmcif_delimitation_pattern, combined_mmcif_text, re.DOTALL)

    # The output leakage list
    ligand_smiles_list = []
    for i, ligand_mmcif in enumerate(ligand_mmcif_list):
        # Find the line containing initial molecule RCSB release date
        rcsb_release_date_line = re.findall(
            r"_chem_comp\.pdbx_initial_date\s+[\d\-]+", ligand_mmcif, re.MULTILINE
        )
        rcsb_release_date = date.fromisoformat(rcsb_release_date_line[0].split()[1])

        # Add to the leakage list if contained in training data
        if rcsb_release_date < cutoff_date:
            # Get canonical SMILES contained in the MMCIF
            rcsb_canonical_smiles_lines = []
            valid_canonical_smiles = set()
            for line in ligand_mmcif.split("\n"):
                if "SMILES_CANONICAL" in line:
                    rcsb_canonical_smiles_lines.append(line)

            # Normalise the canonical SMILES string
            for rcsb_canonical_smiles_line in rcsb_canonical_smiles_lines:
                smiles = rcsb_canonical_smiles_line.split()[-1]
                if smiles[-1] == '"':
                    smiles = smiles[1:-1]
                # If our function accepts the canonical SMILES then add it to the valid set
                try_valid_canonical_smiles = get_canonical_smiles(smiles)
                if try_valid_canonical_smiles is not None:
                    valid_canonical_smiles.add(try_valid_canonical_smiles)

            # If our function deemed at least one canonical SMILES valid then to the leaked SMILES list
            if len(valid_canonical_smiles) > 0:
                ligand_smiles_list.append(valid_canonical_smiles.pop())

    return ligand_smiles_list


def isolate_leaked_gaa(training_data_dir: str | Path, our_datasets_dir: str | Path):
    """
    Isolate SMILES that GAABind was trained on for all datasets and save them to data/datasets/leaked/gaa
    Input:
        training_data_dir - the directory where the sdf folder extracted from PDBbind_v2020_sdf.tar.gz is located
        our_datasets_dir - the directory where the pipeline's ligand datasets are kept in .csv format
    Output:
        ligand_smiles_list - list containing SMILES which were included in GAABind's training data
    """
    training_data_dir += "/gaa/sdf"

    ligand_smiles_list = []
    ligand_sdf_file_list = glob(f"{training_data_dir}/*.sdf")
    for ligand_sdf_path in ligand_sdf_file_list:
        try:
            molecule = rdkit_conformers_from_sdf(ligand_sdf_path)
            ligand_smiles_list.append(Chem.MolToSmiles(molecule))
        except Exception:
            pass

    # Validate the canonical SMILES with our function
    ligand_smiles_list = [
        get_canonical_smiles(ligand_smiles) for ligand_smiles in ligand_smiles_list
    ]

    return ligand_smiles_list


if __name__ == "__main__":
    # Directory where training data for all methods has been placed
    training_data_dir = "external/training_datasets"
    # Pipeline's directories for ligand data in .csv format (public, IFPAN)
    our_datasets_dirs = ["data/datasets", "data/ifpan/datasets"]

    for dataset_dir in our_datasets_dirs:
        leaked_ligand_smiles_set = set()

        # Perform processing and create a union of all leaked SMILES
        leaked_ligand_smiles_set.update(isolate_leaked_gnina(training_data_dir, dataset_dir))
        leaked_ligand_smiles_set.update(isolate_leaked_chai_boltz(training_data_dir, dataset_dir))
        leaked_ligand_smiles_set.update(isolate_leaked_gaa(training_data_dir, dataset_dir))

        # Append column with True/False leaked flag to all datasets
        save_leakage_information(leaked_ligand_smiles_set, dataset_dir)
