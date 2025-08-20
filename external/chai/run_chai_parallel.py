"""
Run CHAI-1 on a protein-ligand complex and extract the predicted ligand pose from the output mmCIF file.

Example usage:

python external/chai/run_chai_parallel.py \
    --fasta-rec external/chai/demo.in.rec.fasta \
    --fasta-lig external/chai/demo.in.lig.fasta \
    --template data/receptors/ClpP.pdbqt \
    --out external/chai/demo.out.npy \
    --seed 42 \
    --num_processes 4 \
    --num_diffn_timesteps 200 \
    --num_diffn_samples 5 \
    --device cuda \
    --use_esm_embeddings False \
    --msa_directory /h/290/stephenzlu/rgfn/data/receptors/metadata

The inference process can be seeded by passing a custom --seed argument (default 42).
"""
import argparse
import logging
import os
import shutil
import tempfile
import time
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from Bio import BiopythonWarning
from chai_lab.chai1 import (
    StructureCandidates,
    make_all_atom_feature_context,
    run_folding_on_context,
)
from chai_lab.data.dataset.all_atom_feature_context import AllAtomFeatureContext
from tqdm import tqdm
from utils import ligands_fasta_to_dict, prepare_pocket_restraints_file

warnings.simplefilter("ignore", BiopythonWarning)

# ------------------------------------------------------------
# Make sure to enter the correct name of default conda environment for the project
BINDING_AFFINITY_CONDA_ENV = "rgfn"
# ------------------------------------------------------------


def process_single_ligand(ligand_data):
    """Process a single ligand-receptor pair. This function will be called in parallel."""
    ligand_id, ligand_seq, rec_fasta_path, args, temp_restraints_path, torch_device = ligand_data

    print(f"Running CHAI-1 for ligand {ligand_id} with sequence {ligand_seq}")

    # Create a temporary fasta file for the ligand, receptor pair
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_fasta:
        with open(rec_fasta_path, "r") as rec_fasta:
            temp_fasta.write(rec_fasta.read())
        temp_fasta.write("\n")
        temp_fasta.write(f">{ligand_id}\n{ligand_seq}\n")

    # Create a temporary output directory for this ligand-receptor combination
    output_dir = tempfile.TemporaryDirectory(prefix=f"chai_output_{ligand_id}_", delete=False)
    output_dir = Path(output_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE if fastas are cif chain names, we also use them to parse chains and restraints
    feature_context = make_all_atom_feature_context(
        fasta_file=Path(temp_fasta.name),
        output_dir=output_dir,
        entity_name_as_subchain=False,
        use_esm_embeddings=args.use_esm_embeddings,
        use_msa_server=args.use_msa_server,
        msa_server_url=args.msa_server_url,
        msa_directory=args.msa_directory,
        constraint_path=temp_restraints_path,
        use_templates_server=False,
        templates_path=Path(args.template) if args.template != "NULL" else None,
        esm_device=torch_device,
    )

    # Print the size of feature_context object in GB
    from sys import getsizeof

    print(f"Feature context size: {getsizeof(feature_context) / (1024 ** 3):.2f} GB")

    # Clean up temporary fasta file
    os.unlink(temp_fasta.name)

    return (output_dir, feature_context, ligand_id)


def run_inference_on_context(inference_data):
    """Run inference on a feature context. This function will be called in parallel."""
    outdir, feature_context, ligand_id, args, torch_device = inference_data

    # Import here to avoid issues with multiprocessing
    from chai_lab.chai1 import StructureCandidates, run_folding_on_context

    print(f"Running inference for ligand {ligand_id}")

    # run CHAI-1 on the feature context
    all_candidates: list[StructureCandidates] = []
    num_trunk_samples = 1
    num_trunk_recycles = 3

    for trunk_idx in range(num_trunk_samples):
        logging.info(f"Trunk sample {trunk_idx + 1}/{num_trunk_samples} for ligand {ligand_id}")
        cand = run_folding_on_context(
            feature_context,
            output_dir=(outdir / f"trunk_{trunk_idx}" if num_trunk_samples > 1 else outdir),
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=args.num_diffn_timesteps,
            num_diffn_samples=args.num_diffn_samples,
            recycle_msa_subsample=0,
            seed=args.seed + trunk_idx if args.seed is not None else None,
            device=torch_device,
            low_memory=True,
            entity_names_as_chain_names_in_output_cif=False,
        )
        all_candidates.append(cand)

    candidates = StructureCandidates.concat(all_candidates)
    candidates = candidates.sorted()
    # scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]
    scores = [rd.ptm_scores.per_chain_pair_iptm[-1].mean().item() for rd in candidates.ranking_data]
    return scores


def main(
    args,
    outdir: Path,
    feature_context: AllAtomFeatureContext,
    num_trunk_samples: int = 1,
    num_trunk_recycles: int = 3,
) -> Tuple[List[float], List[Path]]:
    """Main function to run CHAI-1 on a protein-ligand complex."""
    # run CHAI-1 on the input fasta file
    all_candidates: list[StructureCandidates] = []
    for trunk_idx in range(num_trunk_samples):
        logging.info(f"Trunk sample {trunk_idx + 1}/{num_trunk_samples}")
        cand = run_folding_on_context(
            feature_context,
            output_dir=(outdir / f"trunk_{trunk_idx}" if num_trunk_samples > 1 else outdir),
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=args.num_diffn_timesteps,
            num_diffn_samples=args.num_diffn_samples,
            recycle_msa_subsample=0,
            seed=args.seed + trunk_idx if args.seed is not None else None,
            device=torch_device,
            low_memory=True,
            entity_names_as_chain_names_in_output_cif=False,
        )
        all_candidates.append(cand)

    candidates = StructureCandidates.concat(all_candidates)
    candidates = candidates.sorted()
    # scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]
    scores = [rd.per_chain_pair_iptm[-1].mean().item() for rd in candidates.ranking_data]
    return scores, candidates.cif_paths


if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta-rec", type=str, help="Path to the fasta file containing the receptor chains"
    )
    parser.add_argument(
        "--fasta-lig", type=str, help="Path to the fasta file containing the ligand chains"
    )
    parser.add_argument("--template", type=str, help="Path to template .m8 file", default="NULL")
    parser.add_argument(
        "--out", type=str, help="Path to the output file containing pose pTM scores"
    )
    parser.add_argument(
        "--num_diffn_timesteps", type=int, default=200, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--num_diffn_samples", type=int, default=5, help="Number of diffusion samples"
    )
    parser.add_argument("--device", type=str, help="Device to run the model on")
    parser.add_argument("--use_esm_embeddings", help="Use ESM embeddings")
    parser.add_argument("--use_msa_server", help="Use MSA server")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of processes for parallel execution (default: number of CPU cores)",
    )
    args = parser.parse_args()

    # Check if the template file exists
    if args.template != "NULL" and not Path(args.template).exists():
        raise FileNotFoundError(f"Template file {args.template} not found")

    # Assert that the fasta files exist
    rec_fasta_path = Path(args.fasta_rec)
    lig_fasta_path = Path(args.fasta_lig)

    if not rec_fasta_path.exists():
        raise FileNotFoundError(f"Receptor fasta file {args.fasta_rec} not found")

    if not lig_fasta_path.exists():
        raise FileNotFoundError(f"Ligand fasta file {args.fasta_lig} not found")

    #######################################################
    ############## Prepare the receptor ONCE ##############
    #######################################################

    torch_device = torch.device(args.device if args.device is not None else "cuda:0")

    # if restraints are provided, prepare the restraints file
    temp_restraints = None
    temp_restraints_path = None
    if args.pocket_residues:
        restraints = prepare_pocket_restraints_file(args.template, args.pocket_residues)
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_restraints:
            temp_restraints.write(restraints)
            temp_restraints_path = temp_restraints.name

    # Set number of processes
    num_processes = args.num_processes if args.num_processes is not None else cpu_count()
    print(f"Using {num_processes} processes for parallel execution")

    ######################################################################################
    ################ Run feature context creation in parallel ##########################
    ######################################################################################

    # Prepare data for parallel processing
    ligands_dict = ligands_fasta_to_dict(lig_fasta_path)
    ligand_data_list = [
        (ligand_id, ligand_seq, rec_fasta_path, args, temp_restraints_path, torch_device)
        for ligand_id, ligand_seq in ligands_dict.items()
    ]

    print(f"Creating feature contexts for {len(ligand_data_list)} ligands in parallel...")

    # Process ligands in parallel for feature context creation
    # with Pool(processes=num_processes) as pool:
    #     feature_context_results = list(tqdm(
    #         pool.imap(process_single_ligand, ligand_data_list),
    #         total=len(ligand_data_list),
    #         desc="Creating feature contexts"
    #     ))
    feature_context_results = []
    for ligand_data in tqdm(ligand_data_list, desc="Creating feature contexts"):
        feature_context_results.append(process_single_ligand(ligand_data))
    print("Feature context creation completed. Running inference...")

    ######################################################################################
    ################ Run inference on each feature context in parallel ##################
    ######################################################################################

    # Prepare data for inference
    inference_data_list = [
        (outdir, feature_context, ligand_id, args, torch_device)
        for outdir, feature_context, ligand_id in feature_context_results
    ]

    # Run inference in series
    list_of_scores = []
    for outdir, feature_context, ligand_id in feature_context_results:
        print(f"Running inference for ligand {ligand_id} in directory {outdir}")
        scores = run_inference_on_context((outdir, feature_context, ligand_id, args, torch_device))
        list_of_scores.append(scores)
        shutil.rmtree(outdir)

    # Save the list of lists of scores to the output file as a .npz file
    scores_out_path = Path(args.out)
    np.save(scores_out_path, np.array(list_of_scores))

    t1 = time.time()
    print(f"Total time taken: {t1 - t0:.2f} seconds")
