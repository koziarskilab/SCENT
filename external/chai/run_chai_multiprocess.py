"""
Run CHAI-1 on a protein-ligand complex and extract the predicted ligand pose from the output mmCIF file.

Example usage:

python external/chai/run_chai_multiprocess.py \
    --fasta-rec external/chai/demo.in.rec.fasta \
    --fasta-lig external/chai/demo.in.lig.fasta \
    --template data/receptors/ClpP.pdbqt \
    --out external/chai/demo.out.npy \
    --seed 42 \
    --num_diffn_timesteps 200 \
    --num_diffn_samples 5 \
    --device cuda \
    --use_esm_embeddings False \
    --msa_directory /h/290/stephenzlu/rgfn/data/receptors/metadata \
    --num_threads 2

The inference process can be seeded by passing a custom --seed argument (default 42).
"""
import argparse
import multiprocessing as mp
import os
import tempfile
import time

# suppress Biopython warnings
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from Bio import BiopythonWarning
from chai_lab.chai1 import run_inference
from tqdm import tqdm
from utils import ligands_fasta_to_dict, prepare_pocket_restraints_file

warnings.simplefilter("ignore", BiopythonWarning)

# ------------------------------------------------------------
# Make sure to enter the correct name of default conda environment for the project
BINDING_AFFINITY_CONDA_ENV = "rgfn"
# ------------------------------------------------------------


def worker_process(
    args_dict: dict, rec_fasta_path: str, temp_restraints_path: str | None, ligand_data: tuple
) -> tuple:
    """
    Worker process function that processes a single ligand.

    Args:
        args_dict: Dictionary of parsed command line arguments
        rec_fasta_path: Path to receptor fasta file
        temp_restraints_path: Path to temporary restraints file
        ligand_data: Tuple of (ligand_id, ligand_seq, job_index)

    Returns:
        Tuple of (job_index, scores)
    """
    ligand_id, ligand_seq, job_index = ligand_data

    # Recreate args namespace from dict
    args = argparse.Namespace(**args_dict)

    print(
        f"Process {os.getpid()}: Running CHAI-1 for ligand {ligand_id} with sequence {ligand_seq}"
    )

    try:
        # Create a temporary fasta file for the ligand, receptor pair
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_fasta:
            with open(rec_fasta_path, "r") as rec_fasta:
                temp_fasta.write(rec_fasta.read())
            temp_fasta.write("\n")
            temp_fasta.write(f">{ligand_id}\n{ligand_seq}\n")

        temp_fasta_in_path = Path(temp_fasta.name)

        # Use a unique seed for each process and ligand to ensure different results
        args.seed = args.seed + job_index * 1000

        # Handle temp_restraints
        temp_restraints = None
        if temp_restraints_path:
            temp_restraints = type("MockFile", (), {"name": temp_restraints_path})()

        scores, cif_paths = main(args, temp_fasta_in_path, temp_restraints)

        # Clean up temporary file
        temp_fasta_in_path.unlink()

        print(f"Process {os.getpid()}: Completed ligand {ligand_id}")
        return (job_index, scores)

    except Exception as e:
        print(f"Process {os.getpid()}: Error processing ligand {ligand_id}: {e}")
        import traceback

        traceback.print_exc()
        return (job_index, [])


def main(args, in_fasta_file: Path, temp_restraints=None) -> Tuple[List[float], List[Path]]:
    """
    Main function to run CHAI-1 on a protein-ligand complex.
    Args:
        args: Parsed command line arguments.
        in_fasta_file: Path to the input fasta file containing the receptor and ligand sequences.
        temp_restraints: Temporary file containing pocket restraints, if any.
    Returns:
        List[float]: List of pTM scores for the predicted poses.
        List[Path]: List of paths to the output mmcif files
    """
    # make a temporary directory for the outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)

        # run CHAI-1 on the input fasta file
        # TODO: Add conditioning on experimental structure of the receptor
        candidates = run_inference(
            fasta_file=in_fasta_file,
            output_dir=out_dir,
            seed=args.seed,
            num_diffn_timesteps=args.num_diffn_timesteps,
            num_diffn_samples=args.num_diffn_samples,
            device=args.device,
            use_esm_embeddings=args.use_esm_embeddings,
            use_msa_server=args.use_msa_server,
            msa_server_url=args.msa_server_url,
            msa_directory=args.msa_directory,
            template_hits_path=args.template,
            constraint_path=Path(temp_restraints.name) if temp_restraints else None,
        )

        # sort the candidates using the ranking data
        candidates = candidates.sorted()
        # scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]
        scores = [rd.per_chain_pair_iptm[-1].mean().item() for rd in candidates.ranking_data]
        return scores, candidates.cif_paths


if __name__ == "__main__":
    # Use fork method for multiprocessing on Linux (better compatibility with CHAI-1)
    mp.set_start_method("fork", force=True)

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
        "--num_threads", type=int, default=1, help="Number of parallel processes to use"
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

    # if restraints are provided, prepare the restraints file
    temp_restraints = None
    if args.pocket_residues:
        restraints = prepare_pocket_restraints_file(args.template, args.pocket_residues)
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_restraints:
            temp_restraints.write(restraints)

    ######################################################################################
    ################ Run inference on each of the ligands in the fasta file ##############
    ######################################################################################

    list_of_scores: List[List[float]] = []
    scores_out_path = Path(args.out)

    # Get all ligands to process
    ligands_dict = ligands_fasta_to_dict(lig_fasta_path)
    total_ligands = len(ligands_dict)

    if args.num_threads == 1:
        # Single-threaded execution (original behavior)
        for ligand_id, ligand_seq in tqdm(ligands_dict.items()):
            print(f"Running CHAI-1 for ligand {ligand_id} with sequence {ligand_seq}")

            # create a temporary fasta file for the ligand, receptor pair
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_fasta:
                with open(rec_fasta_path, "r") as rec_fasta:
                    temp_fasta.write(rec_fasta.read())
                temp_fasta.write("\n")
                temp_fasta.write(f">{ligand_id}\n{ligand_seq}\n")

            temp_fasta_in_path = Path(temp_fasta.name)
            scores, cif_paths = main(args, temp_fasta_in_path, temp_restraints)
            list_of_scores.append(scores)

            # Clean up temporary file
            temp_fasta_in_path.unlink()
    else:
        # Multi-process execution
        print(
            f"Starting parallel processing with {args.num_threads} processes for {total_ligands} ligands"
        )

        # Prepare arguments for worker processes
        args_dict = vars(args)
        temp_restraints_path = temp_restraints.name if temp_restraints else None

        # Create work items
        work_items = []
        for job_index, (ligand_id, ligand_seq) in enumerate(ligands_dict.items()):
            work_items.append((ligand_id, ligand_seq, job_index))

        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=args.num_threads) as executor:
            # Submit all jobs
            futures = []
            for work_item in work_items:
                future = executor.submit(
                    worker_process, args_dict, str(rec_fasta_path), temp_restraints_path, work_item
                )
                futures.append(future)

            # Collect results as they complete
            results = {}
            for future in as_completed(futures):
                try:
                    job_index, scores = future.result()
                    results[job_index] = scores
                except Exception as e:
                    print(f"Worker process failed: {e}")
                    # Handle failed jobs by finding which one failed
                    for i in range(total_ligands):
                        if i not in results:
                            results[i] = []
                            break

        # Sort results by job index to maintain order
        list_of_scores = [results[i] for i in range(total_ligands)]

    # Save the list of lists of scores to the output file as a .npz file
    np.save(scores_out_path, np.array(list_of_scores))

    t1 = time.time()
    print(f"Total time taken: {t1 - t0:.2f} seconds")
