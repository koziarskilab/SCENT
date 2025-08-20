import argparse
from pathlib import Path

from chai_lab.data.dataset.msas.colabfold import generate_colabfold_msas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", "-s", type=str, help="sequence to search MSA for")
    parser.add_argument("--msa-dir", "-d", type=str, help="Path to cache searching results")
    parser.add_argument(
        "--server-url",
        "-u",
        type=str,
        default="https://api.colabfold.com",
        help="server for MSA search",
    )
    args = parser.parse_args()
    generate_colabfold_msas(
        protein_seqs=[args.sequence],
        msa_dir=Path(args.msa_dir),
        search_templates=True,
        msa_server_url=args.server_url,
    )
