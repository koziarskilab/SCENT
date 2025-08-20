import hashlib
import re
from io import StringIO
from pathlib import Path

from Bio import PDB
from openbabel import openbabel


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


def chai_hash(sequence: str) -> str:
    """
    Hash a sequence using SHA256. Used here and by ChaiDocker to generate the MSA cache filename.
    """
    hash_object = hashlib.sha256(sequence.upper().encode())
    hash_object_hex = hash_object.hexdigest()
    return f"{hash_object_hex}.aligned.pqt"


def convert_pdbqt_file_to_pdb_str(input_file: str | Path, out_path: Path | None = None) -> str:
    """
    Convert a PDBQT file to a PDB string.
    """
    ob_conversion = openbabel.OBConversion()
    ob_conversion.SetInAndOutFormats("pdbqt", "pdb")
    molecule = openbabel.OBMol()
    ob_conversion.ReadFile(molecule, str(input_file))
    pdb_str = ob_conversion.WriteString(molecule)

    if out_path is not None:
        ob_conversion.WriteFile(molecule, str(out_path))

    return pdb_str


def get_sequence_from_pdb(pdb_str: str) -> str:
    """
    Extract protein sequence in one-letter code from PDB string.
    """
    # Create a PDB parser and structure from the string
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", StringIO(pdb_str))

    # Extract sequence using PPBuilder (Polypeptide Builder)
    ppb = PDB.PPBuilder()
    sequence = ""
    for pp in ppb.build_peptides(structure):
        sequence += str(pp.get_sequence())

    return sequence


def fastas_from_pdb(path: str | Path) -> str:
    """
    Extract the FASTA sequence from a PDB file.
    Args:
        path: path to the PDB file

    Returns:
        str: FASTA sequence of the PDB file
    """

    # set protein name as the file name
    protein_name = Path(path).stem

    # pattern to match CA atoms
    ca_pattern = re.compile(
        "^ATOM\s{2,6}\d{1,5}\s{2}CA\s[\sA]([A-Z]{3})\s([\s\w])|^HETATM\s{0,4}\d{1,5}\s{2}CA\s[\sA](MSE)\s([\s\w])"
    )

    # accumulate the one-letter codes of aminoacids for each chain in a dictionary
    chain_dict = dict()
    with open(path, "r") as fp:
        for line in fp.read().splitlines():
            match_list = ca_pattern.findall(line)
            if match_list:
                resn = match_list[0][0] + match_list[0][2]
                chain = match_list[0][1] + match_list[0][3]
                if chain in chain_dict:
                    chain_dict[chain] += get_one_letter_aa_code(resn)
                else:
                    chain_dict[chain] = get_one_letter_aa_code(resn)

    # list FASTA strings
    fastas_list = [
        f">protein|name={protein_name}_CHAIN_{chain}\n" + chain_dict[chain] + "\n"
        for chain in chain_dict.keys()
    ]

    return "".join(fastas_list)
