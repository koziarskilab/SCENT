import pickle
from glob import glob

import torch

torch.multiprocessing.set_sharing_strategy("file_system")

import warnings

from option import set_args

warnings.filterwarnings("ignore")


def get_problematic(args, conf_size=10):
    problematic_set = set()
    name_list = glob(f"{args.input_path}/*.pkl")
    for idx in range(conf_size * len(name_list)):
        data_idx = idx // conf_size
        conf_idx = idx % conf_size
        file_path = open(name_list[data_idx], "rb")
        data = pickle.load(file_path)
        file_path.close()
        try:
            data["coordinates"] = data["coordinates"][conf_idx]
        except IndexError:
            problematic_set.add(name_list[data_idx])
    return problematic_set


if __name__ == "__main__":
    parser = set_args()
    parser.add_argument(
        "--input_path",
        type=str,
        default="example_processed_data",
        help="Path of processed dataset",
    )
    args = parser.parse_args()

    problematic_mols = get_problematic(args)
    with open(f"{args.input_path}/problematic_mols.log", "w") as f:
        f.write("\n".join(problematic_mols))
