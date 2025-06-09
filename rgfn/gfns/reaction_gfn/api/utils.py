from pathlib import Path
from typing import List


def read_txt_file(file_path: Path) -> List[str]:
    objects = []
    with open(file_path, "r") as f:
        for line in f:
            objects.append(line.strip())

    return objects
