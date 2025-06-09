import os
from pathlib import Path

import gin

from rgfn import ROOT_DIR
from rgfn.trainer.trainer import Trainer
from rgfn.utils.helpers import seed_everything


def helper__test_training__runs_properly(
    config_path: str,
    config_override_str: str,
    tmp_path: Path,
    n_iterations: int = 2,
    forward_n_trajectories: int = 2,
    replay_n_trajectories: int = 2,
):
    """
    A helper function that tests whether the training runs properly. It sets the random seed, changes the working
    directory to the root directory of the project, parses the config file, overrides the config with the provided
    override string, and runs the training for one iteration.

    Args:
        config_path: a path to the config file to be used for training
        config_override_str: a string that overrides the config
        tmp_path: a path to the temporary directory where the results of the training will be saved
    """
    os.chdir(ROOT_DIR)
    seed_everything(42)

    # save config override to file
    config_override_path = tmp_path / "config_override.gin"
    with open(config_override_path, "w") as f:
        f.write(config_override_str)

    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [config_path, config_override_path],
        bindings=[
            f'run_name="test"',
            "WandbLogger.mode='offline'",
            f"user_root_dir='{tmp_path}'",
            f"Trainer.train_forward_n_trajectories={forward_n_trajectories}",
            f"Trainer.train_replay_n_trajectories={replay_n_trajectories}",
        ],
    )
    trainer = Trainer(n_iterations=n_iterations)
    trainer.train()
    trainer.close()
    gin.clear_config()
