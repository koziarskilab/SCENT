import argparse
import itertools
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Literal

import gin

from gin_config import get_time_stamp, gin_config_to_readable_dictionary
from rgfn.trainer.logger.logger_base import LoggerBase
from rgfn.trainer.trainer import Trainer
from rgfn.utils.helpers import infer_metric_direction, seed_everything

BEST_PARAM = "@best_param"


@gin.configurable
def hyperparameter_search(
    base_run_name: str,
    base_config_path: str,
    params: List[Dict[str, List[Any]]] | Dict[str, List[Any]],
    logger: LoggerBase,
    best_metric: str = "loss",
    metric_direction: Literal["auto", "min", "max"] = "auto",
    seed: int = 42,
    skip: int = 0,
    search_mode: Literal["grid", "random"] = "grid",
    num_searches: int = 10,
):
    assert search_mode in [
        "grid",
        "random",
    ], "Search mode must be either 'grid' or 'random'"
    metric_direction = (
        infer_metric_direction(best_metric) if metric_direction == "auto" else metric_direction
    )
    best_valid_metrics: Dict[str, float] = {}
    best_parameters: Dict[str, Any] = {}

    logger.log_code("rgfn")
    logger.log_to_file(gin.operative_config_str(), "grid_operative_config")
    logger.log_to_file(gin.config_str(), "grid_config")
    logger.log_to_file(json.dumps(params, indent=2), "grid_params")
    logger.close()
    params_list = [params] if isinstance(params, dict) else params

    def _build_search_dicts(params_list):
        all_grid_dicts = []
        for param_dict in params_list:
            keys, values = zip(*param_dict.items())
            if search_mode == "grid":
                grid_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
            elif search_mode == "random":
                if num_searches > len(list(itertools.product(*values))):
                    grid_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
                else:
                    sampled_combinations = random.sample(
                        list(itertools.product(*values)), num_searches
                    )
                    grid_dicts = [dict(zip(keys, v)) for v in sampled_combinations]
            else:
                raise ValueError(f"Invalid search mode: {search_mode}")
            all_grid_dicts.extend(grid_dicts)
        return all_grid_dicts

    all_grid_dicts = _build_search_dicts(params_list)

    for idx, grid_dict in enumerate(all_grid_dicts):
        if idx < skip:
            continue
        print(f"Running experiment {idx} with parameters {grid_dict}")
        if "SLURM_JOB_ID" in os.environ:
            # If we're in a SLURM environment, use the job id as the run id
            slurm_id = os.environ["SLURM_JOB_ID"]
            experiment_name = f"{base_run_name}/params_{idx}/{slurm_id}"
        else:
            experiment_name = f"{base_run_name}/params_{idx}/{get_time_stamp()}"
        bindings = [f'run_name="{experiment_name}"']
        grid_dict = {
            key: (best_parameters[key] if value == BEST_PARAM else value)
            for key, value in grid_dict.items()
        }
        for key, value in grid_dict.items():
            if isinstance(value, str) and not (value.startswith("@") or value.startswith("%")):
                binding = f'{key}="{value}"'
            else:
                binding = f"{key}={value}"
            bindings.append(binding)

        config_files = [base_config_path]
        for key, value in grid_dict.items():
            if key.startswith("config_file"):
                config_files.append(value)
        gin.clear_config()
        gin.parse_config_files_and_bindings(config_files, bindings=bindings)
        run_seed = seed
        for key, value in grid_dict.items():
            if key == "seed":
                run_seed = int(value)
        seed_everything(run_seed)
        trainer = Trainer()
        trainer.logger.log_code("rgfn")
        trainer.logger.log_to_file("\n".join(bindings), "bindings")
        trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
        trainer.logger.log_to_file(gin.config_str(), "config")
        trainer.logger.log_config(grid_dict)
        trainer.logger.log_hyperparameters(
            gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)
        )
        valid_metrics = trainer.train()
        trainer.close()

        if metric_direction == "min":
            is_better = valid_metrics[best_metric] < best_valid_metrics.get(
                best_metric, float("inf")
            )
        else:
            is_better = valid_metrics[best_metric] > best_valid_metrics.get(
                best_metric, float("-inf")
            )
        if is_better:
            best_valid_metrics = valid_metrics
            best_parameters = grid_dict | {"id": f"params_{idx}"}

    json_best_parameters = json.dumps(best_parameters, indent=2)
    json_best_valid_metrics = json.dumps(best_valid_metrics, indent=2)

    logger.restart()
    logger.log_to_file(json_best_parameters, "best_params")
    logger.log_to_file(json_best_valid_metrics, "best_valid_metrics")
    logger.close()

    print(f"Best parameters:\n{json_best_parameters}")
    print(f"Best valid metrics:\n{json_best_valid_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    skip = args.skip
    config_path = args.cfg

    config_name = Path(config_path).stem
    run_name = f"{config_name}/{get_time_stamp()}"
    bindings = [f'run_name="{run_name}"']
    gin.parse_config_files_and_bindings([config_path], bindings=bindings)
    hyperparameter_search(base_run_name=run_name, base_config_path=config_path, skip=skip)
