# Parallel sampling from trained RGFN/SCENT models

This directory contains scripts that can be used to sample pools of molecules from a trained RGFN or SCENT policy. This is a work in progress, so there might be some bugs or missing features. Please contact [Stephen]() for any issues or suggestions.

## Overview

The `sample.py` script implements sophisticated parallel sampling methods that can generate diverse molecule mixtures from trained RGFN/SCENT models. The approach follows a two-stage process:

1. **Initial trajectory sampling**: Sample initial trajectories to generate scaffold molecules
2. **Backtracking and mixture generation**: Backtrack from the scaffold and generate diverse mixtures using various sampling strategies

## Key Features

### Backtracking Strategies

The script supports multiple backtracking strategies to diversify scaffold molecules:

- **SIMPLE**: Basic backtracking approach
- **LSD** (Last Step Diversification): Standard diversification at the final reaction step
- **ASMS**: Advanced Scaffold Modification Strategy

### Selection Strategies

Multiple selection strategies are available for generating mixtures:

- **RANDOM**: Randomly sample molecules around the scaffold
- **BOLTZMANN**: Temperature-controlled sampling with adjustable temperature
- **TOPK**: Sample only from top-k percent of actions
- **TOPP**: Sample from top-p percent cumulative probability actions
- **BEAM**: Beam search maintaining a beam width equal to mixture size
- **ENUM**: Enumerate molecules (similar to random sampling)
- **MCTS**: Monte Carlo Tree Search (implementation in progress)
- **GA**: Genetic Algorithm approach

## Usage

### Basic Usage

```bash
python sample.py --cfg <config_file> --checkpoint_path <model_checkpoint>
```

### Command Line Arguments

- `--cfg`: Path to configuration file (default: None)
- `--seed`: Random seed for reproducibility (default: 42)
- `--verbose`: Enable verbose output
- `--checkpoint_path`: Path to trained model checkpoint
- `--output_path`: Output CSV file path (default: "samples.csv")
- `--n_backtrack_steps`: Number of backtracking steps (default: 2, must be > 1)
- `--n_mixtures`: Number of trajectories to sample (default: 100)
- `--mixture_size`: Number of refined samples per trajectory (default: 12)
- `--backtrack_strategy`: Backtracking strategy - choices: SIMPLE, LSD, ASMS (default: LSD)
- `--selection_strategy`: Selection strategy - choices: RANDOM, BOLTZMANN, TOPK, TOPP, BEAM, ENUM, MCTS, GA (default: TOPK)
- `--temperature`: Temperature for sampling (default: 1.0)
- `--top_p`: Top-p sampling threshold (default: 0.9)
- `--top_k`: Top-k sampling threshold (default: 0.7)

### Example

```bash
python sample.py \
    --cfg experiments/scent_qed_proxy/logs/config.txt \
    --checkpoint_path experiments/scent_qed_proxy/train/checkpoints/last_gfn.pt \
    --n_mixtures 50 \
    --mixture_size 10 \
    --backtrack_strategy LSD \
    --selection_strategy BOLTZMANN \
    --temperature 1.5 \
    --output_path my_samples.csv
```

## Output Format

The script generates a CSV file with the following columns:

- `tid`: Trajectory ID (e.g., "traj_0", "traj_1", ...)
- `smiles`: SMILES string of the molecule
- `proxy`: Proxy reward value for the molecule
- `is_scaffold`: Boolean indicating if this is the scaffold molecule (True) or a diversified molecule (False)

Each trajectory produces one scaffold molecule and multiple diversified molecules, all sharing the same trajectory ID.

## Algorithm Details

### Parallel Sampling Process

1. **Source State Sampling**: Generate initial source states for trajectory sampling
2. **Trajectory Generation**: Sample forward trajectories using the trained policy
3. **Backtracking**: Apply the selected backtracking strategy to create diversification points
4. **Mixture Generation**: Use the selection strategy to generate diverse molecules around each scaffold
5. **Reward Computation**: Calculate proxy rewards for all generated molecules

### Last Step Diversification (LSD)

The standard approach for obtaining molecule mixtures is Last Step Diversification, which:

- Diversifies scaffold molecules at the last reaction in their trajectory
- Reacts scaffolds with diverse sets of co-reactants
- Provides pools of related molecules with similar scaffolds
- Useful for lead optimization and local chemical exploration
