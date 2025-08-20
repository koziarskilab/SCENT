The process of obtaining the results is quite complicated. It involves three steps:
1. Save the results following the structure:
```
results/
├── template_name (e.g. rgfn_new_filtered)
│   ├── task_name (e.g. seh)
│   │   ├── model_name (e.g. rgfn_is_decomposable)
│   │   │   ├── sampling (corresponds to final_paths.csv for 3 seeds)
│   │   │   │   ├── paths_0.csv
│   │   │   │   ├── paths_1.csv
│   │   │   │   ├── paths_2.csv
│   │   │   ├── training (corresponds to paths.csv for 3 seeds)
│   │   │   │   ├── paths_0.csv
│   │   │   │   ├── paths_1.csv
│   │   │   │   ├── paths_2.csv

```
you can use a `download.ipynb` scripts to download files conveniently. You can also use `get_run_names_from_wandb.ipynb` to filter the runs you want to download.

2. Run `plot_training_results.ipynb` for the model_names your are interested in. The results will be cached in corresponding folders in `results/` directory.
3. Run `plot_final_results.ipynb` to obtain the final results. The results will be cached in corresponding folders in `results/` directory. You will obtain a latex table copied to your clipboard. You can paste it to your latex document.
