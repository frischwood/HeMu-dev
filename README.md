# HeMu-dev

This repository contains the code and instructions to reproduce the results from the publication [*Retrieval of Surface Solar Radiation through Implicit Albedo Recovery from Temporal Context*](), in which the HelioMont emulator, namely HeMu, was presented. 


If you use this code or data in your research, please cite:

```bibtex
[Add citation information when paper is published]
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/frischwood/HeMu-dev.git
cd HeMu-dev
```

2. Create and activate the conda environment (adapt cuda and pytorch version to the available hardware):
```bash
conda env create -f environment.yml
conda activate HeMu-dev
```

## Data

### Ready-to-use
The ready-to-use data required for reproducing the results is available on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15342144.svg)](https://doi.org/10.5281/zenodo.15342144)

To download and prepare the data:
1. Download the dataset from the Zenodo repository
2. Extract the content to the `data/Helio` directory in this repository

### Download + pre-processing
All data used is publicly available and can be downloaded independently. The download and pre-processing steps are removed here to simplify reproduction, but are available in the production version of HeMu (see [Related links](#related-links))

## Repository Structure
```
HeMu-dev/
├── configs/                    # Configuration files
│   └── Helio/
│       ├── 0_sweeps/          # W&B sweep configuration files
│       ├── 1_train/           # Training configuration files
│       └── 2_inferences/      # Inference configuration files
├── data/                      # Data directory 
    └── Helio/                 # Extract downloaded directory here
        ├── 2015_2020/         # Contains all variables (download!)
│       └── stats/             # Statistics for each variable over 2015-2020 (download!)
├── models/
│   └── saved_models/          # Saved model checkpoints and logs
│       └── Helio/
├── train_and_eval/            # Training and evaluation scripts
├── inference/                 # Generated inference results
├── environment.yml            # Dependencies file
└── README.md
```

## Reproduction of the results

### 1. Prerequisite 
- Weight&Biases (wandb): We use wandb sweep agents to run experiments on context size, permutation tests, ablation, etc.
Login to your [wandb acount](https://wandb.ai/site) and get your API key. Create a new file: ```configs/Helio/wandb_api.key``` and paste your api key in it. Make sure to **remove this file from your versioning system**. 
- Hardware: all experiments were run with an NVIDIA A100-80GB GPU and 128GB of CPU RAM. For smaller capacities, consider reducing the batch_size and/or the experiment time period (all the data is pre-loaded in RAM).

### 2. Experiment sweeps and single training runs:
To launch a sweep:
- Set the path to the sweep .yaml file in  ```configs/Helio/0_sweeps/ini_sweep_reg.py``` and run the script. The wandb CLI will return a sweep ID. The sweeps are defined in the config .yaml files in ```configs/Helio/0_sweeps```. On your wandb dashboard a new sweep was created and is pending.
- Adapt the sweep ID in ```train_and_eval/launch_sweep_agent.py``` and run the script with the following command:
    ```bash
    python train_and_eval/launch_sweep_agent.py --sweep_id <wand_usrname>/<project_name>/<sweep_id> --agent_exp_count <runs_per_agent>
    ```  
    This will launch a wandb agent that will run the different runs of the sweep. The argument ```--agent_exp_count``` sets the number of runs from the sweep the agent will run sequentially. Multiple agents can be launched in parallel. Note: make sure here you're logged-in to your wandb account ```wandb login```.

To launch a single training run:
- Modify project name and experiment names in the bash scripts in ```configs/Helio/1_train/``` and run them according to the desired architecture. 

### 3. Inference
The inference run will produce SSR estimates maps for a given time period based on a trained model. Sweeps and train runs save their runs (loggings, checkpoints, etc.) in subfolders of ```models/saved_models/Helio```. 
- To launch inference of a single run: find the path to the corresponding config.yaml file and paste it in ```configs/Helio/2_inferences/run_infer.sh``` and run the script. A new ```inference/``` folder will be created where the infered files (.netcdf) will be placed.
- To launch inference on all runs of a sweep: set wandb_user_name, project_name and sweep_id in ```configs/Helio/2_inferences/batch_infer.py``` and run the script. This will run an inference for all runs of the sweep in parallel, which might not be supported by the available hard ware. Modify accordingly. 

### 4. Plots

## Related links
A production version of HeMu is in development: [https://github.com/frischwood/HeMu.git](https://github.com/frischwood/HeMu.git)\
HeMu inferences will be accessible and downloadable at [https://HeMu.epfl.ch](https://HeMu.epfl.ch).


