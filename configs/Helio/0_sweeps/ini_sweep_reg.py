"""use to initiate the sweep defined in the yaml file defining the sweep.
Copy the returned sweep_id in the sweep submitting file"""
import wandb
from yaml import load, SafeLoader

project_name = "your_project_name"
yaml_file = "configs/Helio/0_sweeps/sweep_contextSizeExp.yaml"
with open(yaml_file, 'r') as config_file:
    sweep_configuration = load(config_file,Loader=SafeLoader)

wandb.sweep(sweep=sweep_configuration, project=project_name)

