import subprocess as sp
import wandb

"""Scans a sweep for a given target varname and launches inference for them via the runaiSubmitTSViT_infer.sh script.
Modify inference year there."""

api = wandb.Api()
entity, project, sweep_id  = "wandb_usrname", "project_name" , "sweep_id"
sweep = api.sweep(entity + "/" + project + "/" + sweep_id)

target_var_pattern = "SISGHI-No-Horizon"
start_date_infer = "2019-01-01T00"
end_date_infer = "2020-01-01T00"
batch_size = "128" # 512


run_name_list = []
for run in sweep.runs:
    input_var_names = run._attrs["config"]["DATASETS"]["train"]["input_var_names"]
    target_var_names = run._attrs["config"]["DATASETS"]["train"]["target_var_names"]
    # add filters here if needed
    if run.state=="finished":
        print(f"Adding run {run.name}")
        run_name_list.append(run.name)
    else:
        print(f"Run {run.name} qualifies but state is: {run.state}")
        print("passing")
do_proceed = input(f"Found {len(run_name_list)} finished runs to start inference for. Proceed (y/n)? ")

if "n" in do_proceed:
    print("Exiting")
else:
    print(f"Running inference on {start_date_infer}-{end_date_infer}")
    bash_script = "configs/Helio/2_inferences/batch_TSViT_infer.sh"
    for run_name in run_name_list:
        print(run_name)
        sp.call(['bash', bash_script, run_name, start_date_infer, end_date_infer, batch_size])

    