"""Launches an agent of a sweep, the count/agent should be total runs/nb_par_agents
In the case of RCP: quota=6 ==> if total=72 so count=12"""
import wandb
import argparse
import os

def main(args):
    with open("configs/Helio/wandb_api.key") as key:
        os.environ["WANDB_API_KEY"] = key.read()
    wandb.login()
    wandb.agent(args.sweep_id, count=args.agent_exp_count)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Wandb Sweep Launcher')
    parser.add_argument('--sweep_id', help='id of sweep to launch from')
    parser.add_argument('--agent_exp_count', type=int, help='number of sweep trials per agent')
    args = parser.parse_args()
    main(args)
