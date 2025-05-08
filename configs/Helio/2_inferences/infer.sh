#!/bin/bash
python train_and_eval/regression_training_transf.py\
    --project_name "my_project_name"\
    --inference\
    --dataset Helio_sza80\
    --config # path to config file of trained model to infer on\
    --start_date_infer "2019-01-01T00"\
    --end_date_infer "2020-01-01T00"\
    --batch_size 128\
    # --wandb\

 