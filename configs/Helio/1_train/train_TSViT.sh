#!/bin/bash
python train_and_eval/regression_training_transf.py\
       --project_name "my_project_name"\
       --exp_name  "my_experiment_name"\
       --architecture "TSViT_reg" \
       --train \
       --dataset "Helio_sza80"\
       --config configs/Helio/TSViT_reg.yaml \
       --max_seq_len 40\
       --img_res 48\
       --batch_size 2\
       --num_workers 8\
       --start_date_train "2015-01-01T00"\
       --end_date_train "2017-01-01T00"\
       --start_date_eval "2018-01-01T00"\
       --end_date_eval "2019-01-01T00" \
       --dset_patching\
       --num_epochs 10\
       --input_var_names "[SAA,SZA,HRV,IR_039,WV_062,WV_073,IR_087,IR_097,IR_108,IR_120,IR_134,SRTMGL3_DEM,slope,aspectSin,aspectCos,sw_dir_cor]"\
       --target_var_names "[SISGHI-No-Horizon]"\
       --plot_img\
       --plot_scatter\
       --plot_qq\

       # --wandb