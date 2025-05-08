run_name=$1
start_date=$2
end_date=$3
batch_size=$4

python train_and_eval/regression_training_transf.py\
  --project_name "TSViT-reg-v3"\
  --inference\
  --dataset Helio_sza80\
  --config models/saved_models/Helio/TSViT-reg/${run_name}/config_file.yaml\
  --start_date_infer "${start_date}" --end_date_infer "${end_date}"\
  --batch_size ${batch_size}\
  --wandb"\
                    