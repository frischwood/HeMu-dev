from data.Helio.dataloader import get_dataloader as get_helio_dataloader
from data.Helio.data_transforms import Helio_reg_transform
from utils.config_files_utils import read_yaml


DATASET_INFO = read_yaml("data/datasets.yaml")


def get_dataloaders(config):

    model_config = config['MODEL']
    dataset_config = config['DATASETS']
    train_config = dataset_config['train']
    eval_config  = dataset_config['eval']
    infer_config  = dataset_config['infer']
    dataloaders = {}
    
    # TRAIN data -------------------------------------------------------------------------------------------------------
    train_config['base_dir'] = DATASET_INFO[train_config['dataset']]['basedir']
    train_config['paths'] = DATASET_INFO[train_config['dataset']]['paths_train']
    dataset_config['train']['path_stats_csv'] = DATASET_INFO[train_config['dataset']]['paths_stats'] # not nice but ok
    eval_config['base_dir'] = DATASET_INFO[eval_config['dataset']]['basedir']
    eval_config['paths'] = DATASET_INFO[eval_config['dataset']]['paths_eval']
    dataset_config['eval']['path_stats_csv'] = DATASET_INFO[eval_config['dataset']]['paths_stats']
    infer_config['base_dir'] = DATASET_INFO[infer_config['dataset']]['basedir']
    infer_config['paths'] = DATASET_INFO[infer_config['dataset']]['paths_eval']
    dataset_config['infer']['path_stats_csv'] = DATASET_INFO[infer_config['dataset']]['paths_stats']
    
    if model_config["train"] & ('Helio' in train_config['dataset']):
        
        dataloaders['train'] = get_helio_dataloader(
            start_date=train_config["start_date"], end_date=train_config["end_date"], 
            input_var_names=train_config["input_var_names"], target_var_names=train_config["target_var_names"],
            root_dir=train_config['base_dir'], img_res=model_config["img_res"], max_seq_len=model_config["max_seq_len"],
            dset_patching=train_config["dset_patching"],transform=Helio_reg_transform(model_config, dataset_config, is_training=True),
            reg_mode=model_config["reg_mode"],
            batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'],
            infer=False)
        
    # EVAL data --------------------------------------------------------------------------------------------------------
        dataloaders['eval'] = get_helio_dataloader(
            start_date=eval_config["start_date"], end_date=eval_config["end_date"], 
            input_var_names=eval_config["input_var_names"], target_var_names=eval_config["target_var_names"],
            root_dir=eval_config['base_dir'], img_res=model_config["img_res"], max_seq_len=model_config["max_seq_len"],
            dset_patching=eval_config["dset_patching"],transform=Helio_reg_transform(model_config, dataset_config, is_training=True),
            reg_mode=model_config["reg_mode"],
            batch_size=eval_config['batch_size'], shuffle=True, num_workers=eval_config['num_workers'],
            infer=False)  

    # INFER data --------------------------------------------------------------------------------------------------------
    if model_config["inference"] & ('Helio' in infer_config['dataset']):
        dataloaders['infer'] = get_helio_dataloader(
                start_date=infer_config["start_date"], end_date=infer_config["end_date"], 
                input_var_names=infer_config["input_var_names"], target_var_names=infer_config["target_var_names"],
                root_dir=infer_config['base_dir'], img_res=model_config["img_res"], max_seq_len=model_config["max_seq_len"], dset_patching=infer_config["dset_patching"],
                transform=Helio_reg_transform(model_config, dataset_config, is_training=False),
                reg_mode=model_config["reg_mode"],
                batch_size=infer_config['batch_size'], shuffle=False, num_workers=infer_config['num_workers'],
                infer=True)  
    
    return dataloaders

