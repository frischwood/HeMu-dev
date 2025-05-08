import sys
import os
import glob
import signal
from typing import Dict, Any
import hashlib
import json
from datetime import datetime
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from utils.early_stopper import ArchaicEarlyStopper
import wandb
from torch.utils.tensorboard import SummaryWriter
from data.Helio.data_transforms import Denormalize
import numpy as np
import xarray as xr
from pandas import to_datetime
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_reg_metrics
from metrics.loss_functions import get_loss
from utils.summaries import * 

def handler(signum, frame):
    """Used for graceful ctrl+c exits"""
    print(' ==> Got CTRL+C: exiting gracefully')
    exit (0)

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def print_header(config):
    
    print("-----------------------------")
    print(f"HELIO-DML")
    print("-----------------------------")
    print("MODE:") 
    print(f"train={config["MODEL"]["train"]}")
    print(f"infer={config["MODEL"]["inference"]}")
    print("-----------------------------")
    print(f"OUTPUT FOLDER: {config['CHECKPOINT']["save_path"]}")
    print(f"WANDB: {config['wandb']}")
    print("-----------------------------")
    print(f"CONFIG:")
    for key, item in config.items():
        print(f"--->{key}:{item}")
    print("-----------------------------")
    print(f"STARTED: {datetime.now()}")
    print("-----------------------------")
       
def create_parser():
    """parses necessary settings. Default values are modified for best values."""
    class ParseAction_str(argparse.Action):
        """forms a list object from str list"""
        def __call__(self, parser, namespace, values, option_string=None):
            values = [str(v.strip()) for v in values[1:-1].split(",")]
            setattr(namespace, self.dest, values)
    class ParseAction_int(argparse.Action):
        """forms a list object from str list"""
        def __call__(self, parser, namespace, values, option_string=None):
            values = [int(v.strip()) for v in values[1:-1].split(",")]
            setattr(namespace, self.dest, values)
            
    parser = argparse.ArgumentParser(description='Helio-DML Train-Test-Infer')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--project_name', type=str, default="TSViT-reg-v3",
                        help='project name, used by wandb',)
    parser.add_argument("--exp_name", type=str, default="exp_name",
                        help="Name of experiment. In training mode, a config hash is appended. In inference mode, it must exactly match the experiment to use checkpoints form.")
    parser.add_argument("--save_path", type=str, default="models/saved_models/Helio/TSViT-reg",
                        help="Where to save experiments folder at.")
    parser.add_argument("--dataset", type=str, default="Helio_sza80")
    parser.add_argument("--architecture", type=str, default="TSViT_reg")
    parser.add_argument('--device', default='0', type=str,help='gpu ids to use')
    parser.add_argument("--start_date_train", type=str, default="2015-01-01T00")
    parser.add_argument("--end_date_train", type=str, default="2019-01-01T00")
    parser.add_argument("--start_date_eval", type=str, default="2019-01-01T00")
    parser.add_argument("--end_date_eval", type=str, default="2020-01-01T00")
    parser.add_argument("--start_date_infer", type=str, default="2015-01-01T00")
    parser.add_argument("--end_date_infer", type=str, default="2020-01-01T00")
    parser.add_argument("--input_var_names",
                        default=["sw_dir_cor", "slope", "aspectSin", "aspectCos", "SRTMGL3_DEM", "SAA", "SZA", "HRV", "IR_016", "IR_039", "WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"],
                        action=ParseAction_str) # nargs="+"
    parser.add_argument("--target_var_names",
                        default=["SISGHI-No-Horizon"], # SISDIR-No-Horizon", "SISDIF-No-Horizon
                        action=ParseAction_str) # nargs="+"

    parser.add_argument("--img_res", type=int, default=48)
    parser.add_argument("--max_seq_len", type=int, default=40)
    parser.add_argument("--patch_size", type=int, default=3)

    parser.add_argument("--ds_img_res", default=[241,103], action=ParseAction_int)
    parser.add_argument("--dset_patching", action="store_true", default=False, help="Applies ""block"" patching of ds into img_res patches.") # if None data 
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--alpha_regLoss", type=int, default=1) 
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--earlyStopping", action="store_true", default=False)
    parser.add_argument("--patience", type=float, default=20) # patience is nb of eval (!=epochs since eval is not done) every epoch)

    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--eval_epochs", type=int, default=10)
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--train_metrics_steps", type=int, default=200)
    parser.add_argument("--use_hod", action="store_true", default=False, help="enables use of hod in addition to doy")
    parser.add_argument("--plot_img", action="store_true", default=False)
    parser.add_argument("--plot_scatter", action="store_true", default=False)
    parser.add_argument("--plot_qq", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--rerun", action="store_true", default=False)
    parser.add_argument("--perm_test", default=False, help="If not False (default), must be one of input vars, which will be shuffled.")
    return parser

def update_config(args):
    """Updates the config file
    If training mode: all entries of the config file are modified. 
    If Inference mode: most of the settings are taken from the provided config file (path).
    Only wandb usage and start and end dates are updated in inference mode, insuring a good use of the trained model (checkpoint)."""
    # read yaml config file
    config = read_yaml(args.config)
    if args.inference or args.train: 
        # set keys that need to be entered in any case (except rerun)
        config['local_device_ids'] = [int(d) for d in args.device.split(',')]
        config['wandb'] = args.wandb
        config["MODEL"]['train'] = args.train
        config["MODEL"]['inference'] = args.inference
    # if inference only update few keys, rest comes from training copied config file.
    if args.inference and not args.train and not args.rerun:
        config["DATASETS"]["infer"]["start_date"] = args.start_date_infer
        config["DATASETS"]["infer"]["end_date"] = args.end_date_infer
        config["DATASETS"]["infer"]["dataset"] = args.dataset
        config["DATASETS"]["infer"]["dset_patching"] = args.dset_patching
        config["DATASETS"]["infer"]["batch_size"] = args.batch_size
        config["DATASETS"]["infer"]["perm_test"] = args.perm_test
        # config["MODEL"]["max_seq_len"] = args.max_seq_len
        print(f"Inference mode: config taken from: {args.config}")
        print(f"--> Only modifying start and end dates, wandb use.")
    # else if training, set all config entries as entered by user
    elif not args.inference and args.train and not args.rerun:
        config["MODEL"]['project_name'] = args.project_name
        # datasets
        config["DATASETS"]["train"]["dataset"] = args.dataset
        config["DATASETS"]["train"]["start_date"] = args.start_date_train
        config["DATASETS"]["train"]["end_date"] = args.end_date_train
        config["DATASETS"]["train"]["batch_size"] = args.batch_size
        config["DATASETS"]["train"]["num_workers"] = args.num_workers
        config["DATASETS"]["train"]["input_var_names"] = args.input_var_names
        config["DATASETS"]["train"]["target_var_names"] = args.target_var_names
        config["DATASETS"]["train"]["plot_img"] = args.plot_img
        config["DATASETS"]["train"]["plot_scatter"] = args.plot_scatter
        config["DATASETS"]["train"]["plot_qq"] = args.plot_qq
        config["DATASETS"]["train"]["img_res"] = args.ds_img_res
        config["DATASETS"]["train"]["dset_patching"] = args.dset_patching

        config["DATASETS"]["eval"]["dataset"] = args.dataset
        config["DATASETS"]["eval"]["start_date"] = args.start_date_eval
        config["DATASETS"]["eval"]["end_date"] = args.end_date_eval
        config["DATASETS"]["eval"]["batch_size"] = args.batch_size
        config["DATASETS"]["eval"]["num_workers"] = args.num_workers
        config["DATASETS"]["eval"]["input_var_names"] = args.input_var_names
        config["DATASETS"]["eval"]["target_var_names"] = args.target_var_names
        config["DATASETS"]["eval"]["plot_img"] = args.plot_img
        config["DATASETS"]["eval"]["plot_scatter"] = args.plot_scatter
        config["DATASETS"]["eval"]["plot_qq"] = args.plot_qq
        config["DATASETS"]["eval"]["img_res"] = args.ds_img_res
        config["DATASETS"]["eval"]["dset_patching"] = args.dset_patching

        config["DATASETS"]["infer"]["dataset"] = args.dataset
        config["DATASETS"]["infer"]["start_date"] = args.start_date_infer
        config["DATASETS"]["infer"]["end_date"] = args.end_date_infer
        config["DATASETS"]["infer"]["batch_size"] = args.batch_size ### parallel inference!
        config["DATASETS"]["infer"]["num_workers"] = args.num_workers
        config["DATASETS"]["infer"]["input_var_names"] = args.input_var_names
        config["DATASETS"]["infer"]["target_var_names"] = args.target_var_names
        config["DATASETS"]["infer"]["img_res"] = args.ds_img_res
        config["DATASETS"]["infer"]["dset_patching"] = args.dset_patching
        config["DATASETS"]["infer"]["perm_test"] = args.perm_test
        # model
        config["MODEL"]["architecture"] = args.architecture
        # config["MODEL"]['inference_mode'] = args.inference_mode
        config["MODEL"]["img_res"] = args.img_res
        config["MODEL"]["patch_size"] = args.patch_size
        config["MODEL"]["max_seq_len"] = args.max_seq_len
        config["MODEL"]["num_channels"] = len(config["DATASETS"]["train"]["input_var_names"])
        config["MODEL"]["num_classes"] = len(config["DATASETS"]["train"]["target_var_names"])
        config["MODEL"]["use_hod"] = "hod" in args.architecture   
        # checkpoint
        config["CHECKPOINT"]["eval_epochs"] = args.eval_epochs
        config["CHECKPOINT"]["save_epochs"] = args.save_epochs
        config["CHECKPOINT"]["train_metrics_steps"] = args.train_metrics_steps
        # solver
        config["SOLVER"]["num_epochs"] = args.num_epochs 
        config["SOLVER"]["alpha_regLoss"] = args.alpha_regLoss
        config["SOLVER"]["weight_decay"] = args.weight_decay 
        config["SOLVER"]["patience"] = args.patience
        config["SOLVER"]["earlyStopping"] = args.earlyStopping
        # exp_name based on config dict
        if not args.inference: # a priori not needed anymore
            config["MODEL"]["exp_name"] = args.exp_name + "-" + dict_hash(config) #exp_name is unique hash for configuration
        else: 
            config["MODEL"]["exp_name"] = args.exp_name

        # experiment path 
        config['CHECKPOINT']['save_path'] = args.save_path
        config["CHECKPOINT"]["load_from_checkpoint"] = config['CHECKPOINT']["save_path"] + f"/{config['MODEL']['exp_name']}"
        config['CHECKPOINT']['save_path'] = config['CHECKPOINT']["save_path"] + f"/{config['MODEL']['exp_name']}" 
    elif args.rerun:
        print(f"Rerun job name: {config["MODEL"]["exp_name"]}")
        config["MODEL"]["exp_name"] = "rerun_" + config["MODEL"]["exp_name"]
    else:
        print("Mode can be either Inference, Training or Rerun, not a combination. Exiting.")
        sys.exit()
        
    return config

def get_loggingFreq(num_batch_train, num_batch_eval, batch_size, num_epochs):
        print("-----------------------------------")
        print(f"Starting for {num_epochs} epochs:")
        print("-----------------------------------")
        log_freq2Totalratio = 0.1 # log every 10% of batches of dataloader
        eval_freq2trainlLogratio = 5.0 # do eval every 5x logfreq (50%) (inverted ratio)
        log_freq_train = round(log_freq2Totalratio * num_batch_train)
        log_freq_eval = round(log_freq2Totalratio * num_batch_eval)
        eval_freq = eval_freq2trainlLogratio * log_freq_train
        
        for mode, num_batches, log_freq in zip(["train", "eval"],[num_batch_train, num_batch_eval], [log_freq_train, log_freq_eval]):
            total_samples = num_batches * batch_size
            print(f"{mode}:")
            print(f"--->1 epoch = {num_batches} batches")
            print(f"--->1 batch = {batch_size} samples")
            print(f"--->Samples total = {total_samples}")
            print(f"Logging {mode} summaries every {log_freq} steps ({log_freq*batch_size} samples ->{100*log_freq2Totalratio}% of dataset)")
            print("------------------------------------")
        
        print(f"Running eval every {eval_freq} steps ({eval_freq*batch_size} samples -> {10*eval_freq2trainlLogratio}% of dataset)")
        print("-----------------------------------")
        return log_freq_train, log_freq_eval, eval_freq

def infer(writer, dataloaders, config, device):
    """AI is creating summary for infer

    Args:
        writer ([type]): [description]
        dataloaders ([type]): [description]
        config ([type]): [description]
        device ([type]): [description]
    """
    
    def inferByPatch(input, net, patch_size, num_targets, architecture, device, isTargetCenterPix=False):
        """input format BTHWC """
        if "ViT" in architecture:
            x_dim = -2
            y_dim = -3
        elif "ResNet" in architecture:
            x_dim = -1
            y_dim = -2
        else: 
            NotImplementedError
        batch_size = input.shape[0]
        
        if not isTargetCenterPix: # covers the domain by blocks
            x_shape = input.shape[x_dim]
            y_shape = input.shape[y_dim]
            output = torch.zeros([batch_size,num_targets, y_shape, x_shape]).to(device)
            x_fullpatch = int(x_shape / patch_size)
            y_fullpatch = int(y_shape / patch_size)
            for x in range(x_fullpatch):
                x_b = x * patch_size
                x_t = (x + 1) * patch_size
                for y in range(y_fullpatch):
                    y_b = y * patch_size
                    y_t =( y + 1) * patch_size
                    output[...,:,y_b:y_t,x_b:x_t] = net(input[...,y_b:y_t,x_b:x_t,:])
                if y_shape > y_t: # take care of remaining parts in y
                    remainder_y = y_shape - y_t
                    remainder_output = net(input[...,y_b+remainder_y:,x_b:x_t,:])
                    # only add the remaining part
                    output[...,-remainder_y:,x_b:x_t] = remainder_output[...,-remainder_y:,:]
            if x_shape > x_t: # take care of remaining parts in x
                remainder_x = x_shape - x_t
                for y in range(y_fullpatch):
                    y_b = y * patch_size
                    y_t =( y + 1) * patch_size
                    remainder_output = net(input[...,y_b:y_t, x_b+remainder_x:,:])
                    # only add the remaining part
                    output[...,y_b:y_t, -remainder_x:] = remainder_output[...,-remainder_x:]
                if y_shape > y_t: 
                    remainder_y = y_shape - y_t
                    remainder_output = net(input[..., y_b+remainder_y:, x_b + remainder_x:,:])
                    # only add the remaining part
                    output[...,:,-remainder_y:, -remainder_x:] = remainder_output[...,-remainder_y:,-remainder_x:]

        else: # covers the domain by rolling window (padding required)
            assert patch_size % 2 == 1, "Make sure img_res is odd" #here patch size must be odd
            # padding first to access borders
            input = nn.functional.pad(input, pad=(patch_size//2,patch_size//2,patch_size//2,patch_size//2), value=0)
            x_shape = input.shape[x_dim]
            y_shape = input.shape[y_dim]
            output = torch.zeros([batch_size,num_targets, y_shape, x_shape]).to(device)
            for x in range(input.shape[-1]-patch_size):
                for y in range(input.shape[-2]-patch_size):
                    output[...,y+patch_size//2+1,x+patch_size//2+1] = net(input[..., y:y+patch_size, x:x+patch_size])
            # unpadding
            output = output[...,patch_size//2+1:-(patch_size//2-1),patch_size//2+1:-(patch_size//2-1)]
        return output
    
    def writeNC(array, timestamps, var, nc_canvas, write_path):
        # writing in separately (fast enough!)
        for i, timestamp in enumerate(timestamps):
            nc_canvas_t = nc_canvas.sel(time=to_datetime(timestamp, format="%Y%m%d%H%M%S"))
            da = xr.DataArray(data=array[i,...].squeeze(0), coords=nc_canvas_t.coords, dims={"y":nc_canvas_t.dims["y"], "x":nc_canvas_t.dims["x"]}, name=var)
            da = da.expand_dims({"time":[da.coords["time"].values]})
            filename = f"{var}_{to_datetime(timestamp, format="%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H%M%S")}"
            da.to_netcdf(os.path.join(write_path,filename)+".nc")

    def concatNC(data_root, var_list):
        
        for var in var_list:
            print(f"Concatenating nc files: {var}")
            filelist = glob.glob(os.path.join(data_root,f"**/*{var}*.nc"),recursive=True)
            da_list = []
            for file in filelist:
                da_list.append(xr.open_dataset(file))
                # da_list.append(da.expand_dims({"time":[da.coords["time"].values]}))
            ds_concat = xr.concat(da_list,dim="time").sortby("time").drop_duplicates(dim="time", keep="first")

            ds_concat.to_netcdf(os.path.join(data_root,f"{var}.nc"))
    
    # def plotAnalysis():
        
    net = get_model(config, device)
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    infer_write_path = os.path.join(config['CHECKPOINT']["save_path"],"inference")
    if config["DATASETS"]["infer"]["perm_test"]:
        infer_write_path = os.path.join(infer_write_path, config["DATASETS"]["infer"]["perm_test"])
    print(f"Inference: writing to {infer_write_path}")
    os.makedirs(infer_write_path,exist_ok=True)
    print(f"Inference: getting checkpoint in {checkpoint}")
    load_from_checkpoint(net, checkpoint, partial_restore=False, best_only=True)
        
    target_vars = config["DATASETS"]["train"]["target_var_names"]
    net.eval()
    with torch.no_grad():
        epoch=1
        isTargetCenterPix = "ResNet" in config["MODEL"]["architecture"]
        img_res = config["MODEL"]["img_res"]
        batch_metrics_all = []
        for step, (sample, int_target_time_specimen) in enumerate(dataloaders["infer"]):
            
            output = inferByPatch(sample["inputs"].to(device), net, img_res,
                                  len(config["DATASETS"]["infer"]["target_var_names"]),
                                  config["MODEL"]["architecture"],
                                  device,
                                  isTargetCenterPix)

            denormalizer = Denormalize(config["DATASETS"]["infer"]) 
            output = output.detach().cpu() 
            int_target_time_specimen = int_target_time_specimen.cpu()
            for i, target_var_name in enumerate(target_vars):
                output[:,i,...] = denormalizer((output[:,i,...],target_var_name))
                
                # inference has no more reference to compute biases from. 
                # sample["targets"][:,i,...] = denormalizer((sample["targets"][:,i,...],target_var_name))
                # batch_metrics = get_mean_reg_metrics(output=output[:,i,...], 
                #                                     target=sample["targets"][:,i,...],
                #                                     epoch=epoch, step=step, var_name=target_var_name)
                # batch_metrics_all.append(batch_metrics)
                # write_mean_summaries(writer, batch_metrics, batch_metrics_all, step, mean_windowSize=10, mode="infer")
                
                writeNC(output[:,i,...], int_target_time_specimen.numpy(), target_var_name, dataloaders["infer"].dataset.nc_canvas, infer_write_path)
            
            print("Infering --> abs_step: %d, epoch: %d, step: %5d" %(step + 1, epoch, step + 1)) # , mse: %.7f" %(step + 1, epoch, step + 1, batch_metrics[f"{target_var_name}/MSE"]))
        
        concatNC(infer_write_path, target_vars)
         
def train_and_evaluate(net, dataloaders, config, device, writer): 

    def train_step(net, sample, loss_fn, optimizer, device):
        optimizer.zero_grad()
        outputs = net(sample["inputs"].to(device))
        loss = loss_fn(outputs, sample["targets"].to(device))
        loss.backward()
        
        optimizer.step()
        return outputs, loss
  
    def run(net, writer, config):
        # extract from config for easier access
        # num_classes = config['MODEL']['num_classes']
        save_path = config['CHECKPOINT']['save_path']
        target_vars = config["DATASETS"]["train"]["target_var_names"]
        batch_size = config["DATASETS"]["train"]["batch_size"]
        num_epochs = config['SOLVER']['num_epochs']
        train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
        save_epochs = config['CHECKPOINT']["save_epochs"]
        checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
        num_steps_train = len(dataloaders['train'])
        num_steps_eval = len(dataloaders['eval'])
        local_device_ids = config['local_device_ids']
        weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
        lr = float(config['SOLVER']['lr_base'])
        earlyStopping = config["SOLVER"]["earlyStopping"]
        
        torch.cuda.empty_cache()
        if checkpoint:
            print(f"Checking for checkpoint in {checkpoint}")
            load_from_checkpoint(net, checkpoint, partial_restore=False)

        if len(local_device_ids) > 1:
            net = nn.DataParallel(net, device_ids=local_device_ids)
        net.to(device)
        
        loss_fn = get_loss(config, device, reduction="mean")
        trainable_params = get_net_trainable_params(net)
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()
        scheduler = build_scheduler(config, optimizer)
        # archaicEarlyStopper = ArchaicEarlyStopper(patience_eval=10, patience_dL=10)
        
        BEST_LOSS = 1e6 # initial loss very high
        logFreqTrain, logFreqEval, eval_freq = get_loggingFreq(num_steps_train, num_steps_eval, batch_size,num_epochs)
        net.train()
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            # save model ----------------------------------------------------------------------------------------------#
            if epoch % save_epochs == 0: # save the very first time to avoid early kill  
                if len(local_device_ids) > 1:
                    torch.save(net.module.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, 1))
                else:
                    torch.save(net.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, 1))
            # train model ------------------------------------------------------------------------------------------#
            loss_all, train_metrics_all = [], []
            for step, (sample, int_target_time_specimen) in enumerate(dataloaders['train']):
                # overall batch number pass
                abs_step =  epoch * num_steps_train + step
                # forward pass batch
                output, loss = train_step(net, sample, loss_fn, optimizer, device)
                loss_all.append(loss) 
                # print batch statistics ----------------------------------------------------------------------------------#
                # if (abs_step > 0) & (abs_step % train_metrics_steps == 0):
                if (step > 0) and (step % logFreqTrain) == 0:
                    denormalizer = Denormalize(config["DATASETS"]["train"]) 
                    output = output.detach().cpu()
                    
                    write_loss_summaries(writer, loss, loss_all, abs_step, mean_windowSize=logFreqTrain, mode="train")
                    
                    for i, target_var_name in enumerate(target_vars):
                        output[:,i,...] = denormalizer((output[:,i,...],target_var_name))
                        sample["targets"][:,i,...] = denormalizer((sample["targets"][:,i,...], target_var_name))

                        batch_metrics = get_mean_reg_metrics(output=output[:,i,...], 
                                                            target=sample["targets"][:,i,...],
                                                            var_name=target_var_name)
                        train_metrics_all.append(batch_metrics)
                        # write_mean_summaries(writer, batch_metrics, train_metrics_all, abs_step, mean_windowSize=logFreqTrain, mode="train", optimizer=optimizer)
                        if config["DATASETS"]["train"]["plot_img"]:
                            write_images_summaries(writer, output[:,i,...],sample["targets"][:,i,...],sample["year"], sample["month"],sample["day"], sample["hour"], 
                                                        target_var_name, abs_step, mode="train")
                        if config["DATASETS"]["train"]["plot_scatter"]:
                            write_scatter_summaries(writer, output[:,i,...],
                                                    sample["targets"][:,i,...],sample["doy"], sample["year"],
                                                    target_var_name, abs_step, mode="train")
                        if config["DATASETS"]["train"]["plot_qq"]:
                            write_qq_summaries(writer, output[:,i,...],
                                                    sample["targets"][:,i,...],sample["doy"], sample["year"],
                                                    target_var_name, abs_step, mode="train")

                    # update optimizer and earlyStopper
                    mean_train_loss = np.array(torch.stack(loss_all).detach().cpu()).mean()
                    scheduler.step(mean_train_loss)
                    print("Training --> abs_step: %d, epoch: %d, step: %5d, loss: %.4f, meanEpochLoss: %.4f, lr: %.7f" %(abs_step, epoch, step + 1, loss, mean_train_loss, lr))
                
                
                # evaluate model ------------------------------------------------------------------------------------------#
                if (step > 0) and (step % eval_freq) == 0: 
                    net.eval()
                    loss_all, eval_metrics_all = [], []
                    with torch.no_grad():
                        for step, (sample, int_target_time_specimen) in enumerate(dataloaders['eval']):
                            output = net(sample["inputs"].to(device))
                            loss = loss_fn(output, sample["targets"].to(device))
                            loss_all.append(loss)

                            if step % (logFreqEval) == 0:
                                denormalizer = Denormalize(config["DATASETS"]["eval"]) 
                                output = output.detach().cpu() 
                                write_loss_summaries(writer, loss, loss_all, abs_step+step, mean_windowSize=logFreqEval, mode="eval")
                                
                                for i, target_var_name in enumerate(target_vars):
                                    output[:,i,...] = denormalizer((output[:,i,...],target_var_name))
                                    sample["targets"][:,i,...] = denormalizer((sample["targets"][:,i,...],target_var_name))

                                    batch_metrics = get_mean_reg_metrics(output=output[:,i,...], 
                                                                        target=sample["targets"][:,i,...],
                                                                        var_name=target_var_name)
                                    eval_metrics_all.append(batch_metrics)
                                    # write_mean_summaries(writer, batch_metrics,eval_metrics_all, abs_step+step, mean_windowSize=logFreqEval, mode="eval", optimizer=optimizer)
                                    if config["DATASETS"]["eval"]["plot_img"]:
                                        write_images_summaries(writer, output[:,i,...],sample["targets"][:,i,...],sample["year"], sample["month"],sample["day"], sample["hour"], 
                                                            target_var_name, abs_step+step, mode="eval")
                                    if config["DATASETS"]["eval"]["plot_scatter"]:
                                        write_scatter_summaries(writer, output[:,i,...],sample["targets"][:,i,...],sample["doy"], sample["year"],target_var_name, abs_step+step, mode="eval")
                                    if config["DATASETS"]["eval"]["plot_qq"]:
                                        write_qq_summaries(writer, output[:,i,...],sample["targets"][:,i,...],sample["doy"], sample["year"], target_var_name, abs_step+step, mode="eval")
                                print("Evaluation --> abs_step: %d, epoch: %d, step: %5d, loss: %.7f" %(step, epoch, step + 1, loss))
                    
                    mean_eval_loss = np.array(torch.stack(loss_all).detach().cpu()).mean()
                    # checkpointing
                    if mean_eval_loss < BEST_LOSS:
                        print(f"Saving best.pth --> mean_eval_loss = {mean_eval_loss}")
                        if len(local_device_ids) > 1:
                            torch.save(net.module.state_dict(), "%s/best.pth" % (save_path))
                        else:
                            torch.save(net.state_dict(), "%s/best.pth" % (save_path))
                        BEST_LOSS = mean_eval_loss
                        print(f"New best loss: {BEST_LOSS}")
                    net.train()
                    # # update earlyStopping
                    # doStop, evalStop, evalList, deltaStop, deltaList = archaicEarlyStopper.step(mean_eval_loss,mean_train_loss)
                    # if doStop and earlyStopping:
                    #     print("Early stopping activated.")
                    #     if evalStop:
                    #         print(f"evalLoss did not improve for {len(evalList)} times: {evalList}")
                    #     if deltaStop:
                    #         print(f"deltaLoss did not improve for {len(deltaList)} times: {deltaList}")
                        
                    #     return 
                        
    #------------------------------------------------------------------------------------------------------------------#
    run(net, writer, config)
 
if __name__ == "__main__":
    signal.signal(signal.SIGINT, handler)
    print(f"argv: {sys.argv}")
    
    parser = create_parser()
    args = parser.parse_args()
    config = update_config(args)
    print_header(config)
    
    # make save_path dir if not existing
    save_path = config['CHECKPOINT']["save_path"]
    if  save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)
        print(f"Save path created at: {save_path}")
    else:
        print(f"Save path already exists at: {save_path}")
    # copy yaml to exp directory only if training. If Infering, it already exists and is beeing used.
    if config["MODEL"]["train"]:
        copy_yaml(config)
    
    if not config['wandb']:
        print("Deactivated wandb.")
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_MODE"] = "disabled"
    with open("configs/Helio/wandb_api.key") as key:
        os.environ["WANDB_API_KEY"] = key.read()
        wandb.login()
    wandb.tensorboard.patch(root_logdir=save_path) 
    with wandb.init(project=config["MODEL"]["project_name"], name=config["MODEL"]["exp_name"], config=config, sync_tensorboard=True) as wb_run:
        writer = SummaryWriter(save_path)
        device = get_device(config['local_device_ids'], allow_cpu=False)
        dataloaders = get_dataloaders(config)
        net = get_model(config, device)     
        if config["MODEL"]["train"]:
            train_and_evaluate(net=net, writer=writer, dataloaders=dataloaders, config=config, device=device)
        if config["MODEL"]["inference"]:
            infer(writer=writer, dataloaders=dataloaders, config=config, device=device)
