from __future__ import print_function, division
import os
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data
import pickle
import xarray as xr
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def get_distr_dataloader(paths_file, root_dir, rank, world_size, transform=None, batch_size=32, num_workers=4,
                         shuffle=True, return_paths=False):
    """
    return a distributed dataloader
    """
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler)
    return dataloader

def get_dataloader(start_date, end_date, input_var_names,target_var_names, root_dir, img_res, max_seq_len, 
                   dset_patching, transform=None, reg_mode=True, batch_size=32, num_workers=4, shuffle=True, return_paths=False, infer=False, my_collate=None):
    if not dset_patching: 
        dataset = SatImDataset(start_date, end_date, input_var_names, target_var_names, root_dir=root_dir, max_seq_len=max_seq_len, transform=transform, reg_mode=reg_mode, return_paths=return_paths, infer=infer)
    else:
        dataset = SatImDataset_patched(start_date, end_date, input_var_names, target_var_names, root_dir=root_dir, patch_size=img_res, max_seq_len=max_seq_len, transform=transform, reg_mode=reg_mode, return_paths=return_paths, infer=infer)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class SatImDataset(Dataset):
    """Helio dataset yielding raw size images. Used for inference."""

    def __init__(self,start_date, end_date, input_var_names, target_var_names, root_dir, max_seq_len=12, transform=None, reg_mode=True, multilabel=False, return_paths=False, infer=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if type(csv_file) == str:
        #     self.data_paths = pd.read_csv(csv_file, header=None)
        # elif type(csv_file) in [list, tuple]:
        #     self.data_paths = pd.concat([pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file], axis=0).reset_index(drop=True)
        self.root_dir = root_dir

        self.start_date = pd.to_datetime(start_date, format="%Y-%m-%dT%H")
        self.end_date = pd.to_datetime(end_date, format="%Y-%m-%dT%H")
        self.max_seq_len = max_seq_len

        self.reg_mode = reg_mode
        self.infer = infer
        self.input_var_names = input_var_names # list of var names
        self.target_var_names = target_var_names # list of var names
        self.ds_input= None # TODO: might not want to keep all in memory

        self.ds_target = None
     
        self.time_specimen = None

        self.nc_canvas = None # copy of one of the ncfiles skeleton (without data) for inference

        self.transform = transform
        # self.multilabel = multilabel
        # self.return_paths = return_paths

        self.loadDs()
        
    def __len__(self):
        return len(self.time_specimen)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_start = idx - (self.max_seq_len - 1)
        if idx_start < 0:
            idx = (self.max_seq_len - 1) # can't get idx lower than max_seq_len
            idx_start = 0 
        
        idx_range = np.arange(idx_start,idx + 1)
        
        sample = {
            "timestamp":self.time_specimen[idx_range],
            "inputs":self.ds_input[:,idx_range,...], #np.newaxis,
            }
        if not self.infer:
            sample["targets"] = self.ds_target[:,idx,...] #_range not looking at all targets but only the idx of interest
        else: # hack: if infering just feeding an input var so that transforms() go through 
            sample["targets"] = self.ds_input[0,idx,...] 

        if not self.reg_mode:
            sample["targets"] = sample["targets"].astype(int)

        if self.transform:
            sample = self.transform(sample) # toTensor, 


        # if self.return_paths:
        #     return sample, img_name
         
        int_target_time_specimen = int(self.time_specimen[idx_range].strftime("%Y%m%d%H%M%S")[-1]) # useful for tracking actual time

        return sample, int_target_time_specimen
    
    def loadDs(self):
        "Reads nc ds, stores them"
        ds_input_var_cont = []
        ds_target_var_cont = []

        for key in self.input_var_names:
            print(f"loading feature: {key}")
            # ncfile = glob.glob(join(self.dsRoot, f"**/*{key}.nc"), recursive=True)[0]
            ncfile = str(next(Path(self.root_dir).rglob(f"*{key}.nc")))
            # select timestep
            ds_var = xr.open_dataset(ncfile).sel(time=slice(self.start_date, self.end_date))
            # select domain 
            ds_input_var_cont.append(ds_var[key].values)
        if not self.infer:
            for key in self.target_var_names:
                print(f"loading feature: {key}")
                # ncfile = glob.glob(join(self.dsRoot, f"**/*{key}.nc"), recursive=True)[0]
                ncfile = str(next(Path(self.root_dir).rglob(f"*{key}.nc")))
                # select timestep
                ds_var = xr.open_dataset(ncfile).sel(time=slice(self.start_date, self.end_date))
                # select domain 
                ds_target_var_cont.append(ds_var[key].values)
        
        print(f"Getting time specimen from {key} dataset") # is ultimately the specimen from last key of target_names 
        self.time_specimen = pd.to_datetime(ds_var.time.values)
        self.nc_canvas = ds_var.copy() # shallow copy

        self.ds_input = np.array(ds_input_var_cont) # CTHW
        del ds_input_var_cont
        if not self.infer:
            self.ds_target = np.array(ds_target_var_cont) # CTHW
            del ds_target_var_cont
        

    def read(self, idx, abs=False):
        """
        read single dataset sample corresponding to idx (index number) without any data transform applied
        """
        if type(idx) == int:
            img_name = os.path.join(self.root_dir,
                                    self.data_paths.iloc[idx, 0])
        if type(idx) == str:
            if abs:
                img_name = idx
            else:
                img_name = os.path.join(self.root_dir, idx)
        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')
        return sample
    
class SatImDataset_patched(Dataset):
    """Helio dataset split in subset for training."""

    def __init__(self,start_date, end_date, input_var_names, target_var_names, root_dir, patch_size,max_seq_len=12, transform=None, reg_mode=True, multilabel=False, return_paths=False, infer=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            patch_size: dimension of cropped image (not to be confounded with VITs patch_size!)
        """
        # if type(csv_file) == str:
        #     self.data_paths = pd.read_csv(csv_file, header=None)
        # elif type(csv_file) in [list, tuple]:
        #     self.data_paths = pd.concat([pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file], axis=0).reset_index(drop=True)
        self.root_dir = root_dir

        self.start_date = pd.to_datetime(start_date, format="%Y-%m-%dT%H")
        self.end_date = pd.to_datetime(end_date, format="%Y-%m-%dT%H")
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        # self.num_patches = None #resulting number of patches (depends of patch_size and original image size)

        self.reg_mode = reg_mode
        self.infer = infer
        self.input_var_names = input_var_names # list of var names
        self.target_var_names = target_var_names # list of var names
        self.ds_input= None # TODO: might not want to keep all in memory
        self.ds_target = None
        self.time_specimen = None

        
        self.transform = transform
        # self.multilabel = multilabel
        # self.return_paths = return_paths

        self.loadDs()
        
    def __len__(self):
        return self.ds_input.shape[-3]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # # the stride between each time step is the number of patches per timestep
        # stride_t = len(self.time_specimen)
        # get divider and remainder
        patch_num, time_idx = divmod(idx, len(self.time_specimen))
        
        # determine start index
        idx_start = patch_num * len(self.time_specimen) + (time_idx - (self.max_seq_len - 1))
        # treat boundaries for sequences
        if (time_idx - (self.max_seq_len - 1)) < 0: #idx_start
            idx = patch_num * len(self.time_specimen) + (self.max_seq_len - 1) # can't get idx lower than max_seq_len
            idx_start = patch_num * len(self.time_specimen) 
        # get sequence range
        idx_range = np.arange(idx_start,idx + 1)
        time_idx_range = np.arange(time_idx - (self.max_seq_len - 1), time_idx + 1)

        sample = {
            "timestamp":self.time_specimen[time_idx_range],
            "inputs":self.ds_input[:,idx_range,...], #np.newaxis,
            }
        if not self.infer:
            sample["targets"] = self.ds_target[:,idx,...] #_range not looking at all targets but only the idx of interest
        else: # hack: if infering just feeding an input var so that transforms() go through 
            sample["targets"] = self.ds_input[0,idx,...] 

        if not self.reg_mode:
            sample["targets"] = sample["targets"].astype(int)
        if self.transform:
            sample = self.transform(sample) # toTensor, 


        # if self.return_paths:
        #     return sample, img_name
        int_target_time_specimen = int(self.time_specimen[time_idx_range].strftime("%Y%m%d%H%M%S")[-1]) # useful for tracking actual time

        return sample, int_target_time_specimen
    
    def loadDs(self):
        "Reads nc ds, stores them"
        ds_input_var_cont = []
        ds_target_var_cont = []

        for key in self.input_var_names:
            print(f"loading Heliomont: {key}")
            # ncfile = glob.glob(join(self.dsRoot, f"**/*{key}.nc"), recursive=True)[0]
            ncfile = str(next(Path(self.root_dir).rglob(f"*{key}.nc")))
            # select timestep
            ds_var = xr.open_dataset(ncfile).sel(time=slice(self.start_date, self.end_date))
            # select domain 
            ds_input_var_cont.append(ds_var[key].values.astype("float32"))

        if not self.infer:
            for key in self.target_var_names:
                print(f"loading Heliomont: {key}")
                # ncfile = glob.glob(join(self.dsRoot, f"**/*{key}.nc"), recursive=True)[0]
                ncfile = str(next(Path(self.root_dir).rglob(f"*{key}.nc")))
                # select timestep
                ds_var = xr.open_dataset(ncfile).sel(time=slice(self.start_date, self.end_date))
                # select domain 
                ds_target_var_cont.append(ds_var[key].values.astype("float32"))
        
        print(f"getting time specimen from {key} dataset") # is ultimately the specimen from last key of target_names 
        self.time_specimen = pd.to_datetime(ds_var.time.values)


        self.ds_input = np.array(ds_input_var_cont) # CTHW
        del ds_input_var_cont

        if not self.infer:
            self.ds_target = np.array(ds_target_var_cont) # CTHW
            del ds_target_var_cont
        # patching/cropping only here for training and eval
        self.ds_input = self.patch_ds(self.ds_input) 
        if not self.infer:
            self.ds_target = self.patch_ds(self.ds_target)
        print(f"Patched dataset length: {self.ds_input.shape[-3]}")

    def read(self, idx, abs=False):
        """
        read single dataset sample corresponding to idx (index number) without any data transform applied
        """
        if type(idx) == int:
            img_name = os.path.join(self.root_dir,
                                    self.data_paths.iloc[idx, 0])
        if type(idx) == str:
            if abs:
                img_name = idx
            else:
                img_name = os.path.join(self.root_dir, idx)
        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')
        return sample

    def patch_ds(self, ds):
        """patches ds without overlap except for the remainder which is patched overlapping the second to last row or column
        format must be BCTHW. Stacks patches on time dimension. Keeps time coherence within each patch (stride=len(time_specimen))."""
        split_ds = []
        # num_patches = 0
        # if self.patching_method == "block":
        x_fullpatch = int(ds.shape[-1] / self.patch_size)
        y_fullpatch = int(ds.shape[-2] / self.patch_size)
        for x in range(x_fullpatch):
            x_b = x * self.patch_size
            x_t = (x + 1) * self.patch_size
            for y in range(y_fullpatch):
                y_b = y * self.patch_size
                y_t =( y + 1) * self.patch_size
                split_ds.append(ds[...,y_b:y_t,x_b:x_t])
                # num_patches += 1
            if ds.shape[-2] > y_t: # take care of remaining parts in y
                remainder_y = ds.shape[-2] - y_t
                split_ds.append(ds[...,y_b+remainder_y:,x_b:x_t,])
                # num_patches += 1
        if ds.shape[-1] > x_t: # take care of remaining parts in x
            remainder_x = ds.shape[-1] - x_t
            for y in range(y_fullpatch):
                y_b = y * self.patch_size
                y_t =( y + 1) * self.patch_size
                split_ds.append(ds[...,y_b:y_t, x_b + remainder_x:])
                # num_patches += 1
            if ds.shape[-2] > y_t: 
                remainder_y = ds.shape[-2] - y_t
                split_ds.append(ds[..., y_b+remainder_y:, x_b + remainder_x:])
                # num_patches += 1
        
        # self.num_patches = num_patches

            
        # elif self.patching_method == "window":
        #     raise("this methods explodes the memory for now")
        #     # window splitting with step=1
        #     assert self.patch_size % 2 == 1 #here patch size must be odd
        #     # padding first to access borders
        #     ds = np.pad(ds, pad_width=[(0,0),(0,0),
        #                                (self.patch_size//2,self.patch_size//2),
        #                                (self.patch_size//2,self.patch_size//2)])
        #     for x in range(ds.shape[-1]-self.patch_size):
        #         for y in range(ds.shape[-2]-self.patch_size):
        #             split_ds.append(ds[..., y:y+self.patch_size, x:x+self.patch_size])

        # else:
        #     NotImplementedError("patching method not implemented")
        # stack on time dim
        time_stacked_ds = np.concat(split_ds, axis=1)
        
        return time_stacked_ds
        # rolled_input = view_as_windows(self.ds_input)
        # np.array_split(self.input)

