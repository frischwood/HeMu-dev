from __future__ import print_function, division
import numpy as np
import torch
from torchvision import transforms
from copy import deepcopy
from pandas import read_csv



def Helio_reg_transform(model_config, dataset_config, is_training):
    """
    """
    if is_training:
        ds_cfg = dataset_config["train"]
    else:
        ds_cfg = dataset_config["infer"] 
        
    dataset_img_res =   ["img_res"]           # TODO auto read from dataset
    input_img_res = model_config['img_res']
    max_seq_len = model_config['max_seq_len']


    transform_list = []
 
    transform_list.append(ToTensor())         # data from numpy arrays to torch.float32
    transform_list.append(Normalize(ds_cfg))  # normalize all inputs individually
    
   
    if not is_training:  #
        if ds_cfg["perm_test"]:
            var_idx = np.argwhere(np.array(dataset_config["train"]["input_var_names"])==dataset_config["infer"]["perm_test"])
            transform_list.append(ShuffleVar(var_idx))
        
    if is_training and not(dataset_config["train"]["dset_patching"]) and (dataset_img_res != input_img_res):
        transform_list.append(
            Crop(img_size=dataset_img_res, crop_size=input_img_res, random=is_training))  # crop

    if "TSViT" in model_config["architecture"]:
        transform_list.append(TileDates(use_hod=model_config["use_hod"], doy_bins=None))                # tile month, day, hour, minute,
        transform_list.append(CutOrPad(max_seq_len=max_seq_len, random_sample=False, from_start=True))  # pad with zeros to maximum sequence length
        transform_list.append(ToTHWC())

    elif "ConvResNet" in model_config["architecture"]:
        transform_list.append(TileDates(use_hod=model_config["use_hod"], doy_bins=None))    
        transform_list.append(ToCHW())
        if not model_config["inference"]:
            transform_list.append(TargetIsCenterPix())

        
    return transforms.Compose(transform_list)


class ToTHWC(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __call__(self, sample):
        sample['inputs'] = sample['inputs'].permute(1, 2, 3, 0)
        # sample['targets'] = sample['targets'] # removing time dim as only single one # permute(0, 2, 3, 1)
        return sample

class ToTCHW(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __call__(self, sample):
        sample['inputs'] = sample['inputs'].permute(1, 0, 2, 3)
        # sample['targets'] = sample['targets'] # removing time dim as only single one # permute(0, 2, 3, 1)
        return sample

class ToCHW(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __call__(self, sample):
        sample['inputs'] = sample['inputs'].squeeze(1)
        # sample['targets'] = sample['targets'] # removing time dim as only single one # permute(0, 2, 3, 1)
        return sample

class TargetIsCenterPix(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """      
    def __call__(self, sample):
        sample['targets'] = sample['targets'][...,sample['targets'].shape[-2]//2,sample['targets'].shape[-1]//2]
        # sample['targets'] = sample['targets'] # removing time dim as only single one # permute(0, 2, 3, 1)
        return sample

class ShuffleVar(object):
    """S"""
    def __init__(self, var_idx):
        self.var_idx = var_idx

    def __call__(self, sample):
        # sample: C,T,H,W
        rand_y = torch.randperm(sample["inputs"].shape[-2])
        rand_x = torch.randperm(sample["inputs"].shape[-1])
        # shuffle y and x sequentially
        sample["inputs"][self.var_idx,...] = sample["inputs"][self.var_idx.item(),:,rand_y,:]
        sample["inputs"][self.var_idx,...] = sample["inputs"][self.var_idx.item(),:,:,rand_x]
        return sample

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __init__(self, label_type='groups', ground_truths=[]):
        self.label_type = label_type
        self.ground_truths = ground_truths

    def __call__(self, sample):
        tensor_sample = {}
        tensor_sample['inputs'] = torch.tensor(sample['inputs']).to(torch.float32).nan_to_num(nan=0.0)
        tensor_sample['targets'] = torch.tensor(sample['targets']).to(torch.float32).nan_to_num(nan=0.0) # replaces potential nans by 0
   
        tensor_sample['doy'] = torch.tensor(np.array(sample['timestamp'].dayofyear)).to(torch.float32).nan_to_num(nan=0.0)
        tensor_sample['hoy'] = torch.tensor(np.array((sample['timestamp'].dayofyear-1)*24)+sample['timestamp'].hour).to(torch.float32).nan_to_num(nan=0.0)
        tensor_sample['year'] = torch.tensor(np.array(sample['timestamp'].year)).to(torch.float32).nan_to_num(nan=0.0)
        tensor_sample['month'] = torch.tensor(np.array(sample['timestamp'].month)).to(torch.float32).nan_to_num(nan=0.0)
        tensor_sample['day'] = torch.tensor(np.array(sample['timestamp'].day)).to(torch.float32).nan_to_num(nan=0.0)
        tensor_sample['hour'] = torch.tensor(np.array(sample['timestamp'].hour)).to(torch.float32).nan_to_num(nan=0.0)
        tensor_sample['minute'] = torch.tensor(np.array(sample['timestamp'].minute)).to(torch.float32).nan_to_num(nan=0.0)
        

        return tensor_sample

    def replaceReportNans(self, tensor_sample):
        for key in tensor_sample.keys():
            nanSum = tensor_sample[key].isnan().sum()
            if  nanSum > 0:
                print(f"{key} nan ratio (initial): {nanSum/len(tensor_sample[key].view(-1))}")
                # replace nans
                tensor_sample[key] = tensor_sample[key].nan_to_num(nan=0.0)
                print(f"{key} nan ratio (final): {tensor_sample[key].isnan().sum()/ len(tensor_sample[key].view(-1))}")
        
        return tensor_sample       

class Normalize(object):
    """
    Normalize inputs as in https://arxiv.org/pdf/1802.02080.pdf
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __init__(self, config_dataset):
        self.mean_input = {}
        self.std_input = {}
        self.mean_target = {}
        self.std_target = {}
        df_stats = read_csv(config_dataset["path_stats_csv"]).set_index("Unnamed: 0")
        for var in config_dataset["input_var_names"]:
            self.mean_input[var] = df_stats.loc["mean", var]
            self.std_input[var] = df_stats.loc["std", var] 
        for var in config_dataset["target_var_names"]:
            self.mean_target[var] = df_stats.loc["mean", var]
            self.std_target[var] = df_stats.loc["std", var]

    def __call__(self, sample):
        # print('mean: ', sample['img'].mean(dim=(0,2,3)))
        # print('std : ', sample['img'].std(dim=(0,2,3)))

        sample['inputs'] = (sample['inputs'] - np.array(list(self.mean_input.values()))[:,np.newaxis,np.newaxis,np.newaxis].astype("float32")) / np.array(list(self.std_input.values()))[:,np.newaxis,np.newaxis,np.newaxis].astype("float32")

        sample['targets'] = (sample['targets'] - np.array(list(self.mean_target.values()))[:,np.newaxis,np.newaxis].astype("float32")) / np.array(list(self.std_target.values()))[:,np.newaxis,np.newaxis].astype("float32")

        sample['doy'] = sample['doy'] / 366
        assert (sample['doy'] > 0).all() & (sample['doy'] <= 1).all()
        # year = sample['year']
        # sample['month'] = 2*((sample['month']-1)/11)-1
        # sample['day'] = 2*((sample['day']-1)/30)-1
        sample["hoy"] = (sample['hoy'] / (366*24)) # is not cyclic but works for our application
        sample['hour'] = sample['hour'] / 23 # 2*((sample['hour']-1)/23)-1
        assert (sample['hour'] >= 0).all() & (sample['hour'] <= 1).all()
        # sample['minute'] = 2*((sample['minute']-1)/59)-1

        return sample

class Denormalize(object):
    """
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __init__(self, config_dataset):

        self.df_stats = read_csv(config_dataset["path_stats_csv"]).set_index("Unnamed: 0")
        # for var in config_dataset["target_var_names"]: # the order of this list sets order for targets and outputs tensors on the channel dim
        #     self.mean_target[var] = df_stats.loc["mean", var]
        #     self.std_target[var] = df_stats.loc["std", var]

    def __call__(self, tensor_var_tuple):
        tensor, var = tensor_var_tuple
        tensor = (tensor * np.array(self.df_stats.loc["std", var]).astype("float32")) + np.array(self.df_stats.loc["mean", var]).astype("float32")

        return tensor

class Crop(object):
    """Crop randomly the image in a sample. Adapted for non-square dataset

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, img_size, crop_size, random=False):
        self.img_size = img_size
        self.crop_size = crop_size
        self.random = random
        if not random:
            self.top = int((img_size[1] - crop_size) / 2)
            self.left = int((img_size[0] - crop_size) / 2)


    def __call__(self, sample):
        if self.random:
            top = torch.randint(self.img_size[1] - self.crop_size, (1,))[0]
            left = torch.randint(self.img_size[0] - self.crop_size, (1,))[0]
        else:  # center
            top = self.top
            left = self.left
        sample['inputs'] = sample['inputs'][:, :, top:top + self.crop_size, left:left + self.crop_size]
        sample['targets'] = sample['targets'][:, top:top + self.crop_size, left:left + self.crop_size]

        return sample

class TileDates(object):
    """
    Tile a 1d array to height (H), width (W) of an image.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, use_hod=False, doy_bins=None):
        self.doy_bins = doy_bins
        self.use_hod = use_hod

    def __call__(self, sample):
        doy = self.repeat(sample['doy'],H=sample["inputs"].shape[-2], W=sample["inputs"].shape[-1],binned=self.doy_bins is not None)
        year = self.repeat(sample['year'], H=sample["inputs"].shape[-2], W=sample["inputs"].shape[-1],binned=self.doy_bins is not None)
        month = self.repeat(sample['month'], H=sample["inputs"].shape[-2], W=sample["inputs"].shape[-1],binned=self.doy_bins is not None)
        day = self.repeat(sample['day'], H=sample["inputs"].shape[-2], W=sample["inputs"].shape[-1],binned=self.doy_bins is not None)
        hour = self.repeat(sample['hour'], H=sample["inputs"].shape[-2], W=sample["inputs"].shape[-1],binned=self.doy_bins is not None)
        hoy = self.repeat(sample['hoy'],H=sample["inputs"].shape[-2], W=sample["inputs"].shape[-1],binned=self.doy_bins is not None)
        minute = self.repeat(sample['minute'], H=sample["inputs"].shape[-2], W=sample["inputs"].shape[-1],binned=self.doy_bins is not None)
        
        sample['inputs'] = torch.cat(([sample['inputs'], hoy]), dim=0) #  doy month,day,hour,minute
        if self.use_hod:
            sample['inputs'] = torch.cat(([sample['inputs'], hour]), dim=0) # order matters!! hod must be last dim

        # del sample['doy']
        return sample
    
    def repeat(self, tensor, H, W, binned=False):
        if binned:
            out = tensor.unsqueeze(1).unsqueeze(1).repeat(1,1, H, W)#.permute(0, 2, 3, 1)
        else:
            doy_r_cont = []
            for doy in tensor: doy_r_cont.append(doy.repeat(1,1,H,W))
            
            # out = tensor.repeat(1,1, H, W)#.permute(3, 0, 1, 2)
        return torch.cat(doy_r_cont,dim=1)
     
class CutOrPad(object):
    """
    Pad series with zeros (matching series elements) to a max sequence length or cut sequential parts
    items in  : inputs, *inputs_backward, labels
    items out : inputs, *inputs_backward, labels, seq_lengths

    REMOVE DEEPCOPY OR REPLACE WITH TORCH FUN
    """

    def __init__(self, max_seq_len, random_sample=False, from_start=False):
        assert isinstance(max_seq_len, (int, tuple))
        self.max_seq_len = max_seq_len
        self.random_sample = random_sample
        self.from_start = from_start
        assert int(random_sample) * int(from_start) == 0, "choose either one of random, from start sequence cut methods but not both"

    def __call__(self, sample):
        seq_len = deepcopy(sample['inputs'].shape[1])
        sample['inputs'] = self.pad_or_cut(sample['inputs'])
        if "inputs_backward" in sample:
            sample['inputs_backward'] = self.pad_or_cut(sample['inputs_backward'])
        if seq_len > self.max_seq_len:
            seq_len = self.max_seq_len
        sample['seq_lengths'] = seq_len
        return sample

    def pad_or_cut(self, tensor, dtype=torch.float32):
        tensor = tensor.permute(1,0,2,3) # to T,C,H,W
        seq_len = tensor.shape[0]
        diff = self.max_seq_len - seq_len
        if diff > 0:
            tsize = list(tensor.shape)
            if len(tsize) == 1:
                pad_shape = [diff]
            else:
                pad_shape = [diff] + tsize[1:]
            tensor = torch.cat((tensor, torch.zeros(pad_shape, dtype=dtype)), dim=0)
        elif diff < 0:
            if self.random_sample:
                return tensor[self.random_subseq(seq_len)]
            elif self.from_start:
                start_idx = 0
            else:
                start_idx = torch.randint(seq_len - self.max_seq_len, (1,))[0]
            tensor = tensor[start_idx:start_idx+self.max_seq_len]
        tensor = tensor.permute(1,0,2,3) # back to C,T,H,W
        return tensor
    
    def random_subseq(self, seq_len):
        return torch.randperm(seq_len)[:self.max_seq_len].sort()[0]

