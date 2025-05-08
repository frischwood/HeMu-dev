import torch
import numpy as np
import torch.nn as nn

    
def get_mean_reg_metrics(output, target, epoch=0, step=0, unk_masks=None, var_name=""):
    """
    :param logits: (N, D, H, W)
    """
    mse = nn.functional.mse_loss(output, target, reduction="mean").detach().cpu().numpy()
    mae = nn.functional.l1_loss(output,target,reduction="mean").detach().cpu().numpy()
    mbe = (target-output).mean().detach().cpu().numpy()
    # np.mean(torch.mean((target - output), dim=(0,-2,-1)).detach().cpu().numpy())


    return {"%s/MSE" % var_name: mse, "%s/MAE" % var_name: mae, "%s/MBE" % var_name: mbe}
