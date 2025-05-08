import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import io
import os
import torchvision as tv
import torch
import numpy as np
from scipy.stats import gaussian_kde, linregress
import tensorboard as tf
import numpy as np
import wandb
# from torchvision import transforms
# from data.Helio.data_transforms import Denormalize

def write_mean_summaries(writer, metrics, metrics_all, abs_step, mean_windowSize=10, mode="train", optimizer=None):
    for key in metrics:
        writer.add_scalars(main_tag=key, tag_scalar_dict={'%s_1b_Average' % mode: metrics[key]},
                           global_step=abs_step, walltime=None)
        key_cont = []
        for metrics_t in metrics_all:
            if key not in metrics_t.keys():
                pass
            else:
                key_cont.append(metrics_t[key])
        mean_window_metric = np.mean(key_cont) if len(key_cont)<=mean_windowSize else np.mean(key_cont[-mean_windowSize:])
        writer.add_scalars(main_tag=key, tag_scalar_dict={f'{mode}_{mean_windowSize}b_Average':mean_window_metric},
                           global_step=abs_step, walltime=None)
        
    if optimizer is not None:
        writer.add_scalar('learn_rate', optimizer.param_groups[0]["lr"], abs_step)


def write_class_summaries(writer, metrics, abs_step, mode="eval", optimizer=None):
    unique_labels, metrics = metrics
    print("saving per class summaries")
    for key in metrics:
        tag_scalar_dict = {'%s_%s' % (mode, str(i)): val for i, val in zip(unique_labels, metrics[key])}
        writer.add_scalars(main_tag=key, tag_scalar_dict=tag_scalar_dict, global_step=abs_step, walltime=None)
    if optimizer is not None:
        writer.add_scalar('learn_rate', optimizer.param_groups[0]["lr"], abs_step)

def write_histogram_summaries(writer, metrics, abs_step, mode="train"):
    for key in metrics:
        writer.add_histogram("%s_%s" % (mode, key), metrics[key], global_step=abs_step)

def write_images_summaries(writer, output_img, target_img,  var, abs_step, mode="eval"):
    output_img_maxNorm = output_img / output_img.max()
    target_img_maxNorm = target_img / target_img.max()
    writer.add_images(tag=f"{mode} Output: {var}", img_tensor=output_img_maxNorm.unsqueeze(1), global_step=abs_step, dataformats="NCHW")
    writer.add_images(tag=f"{mode} Target: {var}", img_tensor=target_img_maxNorm.unsqueeze(1), global_step=abs_step, dataformats="NCHW")

    
def write_scatter_summaries(writer, output_img, target_img, doy, year, var, abs_step, mode="eval"):
    def density_scatter( x , y, fig = None, ax = None, sort = True, bins = 20, **kwargs )   :
        """
        Scatter plot colored by 2d histogram
        """
        if ax is None :
            fig , ax = plt.subplots()
        data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
        z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

        #To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort :
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        ax.scatter( x, y, c=z, **kwargs )

        norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        cbar.ax.set_ylabel(f'Density (Total={x.shape[0]})')

        return fig, ax
    # fig, ax = plt.subplots()
    x=output_img.reshape([-1]).numpy()
    y=target_img.reshape([-1]).numpy()
    # Calculate the point density
    # xy = np.vstack([x,y])
    # z = gaussian_kde(xy)(xy)
    # compute regression
    m, b, r_value, p_value, std_err = linregress(x, y)
    # ax.scatter(x=x, y=y, c=z, s=50, marker="x")
    fig, ax = density_scatter(x,y)
    ax.plot([0,max(x.max(),y.max())],[0,max(x.max(),y.max())], "k-", lw=1) # 1-1line
    ax.plot(x, m*x+b, color='red')
    ax.set_xlabel("Target")
    ax.set_ylabel("Output")
    # write img
    writer.add_image(tag=f"{mode} scatter: {var}", img_tensor=plotToImage(fig),global_step=abs_step,dataformats="CHW")
    writer.add_scalar(f"{mode} scatter: R^2({var})", r_value**2, abs_step)
    plt.close()

def write_qq_summaries(writer, output_img, target_img, doy, year, var, abs_step, mode="eval"):
    quantiles = np.arange(0,1.01,0.01)
    q_output = np.quantile(output_img.reshape([-1]).numpy(), q=quantiles)
    q_target = np.quantile(target_img.reshape([-1]).numpy(), q=quantiles) 
    fig, ax = plt.subplots()
    ax.scatter(x=q_target, y=q_output, c="grey", label="q-q", linewidth=1.0)
    ax.plot([0,max(q_target.max(),q_output.max())],[0,max(q_target.max(),q_output.max())], "k-", lw=1) # 1-1line
    ax.legend(loc="upper left", prop={"size": 12})
    # ax.set_facecolor("#f0f0f0")
    ax.set_xlabel("q-Target")
    ax.set_ylabel("q-Output")
    
    # write img
    writer.add_image(tag=f"{mode} qq: {var}", img_tensor=plotToImage(fig),global_step=abs_step,dataformats="CHW")
    plt.close()

def plotToImage(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = tv.io.decode_png(torch.Tensor(np.frombuffer(buf.getvalue(), np.uint8)).to(torch.uint8))
    buf.close()
    # Add the batch dimension
    # img = tf.expand_dims(img, 0)
    return img

def write_loss_summaries(writer, loss, loss_all, abs_step, mean_windowSize=10, mode="train"):
    wandb.log({f"sweep_{mode}_loss":loss, "global_step":abs_step})
    writer.add_scalars(main_tag="loss", tag_scalar_dict={'%s_1b_Average' % mode: loss}, global_step=abs_step, walltime=None)
    mean_window_loss = torch.mean(torch.stack(loss_all)) if len(loss_all)<=mean_windowSize else torch.mean(torch.stack(loss_all[-mean_windowSize:]))
    wandb.log({f"sweep_{mode}_loss_avg":mean_window_loss, "global_step":abs_step})
    writer.add_scalars(main_tag="loss", tag_scalar_dict={f'{mode}_{mean_windowSize}b_Average': mean_window_loss}, global_step=abs_step, walltime=None)
# add_video