import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import numpy as np
from scipy.stats import gaussian_kde, linregress

def density_scatter(x , y, fig = None, ax = None, sort = True, bins = 20, **kwargs) :
        """
        Scatter plot colored by 2d histogram
        """
        if ax is None :
            fig , ax = plt.subplots()
        else:
            fig = ax.get_figure()
        data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = False )
        z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "linear", bounds_error = False, fill_value=None)

        #To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort :
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        ax.scatter(x, y, c=z, s=1, marker="x", **kwargs)

        norm = Normalize(vmin = 0, vmax = np.max(z))

        cbar = fig.colorbar(cm.ScalarMappable(norm = norm,**kwargs), ax=ax, pad=0.0, format='%.0e')

        # Get the full range of the colorbar
        tick_values = cbar.get_ticks()
        cbar.set_ticks([])
        top_val = tick_values[-1]
        cbar.ax.text(0.5, 1, f"N={top_val:.0f}", va='top', ha='center', rotation=90, transform=cbar.ax.transAxes, fontsize=9)
        cbar.ax.text(0.5, 0, f"N=0", va='bottom', ha='center', rotation=90, transform=cbar.ax.transAxes, fontsize=9)
        ax.text(0.5,0.95,'$N_{total}$'+ f'={x.shape[0]}', va='center', ha='center', transform=ax.transAxes)
        
        return fig, ax

def scatter_01(x, y, var=None, timestamp=None, density_bins=20, ax=None, **kwargs):
    # compute regression
    # get and remove axes labels from kwargs if any
    if "xlabel" in kwargs.keys():
        xlabel = kwargs["xlabel"]
        kwargs.pop("xlabel", None)
    if "ylabel" in kwargs.keys():
        ylabel = kwargs["ylabel"]
        kwargs.pop("ylabel", None)

    fig, ax = density_scatter(x,y,bins=density_bins, ax=ax, **kwargs)
    ax.plot([0,max(x.max(),y.max())],[0,max(x.max(),y.max())], "k-", lw=1) 
    if "xlabel" in kwargs.keys():
        ax.set_xlabel(xlabel)
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(ylabel)
    return fig, ax