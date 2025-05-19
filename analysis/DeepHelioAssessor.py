import glob
import os
import shutil
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import plots_utils
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class Model():
    """Single model output object. Precomputes bias"""
    def __init__(self,
                 model_path:str, 
                 target_var:str,
                 dates:dict,
                 dir_da=None,
                 dif_da=None,
                 is_mf_ds=False,
                 sis_da=None,
                 ):
        print(f"Instancing Model: {model_path}")
        if not is_mf_ds:
            model_da = xr.open_dataset(model_path).sel(time=slice(dates["start"],dates["end"]))[target_var]
            if "GHI" in target_var:
                target_da = dif_da + dir_da
            elif "DIR" in target_var:
                target_da = dir_da
            else:
                target_da = dif_da
        else: 
            model_da = xr.open_mfdataset(os.path.join(model_path, "**/*.nc"), parallel=True).sel(time=slice(dates["start"],dates["end"]))[target_var]
            target_da = sis_da.sel(x=slice(model_da.x.min(),model_da.x.max()),
                                   y=slice(model_da.y.min(),model_da.y.max()))
            target_da = target_da.reindex_like(model_da, method="nearest") 
        
        self.model_nc = xr.Dataset(data_vars={ f"{target_var}": model_da.drop_duplicates(dim="time",keep="first"),
                                               f"target_{target_var}": target_da,
                                               })
        self.model_nc["be"] = self.model_nc[target_var] - self.model_nc[f"target_{target_var}"]
         
        self.target_var = target_var
        self.model_nc_res = {}

# object to plot 
class mDeepHelioAssessor():
    """ object to plot the experiment results"""
    def __init__(self, 
                 var: str,
                 modes: dict,
                 models_paths: dict, 
                 helio_root: str,
                 smn_path:str,
                 smn_og_var:str,
                 baseline_paths:dict,
                 ablation_paths:dict,
                 permTest_paths:dict,
                 prod_paths:dict,
                 year: int,
                 output_root:str
                 ):
        """
        model_path: dict of models nc with their var as key
        helio_root: path to heliomont dataset root path
        year: assessement year

        smn_root: swissmetnet root path
        """
        self.var = var
        self.modes = modes
        self.year = year
        self.models_paths = models_paths
        self.baseline_paths = baseline_paths
        self.ablation_paths = ablation_paths
        self.permTest_paths = permTest_paths

        self.prod_paths = prod_paths
        self.helio_root = helio_root

        self.smn_path = smn_path
        self.smn_og_var = smn_og_var
   
        self.output_root = output_root

        # modes
        self.ki_ALB_plot = self.modes["ki_ALB_plot"]
        self.seqlen_map_plot = self.modes["seqlen_map_plot"]
        self.scat_ot_plot = self.modes["scat_ot_plot"]
        self.ablation_plot = self.modes["ablation_plot"]
        self.permTest_plot = self.modes["permTest_plot"]
        self.prod_smn_plot = self.modes["prod_smn_plot"]

        
        # to be filled
        self.gdf_smn = None
        self.ds_aux = None
        self.nc_helio = None
        self.nc_alb = None
        self.ch_shp = gpd.read_file("data/aux/ch_boundaries/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp").to_crs("4326")
        self.nc_models = {}
        self.nc_baselines = {}
        self.nc_ablations = {}
        self.nc_permTests = {}
        self.nc_prod = {}

        # dates
        self.dates = {"start":pd.to_datetime(f"{year}-01-01T00:00:00"),
                    "end":pd.to_datetime(f"{year+1}-01-01T00:00:00")}
        # tois
        self.tois = {
                    "ALL":{"start":pd.to_datetime(f"{year}-01-01T00:00:00"),
                                   "end":pd.to_datetime(f"{year+1}-01-01T00:00:00")},
                    "JFM":{"start":pd.to_datetime(f"{year}-01-01T00:00:00"),
                                   "end":pd.to_datetime(f"{year}-04-01T00:00:00")},
                    "AMJ":{"start":pd.to_datetime(f"{year}-04-01T00:00:00"),
                                   "end":pd.to_datetime(f"{year}-07-01T00:00:00")},
                    "JAS":{"start":pd.to_datetime(f"{year}-07-01T00:00:00"),
                                   "end":pd.to_datetime(f"{year}-10-01T00:00:00")},
                    "OND":{"start":pd.to_datetime(f"{year}-10-01T00:00:00"),
                                   "end":pd.to_datetime(f"{year+1}-01-01T00:00:00")}
                    }

        # regions defined by altitude bands or albedo bands
        self.alti_bands = {"low":[ 0, 1000, 2000, 3000],"top":[1000, 2000, 3000, 4359]}
        self.alti_bands_smn = {"low":[ 0, 1000, 2000, 0],"top":[1000, 2000, 4000, 4000]}
        self.ALB_bands = {"low":[0,0.3,0.6,0],"top":[0.3,0.6,1,1]}
        self.KI_bands = {"low":[0,0.4,0.8,0],"top":[0.4,0.8,1.2,1.2]}
        
        # plot cosmetics
        self.model_key_order_dict = {"HRV":"HRV","IR_016":"IR_016",
                                "IR_039":"IR_039","WV_062":"WV_062", 
                                "WV_073":"WV_073","IR_087":"IR_087",
                                "IR_097":"IR_097","IR_108":"IR_108",
                                "IR_120":"IR_120","IR_134":"IR_134",
                                "SAA":"SAA","SZA":"SZA","SRTMGL3_DEM":"DEM",
                                "aspectCos":"Cos(Aspect)", "aspectSin":"Sin(Aspect)",
                                "slope":"Slope","sw_dir_cor":"$f_{corr}$"}
        # read datasets
        self.read_data()

        #################
        ###
        # 1 intermodels plots RMSE vs. seqlen with and without ALB (hue: elevation, KI)
        ###
        if self.ki_ALB_plot:
            self.plot_ki_alb()
            self.plot_seqlen_alti()
            
            
        if self.seqlen_map_plot:
            self.plot_seqlenMaps()
        
        ###
        # 2 scatter plots target vs. output
        ###
        if self.scat_ot_plot: 
            sample4scatter_ratio = 1
            # self.plot_scat_ot_alti(sample4scatter_ratio, seqlen_sel=42, prod=False)
            self.plot_scat_ot(sample4scatter_ratio, seqlen_sel=40, prod=False)
        
        ###
        # 3 ablation
        ###
        if self.ablation_plot:
            self.plot_ablation()
            
        if self.permTest_plot:
            self.plot_permTest()
        
        ###
        # 4 prod
        ###
        if self.prod_smn_plot:
            self.smn_compare_plot()    

        if self.gen_plot:
            self.plot_gen()
            
        #################
        
    def read_data(self):    

        ############## srtm/nasadem data
        srtm_da = xr.open_dataset(glob.glob(os.path.join(self.helio_root,f"**/SRTMGL3_DEM.nc"),recursive=True)[0]).isel(time=0)["SRTMGL3_DEM"]

        ############## target data
        if "DIF" not in self.var:
            dir_da = xr.open_dataset(glob.glob(os.path.join(self.helio_root,f"**/{"SISDIR-No-Horizon"}*.nc"),recursive=True)[0]).sel(time=slice(self.dates["start"],self.dates["end"]))["SISDIR-No-Horizon"]
            # self.nc_helio  = xr.Dataset({"SISDIR-No-Horizon":nc_helio_dir["SISDIR-No-Horizon"]})

        if "DIR" not in self.var:
            dif_da = xr.open_dataset(glob.glob(os.path.join(self.helio_root,f"**/{"SISDIF-No-Horizon"}*.nc"),recursive=True)[0]).sel(time=slice(self.dates["start"],self.dates["end"]))["SISDIF-No-Horizon"]

        # alb - check and transpose coordinates if needed
        alb_da = xr.open_dataset(glob.glob(os.path.join(self.helio_root,f"**/ALB.nc"),recursive=True)[0]).sel(time=slice(self.dates["start"],self.dates["end"]))["ALB"]
        
        # Ki - check and transpose coordinates if needed
        ki_da = xr.open_dataset(glob.glob(os.path.join(self.helio_root,f"**/KI.nc"),recursive=True)[0]).sel(time=slice(self.dates["start"],self.dates["end"]))["KI"]

        self.ds_aux = xr.Dataset(data_vars={"KI":ki_da,
                                        "ALB":alb_da,
                                        "srtm":srtm_da,
                                        })

        ############## seqlen model data
        if (not self.prod_smn_plot) and (not self.ablation_plot) and (not self.permTest_plot):
            
            for model_key, model_path in self.models_paths.items():
                self.nc_models[model_key] = Model(model_path, self.var, self.dates, dir_da, dif_da)
            
        ############## baseline data
        if self.ki_ALB_plot:
            for model_key, model_path in self.baseline_paths.items():
                self.nc_baselines[model_key] = Model(model_path, self.var, self.dates, dir_da, dif_da)

        ############## ablation data
        if self.ablation_plot:
            for model_key, model_path in self.ablation_paths.items():
                self.nc_ablations[model_key] = Model(model_path, self.var, self.dates, dir_da, dif_da)

        ############## permTest data
        if self.permTest_plot:
            for model_key, model_path in self.permTest_paths.items():
                self.nc_permTests[model_key] = Model(model_path, self.var, self.dates, dir_da, dif_da)

        ############# prod data
        if self.prod_smn_plot:
            for key in self.prod_paths.keys():
                self.nc_prod[key] = Model(self.prod_paths[key],self.var, self.dates, dir_da, dif_da)
            self.gdf_smn = gpd.read_parquet(self.smn_path).rename(columns={self.smn_og_var:self.var}).set_index("stn") 
            self.gdf_smn = self.gdf_smn[self.gdf_smn["nan_ratio"] < 0.001] # remove bad stations         

    def plot_seqlen_alti(self): 
        stat_files_output = os.path.join(self.output_root["albseqlen"],"statFiles")
        Path(stat_files_output).mkdir(exist_ok=True, parents=True)

        for stat_type, ymin, ymax in zip(["rmse", "mbe","mae"],[0,-50,20],[120,50,90]):
            fig, ax = plt.subplots(1, len(self.alti_bands["low"]), figsize=(13.5,3.5), sharex=False, sharey=False)
            fig.supxlabel("Context size")
            fig.supylabel(f"{stat_type}".upper() + " [W/$m²$]")

            legend_elements =   [ 
                                Line2D([0], [0], marker='s', color='k', label='TSViT-r',
                                    markerfacecolor='k', markersize=10, linewidth=0),
                                Line2D([0], [0], color='k', lw=1, linestyle="--", label='ConvResNet'),
                                Patch(facecolor='k', edgecolor='k',label='$\\alpha \\notin$ input'),
                                Patch(facecolor='b', edgecolor='b',label='$\\alpha \\in$ input'),
                                ]
            
            for var, marker, color in zip(["wAlb", "woAlb"],["s","s"],["b","k"]):
                for m, (model_key, model) in enumerate(self.nc_models.items()):
                    if var in model_key:
                        print(f"Plotting {model_key}: bias vs. seqlen")

                        # elevation
                        for i,(alti_band_low, alti_band_top) in enumerate(zip(self.alti_bands["low"],self.alti_bands["top"])):
                            band_idx = (self.ds_aux["srtm"] >= alti_band_low) * (self.ds_aux["srtm"] < alti_band_top)
                            if stat_type == "mbe":
                                stat = (model.model_nc.where(band_idx)["be"]).mean()
                            if stat_type == "mae":
                                stat = abs(model.model_nc.where(band_idx)["be"]).mean()
                            if stat_type == "rmse":    
                                stat = (((model.model_nc.where(band_idx)["be"])**2).mean())**0.5
                            seqlen = int(model_key.split("seqlen")[-1])
                            ax[i].scatter(x=seqlen, y=stat, marker=marker, color=color, alpha=0.75)
                            # plot baseline (ResNet)
                            if model_key in self.nc_baselines.keys():
                                if stat_type == "mbe":
                                    stat_bl = self.nc_baselines[model_key].model_nc.where(band_idx)["be"].mean()
                                if stat_type == "mae":
                                    stat_bl = abs(self.nc_baselines[model_key].model_nc.where(band_idx)["be"]).mean()
                                if stat_type == "rmse":
                                    stat_bl = (((self.nc_baselines[model_key].model_nc.where(band_idx)["be"])**2).mean())**0.5 
                                
                                ax[i].hlines(y=stat_bl, xmin=1, xmax=int(list(self.nc_models.keys())[-1].split("seqlen")[-1]), color=color, alpha=0.5, linestyles="--")
                            sel_ratio = (sum(band_idx) / len(band_idx)).mean().values
                            # cosmetics
                            ax[i].grid(axis="y", alpha=0.2)
                            if m == (len(self.nc_models.items())-1):
                                alb_mean = self.ds_aux.where(band_idx)["ALB"].mean().values
                                # alb_mean = model_df.loc[band_idx,"ALB"].mean()
                                alb_mean_rounded = np.round(alb_mean,2)
                                sel_ratio_rounded = np.round(sel_ratio,2)
                                ax[i].text(0.5, 0.1, f'f={str(sel_ratio_rounded)}, $\\mu(\\alpha)$={str(alb_mean_rounded)}',fontsize=9, transform=ax[i].transAxes, horizontalalignment='center')
                            ax[i].set_ylim([ymin, ymax])
                            if i == 0:
                                ax[i].set_ylabel("")
                            else:
                                ax[i].tick_params(axis="y", length=0)
                                ax[i].set_yticklabels([])
                            if alti_band_top != max(self.alti_bands["top"]):
                                ax[i].set_title(f"$h\\in$[{alti_band_low}m-{alti_band_top}m[")
                            else:
                                ax[i].set_title(f"$h\\in$[{alti_band_low}m-{alti_band_top}m]")
     
                            with open(stat_files_output + f"/{var}_stats_AltiB{alti_band_low}-{alti_band_top}.txt","a") as f:
                                f.writelines([(f"Seqlen {seqlen}: {stat_type} --> {str(stat.values)} W/m² (baseline: {str(stat_bl.values)})\n")])
            
            ax[-1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), ncols=1)
            fig.tight_layout()
            fig.savefig(self.output_root["albseqlen"] + f"/1_{stat_type}_seqlen_alti_{self.year}.jpeg", dpi=300, bbox_inches = 'tight')
            plt.close("all")

    def plot_ki_alb(self):
        stat_files_output = os.path.join(self.output_root["albseqlen"],"statFiles")
        if os.path.exists(stat_files_output):
            print(f"Deleting:{stat_files_output}")
            shutil.rmtree(stat_files_output)
        Path(stat_files_output).mkdir(exist_ok=True, parents=True)
        
        for stat_type, ymin, ymax in zip(["rmse", "mbe","mae"],[0,-40,20],[120,40,90]):
            fig, ax = plt.subplots(len(self.ALB_bands["low"]), len(self.KI_bands["low"]), figsize=(13.5,12), sharex=False, sharey=False)
            fig.supxlabel("Context size")
            fig.supylabel(f"{stat_type}".upper() + " [W/$m²$]")

            legend_elements =   [ 
                                Line2D([0], [0], marker='s', color='k', label='TSViT-r',
                                    markerfacecolor='k', markersize=10, linewidth=0),
                                Line2D([0], [0], color='k', lw=1, linestyle="--", label='ConvResNet'),
                                Patch(facecolor='k', edgecolor='k',label='$\\alpha \\notin$ input'),
                                Patch(facecolor='b', edgecolor='b',label='$\\alpha \\in$ input'),
                                ]
            
            for var, marker, color in zip(["wAlb", "woAlb"],["s","s"],["b","k"]):
                for m, (model_key, model) in enumerate(self.nc_models.items()):
                    if var in model_key:
                        print(f"Plotting {model_key}: bias vs. seqlen")
                        for j,(alb_band_low, alb_band_top) in enumerate(zip(self.ALB_bands["low"],self.ALB_bands["top"])):
                            for k, (ki_band_low, ki_band_top) in enumerate(zip(self.KI_bands["low"],self.KI_bands["top"])):
        
                                ki_band_idx = (self.ds_aux["KI"] >= ki_band_low) * (self.ds_aux["KI"] < ki_band_top)
                                alb_band_idx = (self.ds_aux["ALB"] >= alb_band_low) * (self.ds_aux["ALB"] < alb_band_top)
                                if stat_type == "mbe":
                                    stat = (model.model_nc.where(ki_band_idx*alb_band_idx)["be"]).mean()
                                if stat_type == "mae":
                                    stat = (abs(model.model_nc.where(ki_band_idx*alb_band_idx)["be"])).mean()
                                if stat_type == "rmse":  
                                    stat = (((model.model_nc.where(ki_band_idx*alb_band_idx)["be"])**2).mean())**0.5  
                                seqlen = int(model_key.split("seqlen")[-1])
                                ax[k,j].scatter(x=seqlen, y=stat, marker=marker, color=color, alpha=0.75)
                                if model_key in self.nc_baselines.keys():
                                    if stat_type == "mbe":
                                        stat_bl = self.nc_baselines[model_key].model_nc.where(ki_band_idx*alb_band_idx)["be"].mean()
                                    if stat_type == "mae":
                                        stat_bl = abs(self.nc_baselines[model_key].model_nc.where(ki_band_idx*alb_band_idx)["be"]).mean()
                                    if stat_type == "rmse":
                                        stat_bl = (((self.nc_baselines[model_key].model_nc.where(ki_band_idx*alb_band_idx)["be"])**2).mean())**0.5 
                                    ax[k,j].hlines(y=stat_bl, xmin=1, xmax=int(list(self.nc_models.keys())[-1].split("seqlen")[-1]), color=color, alpha=0.5, linestyles="--")
                                sel_ratio = (sum(ki_band_idx*alb_band_idx) / len(ki_band_idx)).mean().values
                                # cosmetics
                                ax[k,j].grid(axis="y", alpha=0.2)
                                # bbox = dict(boxstyle='round', fc='w', ec='w', alpha=0)
                                if m == (len(self.nc_models.items())-1):
                                    alti_mean = self.ds_aux.where(ki_band_idx*alb_band_idx)["srtm"].mean().values
                                    # alti_mean = model_df.loc[ki_band_idx*alb_band_idx,"srtm"].mean()
                                    ax[k,j].text(0.5, 0.1, f'f={str(np.round(sel_ratio,2))}, $\\mu(h)$={str(np.round(alti_mean))}m',fontsize=9, transform=ax[k,j].transAxes, horizontalalignment='center')
                                ax[k,j].set_ylim([ymin, ymax])
                                if j == 0:
                                    if ki_band_top != 1.2:
                                        ax[k,j].set_ylabel(f"$k_T^*\\in$[{ki_band_low}-{ki_band_top}[")
                                    else:
                                        ax[k,j].set_ylabel(f"$k_T^*\\in$[{ki_band_low}-{ki_band_top}]")
                                else:
                                    ax[k,j].tick_params(axis="y", length=0)
                                    ax[k,j].set_yticklabels([])
                                
                                if k == 0:
                                    if alb_band_top != 1:
                                        ax[k,j].set_title(f"$\\alpha \\in$[{alb_band_low}-{alb_band_top}[")
                                    else: 
                                        ax[k,j].set_title(f"$\\alpha \\in$[{alb_band_low}-{alb_band_top}]")
                                if k < (len(self.ALB_bands["low"])-1):
                                    ax[k,j].tick_params(axis="x", length=0)
                                    ax[k,j].set_xticklabels([])

                                with open(stat_files_output + f"/{var}_stats_Alb{alb_band_low}-{alb_band_top}_Ki{ki_band_low}-{ki_band_top}.txt","a") as f:
                                    f.writelines([(f"Seqlen {seqlen}: {stat_type} --> {str(stat.values)} W/m² (baseline: {str(stat_bl.values)})\n")])

            
            ax[0,-1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), ncols=1)
            fig.tight_layout()
            fig.savefig(self.output_root["albseqlen"] + f"/1_{stat_type}_seqlen_albKi_{self.year}.jpeg", dpi=300, bbox_inches = 'tight')
            plt.close("all")

    def plot_seqlenMaps(self):
               
        seqlen_to_plot = [1,20,40,120]
        nb_seqlen = len(seqlen_to_plot)

        fig_w, axes_w = plt.subplots(2, nb_seqlen, figsize=(nb_seqlen * 4, 4),subplot_kw=dict(projection=ccrs.PlateCarree()), sharex=True, sharey=True)
        fig_wo, axes_wo = plt.subplots(2, nb_seqlen, figsize=(nb_seqlen * 4, 4),subplot_kw=dict(projection=ccrs.PlateCarree()), sharex=True, sharey=True)
        fig_diff, axes_diff = plt.subplots(2, nb_seqlen, figsize=(nb_seqlen * 4, 4),subplot_kw=dict(projection=ccrs.PlateCarree()), sharex=True, sharey=True)        

        for i,seqlen in enumerate(seqlen_to_plot):
            model_wAlb = self.nc_models[f"wAlb-seqlen{seqlen}"]
            model_woAlb = self.nc_models[f"woAlb-seqlen{seqlen}"]
            print(f"Plotting rmse map vs. seqlen: {seqlen}")
            mbe_map_w = model_wAlb.model_nc["be"].mean(dim="time")
            rmse_map_w = ((model_wAlb.model_nc["be"]**2).mean(dim="time"))**0.5
            mbe_map_wo = model_woAlb.model_nc["be"].mean(dim="time")
            rmse_map_wo = ((model_woAlb.model_nc["be"]**2).mean(dim="time"))**0.5
            mbe_map_diff = mbe_map_wo - mbe_map_w
            rmse_map_diff = rmse_map_wo - rmse_map_w
            

            if i != 0:
                ylabel = False
                gridDrawStrX = "x"
                gridDrawStrY = False
            else:
                ylabel = True
                gridDrawStrX = True
                gridDrawStrY = "y"
            if i != (nb_seqlen-1):
                cb = False
            else:
                cb = True
                
            
            self._plot_heliomap_0(mbe_map_w, ax=axes_w[0,i],vmin=-70, vmax=70, cmap="coolwarm", xlabel=False, ylabel=ylabel,cb=cb, cb_label="$\\Delta$ MBE", title=f"Context size: {seqlen} steps",gridDrawStr=gridDrawStrY)
            self._plot_heliomap_0(rmse_map_w, ax=axes_w[1,i], vmin=20, vmax=150, cmap="viridis", xlabel=True, ylabel=ylabel,cb=cb, cb_label="$\\Delta$ RMSE", gridDrawStr=gridDrawStrX)
            self._plot_heliomap_0(mbe_map_wo, ax=axes_wo[0,i],vmin=-70, vmax=70, cmap="coolwarm", xlabel=False, ylabel=ylabel,cb=cb, cb_label="$\\Delta$ MBE", title=f"Context size: {seqlen} steps", gridDrawStr=gridDrawStrY)
            self._plot_heliomap_0(rmse_map_wo, ax=axes_wo[1,i], vmin=20, vmax=150, cmap="viridis", xlabel=True, ylabel=ylabel,cb=cb, cb_label="$\\Delta$ RMSE", gridDrawStr=gridDrawStrX)
            self._plot_heliomap_0(mbe_map_diff, ax=axes_diff[0,i],vmin=-150, vmax=150, cmap="coolwarm", xlabel=False, ylabel=ylabel,cb=cb, cb_label="$\\Delta$ MBE", title=f"Context size: {seqlen} steps", gridDrawStr=gridDrawStrY)
            self._plot_heliomap_0(rmse_map_diff, ax=axes_diff[1,i], vmin=-50, vmax=50, cmap="coolwarm", xlabel=True, ylabel=ylabel,cb=cb, cb_label="$\\Delta$ RMSE", gridDrawStr=gridDrawStrX)
            i += 1
        

        # remove unnecessary labels and ticks 
        for axes in [axes_w, axes_wo, axes_diff]:
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    if j !=0: 
                        axes[i,j].set_yticks([],[])
                        axes[i,j].set_ylabel("")
                    if i == 0:
                        axes[i,j].set_xticks([],[])
                        # axes[i,j].set_xticklabels([])
                        axes[i,j].set_xlabel("")
                    
        fig_w.tight_layout()
        fig_w.savefig(self.output_root["albseqlen"] + f"/3_statwAlb_seqlen_map_{self.year}.jpeg", dpi=300, bbox_inches = 'tight')
        fig_wo.tight_layout()
        fig_wo.savefig(self.output_root["albseqlen"] + f"/3_statwoAlb_seqlen_map_{self.year}.jpeg", dpi=300, bbox_inches = 'tight')
        fig_diff.tight_layout()
        fig_diff.savefig(self.output_root["albseqlen"] + f"/3_statdiff_seqlen_map_{self.year}.jpeg", dpi=300, bbox_inches = 'tight')

    def plot_scat_ot(self, sample4scatter_ratio, seqlen_sel, prod):
        
        # single scatter
        print("Plotting scatter all")
        fig, ax = plt.subplots(1,3,figsize=(12,4), sharey=True)
        fig.supylabel("Output:  $GHI_{HeMu}$ [W/$m²$]")
        fig.supxlabel("Target:  $GHI_{HelioMont}$ [W/$m²$]")
        
        if prod:
            model = self.nc_prod[seqlen_sel].model_nc
            output_root = self.output_root["prod"]
        else:
            model = self.nc_models[f"woAlb-seqlen{seqlen_sel}"].model_nc
            output_root = self.output_root["albseqlen"]
            del(self.nc_models)
        indivScat_kwargs=dict(xlabel="", ylabel="")
        self._plot_scatter_01(y = model[self.var].values.flatten(),
                            x = model[f"target_{self.var}"].values.flatten(),
                            sample_ratio=sample4scatter_ratio,
                            ax=ax[0], **indivScat_kwargs)
        self._plot_scatter_01(y = model[self.var].resample({"time":"1D"}).mean().values.flatten(),
                            x = model[f"target_{self.var}"].resample({"time":"1D"}).mean().values.flatten(),
                            sample_ratio=sample4scatter_ratio,
                            ax=ax[1], **indivScat_kwargs)
        self._plot_scatter_01(y = model[self.var].resample({"time":"ME"}).mean().values.flatten(),
                            x = model[f"target_{self.var}"].resample({"time":"ME"}).mean().values.flatten(),
                            sample_ratio=sample4scatter_ratio,
                            ax=ax[2], **indivScat_kwargs)
        ax[0].set_title("Instantaneous")
        ax[1].set_title("Daily")
        ax[1].tick_params(axis="y", length=0)
        ax[2].set_title("Monthly")
        ax[2].tick_params(axis="y", length=0)
        for ax_ in ax:
            ax_.set_ylim([0,1250])
            ax_.set_xlim([0,1250])
        plt.tight_layout()
        fig.savefig(output_root + f"/2p_out_tar_Allscat_sample{sample4scatter_ratio}_{self.year}.jpeg", dpi=300, bbox_inches = 'tight')
        plt.close("all")

        # # split by csi and albedo
        # print("Plotting scatter by albedo and csi")
        # fig, ax = plt.subplots(len(self.ALB_bands["low"]), len(self.KI_bands["low"]), figsize=(14,12),
        #                        sharex=True, sharey=True, gridspec_kw = {'wspace':0.1*(14/12), 'hspace':0.1})
        # fig.subplots_adjust(left=0.1*(14/12), bottom=0.1)
        # fig.supylabel("Output [W/$m²$]")
        # fig.supxlabel("Target [W/$m²$]")
        # for j,(alb_band_low, alb_band_top) in enumerate(zip(self.ALB_bands["low"],self.ALB_bands["top"])):
        #         for k, (ki_band_low, ki_band_top) in enumerate(zip(self.KI_bands["low"],self.KI_bands["top"])):
        #             ki_band_idx = (self.ds_aux["KI"] >= ki_band_low) * (self.ds_aux["KI"] < ki_band_top)
        #             alb_band_idx = (self.ds_aux["ALB"] >= alb_band_low) * (self.ds_aux["ALB"] < alb_band_top)

        #         indivScat_kwargs=dict(xlabel="", ylabel="")
        #         self._plot_scatter_01(y = model.where(ki_band_idx*alb_band_idx)[self.var].values.flatten(),
        #                                 x = model.where(ki_band_idx*alb_band_idx)[f"target_{self.var}"].values.flatten(),
        #                                 sample_ratio=sample4scatter_ratio,
        #                                 ax=ax[k,j], **indivScat_kwargs)
        #         # cosmetics
        #         ax[k,j].grid(axis="y", alpha=0.2)
        #         if j == 0:
        #             ax[k,j].set_ylabel(f"$k_T^*\\in$:[{ki_band_low}-{ki_band_top}]")
        #         else:
        #             ax[k,j].tick_params(axis="y", length=0)
                
        #         if k == 0:
        #             ax[k,j].set_title(f"$\\alpha\\in$[{alb_band_low}-{alb_band_top}[")
        #         if k < (len(self.ALB_bands["low"])-1):
        #             ax[k,j].tick_params(axis="x", length=0)
        # # fig.tight_layout()
        # fig.savefig(output_root + f"/2p_out_tar_scat_sample{sample4scatter_ratio}_{self.year}.jpeg", dpi=300, bbox_inches = 'tight')
  
    def plot_ablation(self):
        fig, ax = plt.subplots(len(self.ALB_bands["low"]), len(self.KI_bands["low"]), figsize=(13.5,12), sharex=True, sharey=True)
        fig.supxlabel("Ablated feature")
        fig.supylabel("$\\Delta$ MAE [W/$m²$]")
        # get reference
        model_ref = self.nc_ablations["reference"].model_nc
        self.nc_ablations.pop("reference")
        # iterate on all ablations (sorted by value)
        for i, (model_key, plot_key) in enumerate(self.model_key_order_dict.items()):
            print(f"Plotting {model_key}-ablated score")
            model = self.nc_ablations[model_key]   
            # model_df = model.model_nc.to_dataframe()
            # # add classifying columm
            # model_df["KI_band"] = np.nan
            # model_df["ALB_band"] = np.nan
            
            for j,(alb_band_low, alb_band_top) in enumerate(zip(self.ALB_bands["low"],self.ALB_bands["top"])):
                for k, (ki_band_low, ki_band_top) in enumerate(zip(self.KI_bands["low"],self.KI_bands["top"])):
                    print(f"bar plot: {ki_band_low}-{ki_band_top} x {alb_band_low}-{alb_band_top}")
                    ki_band_idx = (self.ds_aux["KI"] >= ki_band_low) * (self.ds_aux["KI"] < ki_band_top)
                    alb_band_idx = (self.ds_aux["ALB"] >= alb_band_low) * (self.ds_aux["ALB"] < alb_band_top)
                   
                    stat_ref = abs(model_ref.where(ki_band_idx*alb_band_idx)["be"]).mean()#**2).mean())**0.5
                    stat = abs(model.model_nc.where(ki_band_idx*alb_band_idx)["be"]).mean()#**2).mean())**0.5
                    
                    ax[k,j].bar(i, stat-stat_ref, color="k")
                    
                    ax[k,j].set_ylim((-5,80))
                    if i == 0:
                        ax[k,j].grid(axis="y", alpha=0.3)
                        ax[k,j].grid(axis="x", alpha=0.3)
                        ax[k,j].set_xticks(np.arange(len(self.model_key_order_dict.keys())),tuple(self.model_key_order_dict.values()), rotation=90)
                    # cosmetics
                    if j == 0:
                        if ki_band_top != 1.2:
                            ax[k,j].set_ylabel(f"$k_T^*$:[{ki_band_low}-{ki_band_top}[")
                        else:
                            ax[k,j].set_ylabel(f"$k_T^*$:[{ki_band_low}-{ki_band_top}]")
                    else:
                        # ax[k,j].set_yticks([], [])
                        ax[k,j].tick_params(axis="y", length=0)
                    if k == 0:
                        if alb_band_top != 1:
                            ax[k,j].set_title(f"ALB:[{alb_band_low}-{alb_band_top}[")
                        else:
                            ax[k,j].set_title(f"ALB:[{alb_band_low}-{alb_band_top}]")
                    if k < (len(self.ALB_bands["low"])-1):
                        ax[k,j].tick_params(axis="x", length=0) 
            
            # del(model_df)
            
        fig.tight_layout()
        fig.savefig(self.output_root["ablation"] + f"/3_ablation_{self.year}.jpeg", dpi=300, bbox_inches = 'tight')
        
    def plot_permTest(self):
        fig, ax = plt.subplots(len(self.ALB_bands["low"]), len(self.KI_bands["low"]), figsize=(13.5,12), sharex=True, sharey=True)
        fig.supxlabel("Shuffled feature")
        fig.supylabel("$\\Delta$ MAE [W/$m²$]")
        # get reference
        model_ref = self.nc_permTests["reference"].model_nc
        self.nc_permTests.pop("reference")

        
        for i, (model_key, plot_key) in enumerate(self.model_key_order_dict.items()):
            print(f"Plotting {model_key}-permTest score")
            model = self.nc_permTests[model_key]
            for j,(alb_band_low, alb_band_top) in enumerate(zip(self.ALB_bands["low"],self.ALB_bands["top"])):
                for k, (ki_band_low, ki_band_top) in enumerate(zip(self.KI_bands["low"],self.KI_bands["top"])):
                
                    ki_band_idx = (self.ds_aux["KI"] >= ki_band_low) * (self.ds_aux["KI"] < ki_band_top)
                    alb_band_idx = (self.ds_aux["ALB"] >= alb_band_low) * (self.ds_aux["ALB"] < alb_band_top)
                   
                    stat_ref = abs(model_ref.where(ki_band_idx*alb_band_idx)["be"]).mean()#**2).mean())**0.5
                    stat = abs(model.model_nc.where(ki_band_idx*alb_band_idx)["be"]).mean()#**2).mean())**0.5 
                    ax[k,j].bar(i, stat-stat_ref, color="k")
                    
                    if i == 0:
                        ax[k,j].grid(axis="y", alpha=0.3)
                        ax[k,j].grid(axis="x", alpha=0.3)
                        ax[k,j].set_xticks(np.arange(len(self.model_key_order_dict.keys())),tuple(self.model_key_order_dict.values()), rotation=90)
                    # cosmetics
                    if j == 0:
                        if ki_band_top != 1.2:
                            ax[k,j].set_ylabel(f"$k_T^* \\in$[{ki_band_low}-{ki_band_top}[")
                        else:
                            ax[k,j].set_ylabel(f"$k_T^* \\in$[{ki_band_low}-{ki_band_top}]")
                    else:
                        ax[k,j].tick_params(axis="y", length=0)
                    if k == 0:
                        if alb_band_top != 1:
                            ax[k,j].set_title(f"$\\alpha \\in$[{alb_band_low}-{alb_band_top}[")
                        else:
                            ax[k,j].set_title(f"$\\alpha \\in$[{alb_band_low}-{alb_band_top}]")
                    if k < (len(self.ALB_bands["low"])-1):
                        ax[k,j].tick_params(axis="x", length=0) 
            
        fig.tight_layout()
        fig.savefig(self.output_root["permTest"] + f"/3_permTest_{self.year}.jpeg", dpi=300, bbox_inches = 'tight')
     
    def smn_compare_plot(self):
        """compares model and helio to SMN ground stations"""
        agg_str = ["1h","1D","ME"]
        agg_str_dict = {"1h":"Instantaneous", "1D": "Daily", "ME":"Monthly"}

        vmins = [0,0,-20] # modet, target, diff
        vmaxs = [180,180,20]
        cmaps = [mpl.cm.viridis,mpl.cm.viridis,mpl.cm.coolwarm]
        
        for key_nc_prod, nc_prod in self.nc_prod.items():       
            fig_model, axes_model = plt.subplots(len(self.alti_bands["low"]),len(agg_str), figsize=(14,11), sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
            fig_target, axes_target = plt.subplots(len(self.alti_bands["low"]),len(agg_str), figsize=(14,11), sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
            fig_diff, axes_diff = plt.subplots(len(self.alti_bands["low"]),len(agg_str), figsize=(14,11), sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
            # fig.supxlabel("Time resolution")
            for i, (fig, ax, cbar_label) in enumerate(zip([fig_model,fig_target,fig_diff], [axes_model,axes_target,axes_diff],["RMSE", "RMSE", "$\\Delta RMSE$"])):
                fig.supylabel("Altitude band")
                fig.subplots_adjust(left=0.05,right=0.9)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cmap = cmaps[i]
                norm = mpl.colors.Normalize(vmin=vmins[i], vmax=vmaxs[i])
                fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax, orientation='vertical', label=f'{cbar_label} [W/$m^2$]')
            for j, t_res in enumerate(agg_str): # ME: month end
                for i, (alti_low, alti_top) in enumerate(zip(self.alti_bands_smn["low"],self.alti_bands_smn["top"])): # filter altitudes
                    print(f"Prod plot: {t_res}:{alti_low}-{alti_top}")
                    gdf_smn_alti = self.gdf_smn[(self.gdf_smn["altitude"] >= alti_low) & (self.gdf_smn["altitude"] < alti_top)]
                    gdf_smn_alti["rmse"] = np.nan
                    gdf_smn_alti["mae"] = np.nan
                    gdf_smn_alti["mbe"] = np.nan
                    for stn in gdf_smn_alti.index:
                        # read model
                        if isinstance(nc_prod, Model):
                            da = nc_prod.model_nc[self.var]
                            da_target = nc_prod.model_nc[f"target_{self.var}"]
                        else:
                            da = nc_prod[self.var]
                            da_target = nc_prod[f"target_{self.var}"]
                        df_model = da.sel(x=self.gdf_smn.loc[stn,"lon"], y=self.gdf_smn.loc[stn,"lat"], method="nearest").to_dataframe().resample("1h").asfreq(fill_value=np.nan)
                        df_target = da_target.sel(x=self.gdf_smn.loc[stn,"lon"], y=self.gdf_smn.loc[stn,"lat"], method="nearest").to_dataframe().resample("1h").asfreq(fill_value=np.nan)
                        # read stns
                        df_smn = pd.DataFrame({"time" : pd.to_datetime(self.gdf_smn.loc[stn,"time"]),self.var:self.gdf_smn.loc[stn,self.var]}).set_index("time")
                        # crop time extent to df_model (align datasets)
                        df_smn = df_smn.loc[df_model.index]
                        # actually remove sza filtered values from meas (otherwise big error!)
                        df_smn.loc[df_model[self.var].isna(),:] = np.nan 
                        df_target.loc[df_model[self.var].isna(),:] = np.nan         
                        # resample datasets
                        df_smn = df_smn.resample(t_res).mean()
                        df_model = df_model.resample(t_res).mean()
                        df_target = df_target.resample(t_res).mean()
                        
                        # compute stats
                        be_model = df_model[self.var] - df_smn[self.var] 
                        be_target = df_target[f"target_{self.var}"] - df_smn[self.var] 
                        # be_diff = be_model - be_target

                        for varType, be in zip(["target","model"],[be_target, be_model]):
                            gdf_smn_alti.loc[stn,f"mbe_{varType}"] = np.nanmean(be)
                            gdf_smn_alti.loc[stn,f"mae_{varType}"] = np.nanmean(abs(be))
                            gdf_smn_alti.loc[stn,f"rmse_{varType}"] = (np.nanmean(be**2))**0.5
                        
                        gdf_smn_alti.loc[stn,f"mbe_diff"] = gdf_smn_alti.loc[stn,f"mbe_model"] - gdf_smn_alti.loc[stn,f"mbe_target"]
                        gdf_smn_alti.loc[stn,f"mae_diff"] = gdf_smn_alti.loc[stn,f"mae_model"] - gdf_smn_alti.loc[stn,f"mae_target"]
                        gdf_smn_alti.loc[stn,f"rmse_diff"] = gdf_smn_alti.loc[stn,f"rmse_model"] - gdf_smn_alti.loc[stn,f"rmse_target"]

                    for k, (varTyp, axes, vmin, vmax)in enumerate(zip(["model","target","diff"],[axes_model,axes_target,axes_diff],vmins, vmaxs)):
                        rmse = gdf_smn_alti[f"rmse_{varTyp}"].mean()
                        mae = gdf_smn_alti[f"mae_{varTyp}"].mean()
                        mbe = gdf_smn_alti[f"mbe_{varTyp}"].mean()

                        self.ch_shp.plot(ax=axes[i,j],edgecolor="k", facecolor="w")

                        gdf_smn_alti.plot(column=f"rmse_{varTyp}", ax=axes[i,j], legend=False, vmin=vmin, vmax=vmax, cmap=cmaps[k])

                        axes[i,j].spines['top'].set_visible(False)
                        axes[i,j].spines['right'].set_visible(False)
                        axes[i,j].spines['bottom'].set_visible(False)
                        axes[i,j].spines['left'].set_visible(False)
                        if j == 0:
                            axes[i,j].set_ylabel(f"h$\\in$[{alti_low}m-{alti_top}m[")
                        else:
                            axes[i,j].set_yticklabels([])
                        axes[i,j].tick_params(axis="y", length=0)
                        
                        
                        if i < (len(self.alti_bands["low"])-1):
                            axes[i,j].set_xticklabels([])
                        axes[i,j].tick_params(axis="x", length=0)
                        if i == 0:
                            axes[i,j].set_title(f"{agg_str_dict[t_res]}")
                       
            # save plot
            for varTyp, fig in zip(["model", "target", "diff"],[fig_model, fig_target, fig_diff]):
                filename = f"4_prod_{varTyp}_{key_nc_prod}_y{self.year}"
                fig.savefig(os.path.join(self.output_root["prod"], f"{filename}.jpeg"), dpi=300)
            plt.close("all")

    def _plot_scatter_01(self, x, y, sample_ratio, title=None, ax=None, r2=True, **kwargs):
        print("Scatter plot: removing nans from vectors x and y.")
        idx_nan = np.isnan(x)
        x = x[~idx_nan]
        y = y[~idx_nan]
        idy_nan = np.isnan(y)
        x = x[~idy_nan]
        y = y[~idy_nan]
        print("Negative model output set to 0")
        x[x<0] = 0

        if sample_ratio < 1:
            rng = np.random.default_rng()
            idx = rng.choice(np.arange(sample_ratio*x.shape[0]), [int(sample_ratio*x.shape[0])]).astype(int)
            fig, ax = plots_utils.scatter_01(x[idx], y[idx], density_bins=5, ax=ax, cmap="Spectral_r", **kwargs)
        else:
            fig, ax = plots_utils.scatter_01(x, y, density_bins=5, ax=ax, cmap="Spectral_r", **kwargs) 
            
        mbe = np.nanmean(x-y)
        rmse = (np.nanmean((x-y)**2))**0.5
        mae = np.nanmean(abs(x-y))
        if title is not None:
            ax.set_title(title)
        if r2:
            bbox = dict(boxstyle='round', fc='blanchedalmond', ec='w', alpha=0.5)
            ax.text(0.8, 0.05, f'RMSE = {rmse:.1f}\nMBE = {mbe:.1f}\nMAE = {mae:.1f}', 
                    fontsize=9, bbox=bbox, transform=ax.transAxes, horizontalalignment='left')

        return fig, ax

    def _plot_heliomap_0(self, da, **kwargs):

        # extract cosmetics
        if "title" in kwargs.keys():
            title=kwargs["title"]
            kwargs.pop("title")
        else:
            title=""
        if "xlabel" in kwargs.keys():
            xlabel=kwargs["xlabel"]
            kwargs.pop("xlabel")
        else:
            xlabel=False
        
        if "ylabel" in kwargs.keys():
            ylabel=kwargs["ylabel"]
            kwargs.pop("ylabel")
        else:
            ylabel=False
            
        if "cb" in kwargs.keys():
            cb=kwargs["cb"]
            kwargs.pop("cb")
        else:
            cb=False
        
        if "cb_label" in kwargs.keys():
            cb_label=kwargs["cb_label"]
            kwargs.pop("cb_label")
        else:
            cb_label=""
        
        if "gridDrawStr" in kwargs.keys():
            gridDrawStr=kwargs["gridDrawStr"]
            kwargs.pop("gridDrawStr")
        
        # plot
        p = da.plot(transform=ccrs.PlateCarree(),**kwargs)
        p.axes.add_feature(cfeature.BORDERS, linestyle='-')
        p.axes.add_feature(cfeature.COASTLINE, linestyle='-')
        gls = p.axes.gridlines(draw_labels=gridDrawStr, alpha=0.5)
        gls.top_labels=False   # suppress top labels
        gls.right_labels=False # suppress right labels
        gls.xlines=False
        gls.ylines=False

        # xtick and xticklabels
        if not ylabel:
            # p.axes.set_yticks([])
            p.axes.set_yticklabels([])
        if not xlabel:
            # p.axes.set_xticks([])
            p.axes.set_xticklabels([])
        # colorbar
        if not cb:
            p.colorbar.ax.set_visible(False)
        else:
            p.colorbar.ax.set_ylabel(cb_label)
        # set cosmetics
        ax = kwargs["ax"]
        ax.set_title(title)


        return kwargs["ax"].get_figure()

