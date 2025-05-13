
import os
from DeepHelioAssessor import mDeepHelioAssessor

if __name__ ==  "__main__":

    var = "SISGHI-No-Horizon" # select which variable from HelioMont to assess
    helio_root = "data/Helio/2015_2020/sza80" # target root folder
    smn_og_var = "GHI_10min_avg" # select which smn resolution
    smn_path = f"data/SMN/stn_CH_{smn_og_var}_2015_2020_123units.pqt" # ground station gdf path
    
    modes = {"ki_ALB_plot":False, # Context size experiment scatter plots
             "seqlen_map_plot":False, # Context size experiment maps
             "scat_ot_plot":False, # output-target scatter
             "ablation_plot":False, # ablation plot
             "permTest_plot":False, # permutation test like plot
             "prod_smn_plot":False # prod model plot with smn stations
             } 
    
    assessment_years = [2019] # years to asses
    for year in assessment_years:
        # intermodels comparison
        exp_root = "analysis/runs/TSViT-r/SISGHI/contextSizeExp"
        baseline_root = "analysis/runs/ConvResNet/SISGHI"
        ablation_root = "analysis/runs/TSViT-r/SISGHI/ablation_var"
        permTest_root = "analysis/runs/TSViT-r/SISGHI/permTest"
        prod_root = "analysis/runs/TSViT-r/SISGHI/prod" 
        models_path_dict = {}
        baseline_path_dict = {}
        ablation_path_dict = {}
        permTest_path_dict = {}
        prod_path_dict = {}
        for targetVar in [var.split("-")[0][-3:]]:
            
            output_root = {
                            "albseqlen": exp_root + "/" + targetVar,
                            "ablation": ablation_root + "/" + targetVar,
                            "permTest": permTest_root + "/" + targetVar,
                            "prod": prod_root + "/" + targetVar,
                            }
            #############
            # 1. Context size experiment
            #############
            for inputVar in ["wAlb", "woAlb"]:   
                # baseline
                folderfilelist = os.listdir(baseline_root + f"/{targetVar}" + f"/{inputVar}" + f"/1/")
                if len(folderfilelist) == 1:
                    baseline_filename = folderfilelist[0] 
                    baseline_path_dict[f"{inputVar}-seqlen1"] = baseline_root + f"/{targetVar}" + f"/{inputVar}" + f"/1" + f"/{baseline_filename}"
                # tsvit
                for seqlen in [1,5,10,20,30,40,80,120]: #
                    folderfilelist = os.listdir(exp_root + f"/{targetVar}" + f"/{inputVar}" + f"/{seqlen}")
                    if len(folderfilelist) == 1:
                        filename = folderfilelist[0]
                        models_path_dict[f"{inputVar}-seqlen{seqlen}"] = exp_root + f"/{targetVar}" + f"/{inputVar}" + f"/{seqlen}" + f"/{filename}" #+ f"/SIS{var}-No-Horizon.nc" 
            
            ###############
            # 2. Ablation
            ###############
            folderlist = os.listdir(ablation_root + f"/{targetVar}")
            for ablatedVar in folderlist:
                if os.path.isdir(ablation_root + f"/{targetVar}" + f"/{ablatedVar}"):
                    ablation_path_dict[f"{ablatedVar}"] = ablation_root + f"/{targetVar}" + f"/{ablatedVar}" + f"/{var}.nc"
                    
            ################
            # Permutation
            ################
            folderlist = os.listdir(permTest_root + f"/{targetVar}")
            for permVar in folderlist:
                if os.path.isdir(permTest_root + f"/{targetVar}" + f"/{permVar}"):
                    permTest_path_dict[f"{permVar}"] = permTest_root + f"/{targetVar}" + f"/{permVar}" + f"/{var}.nc"
            
            ################
            # Station measurements
            ################
            epoch_num = 10
            filter_sza = 80
            prodseqlen = [40]
            training_nbYears = 2
            for seqlen in prodseqlen:
                file = prod_root + f"/{targetVar}/{seqlen}ctx/{epoch_num}ep/{filter_sza}sza/{training_nbYears}y/{var}.nc"
                if os.path.exists(file):
                    prod_path_dict[f"_{seqlen}ctx_ep{epoch_num}_{training_nbYears}tyears_sza{filter_sza}"] = file
            
            
        mDeepHelioAssessor(var, modes, models_path_dict, helio_root,
                                    smn_path, smn_og_var, baseline_path_dict,
                                    ablation_path_dict, permTest_path_dict,
                                    prod_path_dict, year, output_root)