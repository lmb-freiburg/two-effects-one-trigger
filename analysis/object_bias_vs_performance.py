import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
import numpy as np
from scipy.stats import kendalltau
from tueplots import bundles
import seaborn as sns
import pandas as pd
from copy import deepcopy

from utils import read_vals_from_json_files, get_small_and_large_indices

obj_metrics = ['moad_txt', 'moad_img']
datasets = ["mit-states", "ut-zappos"]
robust = True

def moad(sim_o, sim_no, sim_a, sim_na):
    return (sim_o - sim_no) - (sim_a - sim_na)

def main(args):
    json_files = sorted([os.path.join(args.output_dir_gap, f) for f in os.listdir(args.output_dir_gap) if ".json" in f and "coco" in f and f"coco_" == f[:len("coco")+1] and f"coco_old" not in f])
    assert len(json_files) == 115, "Expected 115 models, got " + str(len(json_files))
    coco_performance = read_vals_from_json_files(json_files, ["img2txt.R@1"])["img2txt.R@1"]

    model_names = [os.path.basename(f)[len("coco_"):-len(".json")] for f in json_files]    
    small_allowed_indices, large_allowed_indices = get_small_and_large_indices(model_names, coco_performance)
    assert len(small_allowed_indices) + len(large_allowed_indices) == 101
    
    plt.rcParams.update(bundles.cvpr2024(column="half", nrows=len(datasets), ncols=len(obj_metrics)))
    fig, axes = plt.subplots(len(datasets), len(obj_metrics))

    for didx, dataset in enumerate(datasets):
        print("#"*5, dataset, "#"*5)

        json_files = sorted([os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if ".json" in f and dataset in f and f"{dataset}_" == f[:len(dataset)+1] and f"{dataset}_old" not in f])
        assert len(json_files) == 115, "Expected 115 models, got " + str(len(json_files))

        vals_dict = read_vals_from_json_files(json_files, ["acc", "obj_bias"])
        performance = deepcopy(vals_dict["acc"])
        obj_bias = deepcopy(vals_dict["obj_bias"])
        
        if dataset == "mit-states":
            ylabel = "MIT-States"
        elif dataset == "ut-zappos":
            ylabel = "UT-Zappos"
        else:
            raise NotImplementedError
        ylabel += "\nattr accuracy"

        for obj_dix, obj_metric in enumerate(obj_metrics):
            objs = []
            for o in obj_bias:
                if "moad_txt" == obj_metric:
                    objs.append(
                        moad(o["txt_obj"]["mean_intra_sim"], o["txt_obj"]["mean_cross_sim"], o["txt_attr"]["mean_intra_sim"], o["txt_attr"]["mean_cross_sim"])
                    )
                elif "moad_img" == obj_metric:
                    objs.append(
                        moad(o["img_obj"]["mean_intra_sim"], o["img_obj"]["mean_cross_sim"], o["img_attr"]["mean_intra_sim"], o["img_attr"]["mean_cross_sim"])
                    )
                else:
                    raise NotImplementedError
            
            xlabel = "MOAD text" if "txt" in obj_metric else "MOAD image"
            ax = axes[didx, obj_dix]
            
            small_performance = [p for i, p in enumerate(performance) if i in small_allowed_indices]
            small_moad = np.array([d for i, d in enumerate(objs) if i in small_allowed_indices])
            
            large_performance = [p for i, p in enumerate(performance) if i in large_allowed_indices]
            large_moad = np.array([d for i, d in enumerate(objs) if i in large_allowed_indices])

            assert len(small_performance) + len(large_performance) == 101

            plot_df = pd.DataFrame({
                xlabel: large_moad,
                ylabel: large_performance,
            })
            sns.regplot(plot_df, x=xlabel, y=ylabel, ax=ax, scatter_kws={"s": 1, "zorder": 10, "color": sns.color_palette()[0]}, line_kws={"color": sns.color_palette()[0]}, robust=robust, fit_reg=False, label="large")

            plot_df = pd.DataFrame({
                xlabel: small_moad,
                ylabel: small_performance,
            })
            sns.regplot(plot_df, x=xlabel, y=ylabel, ax=ax, scatter_kws={"s": 1, "zorder": 10,  "color": sns.color_palette()[1]}, line_kws={"color": sns.color_palette()[1]}, robust=robust, fit_reg=False, label="medium")
            
            ax.grid()
            sns.despine(ax=ax)
            
            if obj_dix != 0:
                axes[didx, obj_dix].set_ylabel("")
            if didx != len(datasets)-1:
                axes[didx, obj_dix].set_xlabel("")

            if not (obj_dix == 0 and didx == 0):
                ax.legend([],[], frameon=False)
            else:
                ax.legend(title='Dataset size', labelspacing=0.2, handletextpad=0.2, borderpad=0.2, loc='best')

            print(obj_metric, "small", kendalltau(small_performance, small_moad))
            print(obj_metric, "large", kendalltau(large_performance, large_moad))
            print(obj_metric, "all", kendalltau(performance, objs))

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/object_bias_vs_performance.pdf")
    plt.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/performance_vs_object_bias")
    parser.add_argument("--output_dir_gap", type=str, default="results/performance_vs_modality_gap")
    args = parser.parse_args()
    main(args)