import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
import numpy as np
from scipy.stats import kendalltau, permutation_test
import seaborn as sns
import pandas as pd
import torch
from tqdm import tqdm
import math
from scipy.special import gamma
import open_clip
from torch import nn
import json

from tueplots import bundles

from utils import read_vals_from_json_files, get_small_and_large_indices, dataset_to_size


datasets = ["coco", "imagenet"]

robust = True
correlation_method = kendalltau
PVALUE = 0.05

def main(args):
    json_files = sorted([os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if ".json" in f and "coco" in f and f"coco_" == f[:len("coco")+1] and f"coco_old" not in f])
    assert len(json_files) == 115, "Expected 115 models, got " + str(len(json_files))
    coco_performance = read_vals_from_json_files(json_files, ["img2txt.R@1"])["img2txt.R@1"]

    model_names = [os.path.basename(f)[len("coco_"):-len(".json")] for f in json_files]    
    small_allowed_indices, large_allowed_indices = get_small_and_large_indices(model_names, coco_performance)
    assert len(small_allowed_indices) + len(large_allowed_indices) == 101

    for coco_type in ["i2t", "t2i"]:
        plt.rcParams.update(bundles.cvpr2024(column="full", nrows=1, ncols=4))
        fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False)

        for didx, dataset in enumerate(datasets):
            if dataset == "coco":
                tab_correlation_str = "MS COCO &"
            elif dataset == "imagenet":
                tab_correlation_str = "ImageNet &"
            else:
                raise NotImplementedError

            for dist_idx, dist_metric in enumerate([
                "L2M",
                "RMG",
                # "C2I_plus_cross_matching_intra", # RMG
            ]):
                xlabel = dist_metric

                ax = axes[dist_idx*2+didx]
                if dist_metric == "L2M":
                    print("#"*5, dataset, "#"*5)

                if dataset == "imagenet":
                    ylabel = "ImageNet\naccuracy"
                elif dataset == "coco":
                    ylabel = "MS COCO\nR@1"
                else:
                    raise NotImplementedError
                
                json_files = sorted([os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if ".json" in f and dataset in f and f"{dataset}_" == f[:len(f"{dataset}")+1] and f"{dataset}_old" not in f])
                assert len(json_files) == 115, "Expected 115 models, got " + str(len(json_files))
                
                if dist_metric == "L2M":
                    dist_metric_key = "l2_means"
                elif dist_metric == "RMG":
                    dist_metric_key = "rmg"
                else:
                    raise NotImplementedError
                shared_keys = ["emb_dim", "model_size", "train_dataset", dist_metric_key]
                if dataset == "imagenet":
                    key_perf = "acc"
                    vals_dict = read_vals_from_json_files(json_files, shared_keys + [key_perf])
                    performance = vals_dict["acc"]
                elif dataset == "coco":
                    key_perf = "img2txt.R@1" if coco_type == "i2t" else "txt2img.R@1"
                    vals_dict = read_vals_from_json_files(json_files, shared_keys + [key_perf])
                    performance = vals_dict[key_perf]
                embed_dims = vals_dict["emb_dim"]
                model_sizes = vals_dict["model_size"]
                dataset_sizes = []
                for d in vals_dict["train_dataset"]:
                    tmp = [k if d == "v1" and "nllb" in d else k for k in dataset_to_size.keys() if k in d]
                    if len(tmp) != 1:
                        print(d)
                    dataset_sizes.append(dataset_to_size[tmp[0]])
                dists = vals_dict[dist_metric_key]

                small_cond = lambda idx: idx in small_allowed_indices
                small_performance = np.array([p for i, p in enumerate(performance) if small_cond(i)])
                small_model_sizes = np.array([d for i, d in enumerate(model_sizes) if small_cond(i)])
                small_embeds = np.array([d for i, d in enumerate(embed_dims) if small_cond(i)])
                small_dataset_sizes = np.array([d for i, d in enumerate(dataset_sizes) if small_cond(i)])
                small_dists = np.array([d for i, d in enumerate(dists) if small_cond(i)])

                large_cond = lambda idx: idx in large_allowed_indices
                large_performance = [p for i, p in enumerate(performance) if large_cond(i)]
                large_model_sizes = np.array([d for i, d in enumerate(model_sizes) if large_cond(i)])
                large_embeds = np.array([d for i, d in enumerate(embed_dims) if large_cond(i)])
                large_dataset_sizes = np.array([d for i, d in enumerate(dataset_sizes) if large_cond(i)])
                large_dists = np.array([d for i, d in enumerate(dists) if large_cond(i)])

                assert len(small_performance) + len(large_performance) == 101
                
                permutation_pvalue = permutation_test((small_performance,), lambda x: correlation_method(x, small_dists).statistic, permutation_type='pairings').pvalue
                print(xlabel, "medium", correlation_method(small_performance, small_dists), permutation_pvalue)
                tab_correlation_str += " \\textcolor{seabornorange}{" + str(round(correlation_method(small_performance, small_dists).correlation, 3)) + " (\\" + f"{'cmark' if permutation_pvalue < PVALUE else 'xmark'}" + ")} /"
                
                permutation_pvalue = permutation_test((large_performance,), lambda x: correlation_method(x, large_dists).statistic, permutation_type='pairings').pvalue
                print(xlabel, "large", correlation_method(large_performance, large_dists), permutation_pvalue)
                tab_correlation_str += " \\textcolor{seabornblue}{" + str(round(correlation_method(large_performance, large_dists).correlation, 3)) + " (\\" + f"{'cmark' if permutation_pvalue < PVALUE else 'xmark'}" + ")} &"

                if dist_metric == "RMG": # only needed once
                    permutation_pvalue = permutation_test((small_performance,), lambda x: correlation_method(x, small_model_sizes).statistic, permutation_type='pairings').pvalue
                    print("Perf vs. model size", "medium", correlation_method(small_performance, small_model_sizes), permutation_pvalue)
                    tab_correlation_str += " \\textcolor{seabornorange}{" + str(round(correlation_method(small_performance, small_model_sizes).correlation, 3)) + " (\\" + f"{'cmark' if permutation_pvalue < PVALUE else 'xmark'}" + ")} /"

                    permutation_pvalue = permutation_test((large_performance,), lambda x: correlation_method(x, large_model_sizes).statistic, permutation_type='pairings').pvalue
                    print("Perf vs. model size", "large", correlation_method(large_performance, large_model_sizes), permutation_pvalue)
                    tab_correlation_str += " \\textcolor{seabornblue}{" + str(round(correlation_method(large_performance, large_model_sizes).correlation, 3)) + " (\\" + f"{'cmark' if permutation_pvalue < PVALUE else 'xmark'}" + ")} &"

                    permutation_pvalue = permutation_test((small_performance,), lambda x: correlation_method(x, small_embeds).statistic, permutation_type='pairings').pvalue
                    print("Perf vs. emb dim", "medium", correlation_method(small_performance, small_embeds), permutation_pvalue)
                    tab_correlation_str += " \\textcolor{seabornorange}{" + str(round(correlation_method(small_performance, small_embeds).correlation, 3)) + " (\\" + f"{'cmark' if permutation_pvalue < PVALUE else 'xmark'}" + ")} /"

                    permutation_pvalue = permutation_test((large_performance,), lambda x: correlation_method(x, large_embeds).statistic, permutation_type='pairings').pvalue
                    print("Perf vs. emb dim", "large", correlation_method(large_performance, large_embeds), permutation_pvalue)
                    tab_correlation_str += " \\textcolor{seabornblue}{" + str(round(correlation_method(large_performance, large_embeds).correlation, 3)) + " (\\" + f"{'cmark' if permutation_pvalue < PVALUE else 'xmark'}" + ")} &"

                    permutation_pvalue = permutation_test((small_performance,), lambda x: correlation_method(x, small_dataset_sizes).statistic, permutation_type='pairings').pvalue
                    print("Perf vs. dataset size", "medium", correlation_method(small_performance, small_dataset_sizes), permutation_pvalue)
                    tab_correlation_str += " \\textcolor{seabornorange}{" + str(round(correlation_method(small_performance, small_dataset_sizes).correlation, 3)) + " (\\" + f"{'cmark' if permutation_pvalue < PVALUE else 'xmark'}" + ")} /"

                    permutation_pvalue = permutation_test((large_performance,), lambda x: correlation_method(x, large_dataset_sizes).statistic, permutation_type='pairings').pvalue
                    print("Perf vs. dataset size", "large", correlation_method(large_performance, large_dataset_sizes), permutation_pvalue)
                    tab_correlation_str += " \\textcolor{seabornblue}{" + str(round(correlation_method(large_performance, large_dataset_sizes).correlation, 3)) + " (\\" + f"{'cmark' if permutation_pvalue < PVALUE else 'xmark'}" + ")} \\\\"

                plot_df = pd.DataFrame({
                    xlabel: large_dists,
                    ylabel: large_performance,
                    "dset_size": ["Large size"]*len(large_dists),
                })
                scatter1 = sns.regplot(plot_df, x=xlabel, y=ylabel, ax=ax, scatter_kws={"s": 1, "zorder": 10}, line_kws={"color": sns.color_palette()[0]}, robust=robust, label="large")
                plot_df = pd.DataFrame({
                    xlabel: small_dists,
                    ylabel: small_performance,
                    "dset_size": ["Medium size"]*len(small_dists),
                })
                scatter2 = sns.regplot(plot_df, x=xlabel, y=ylabel, ax=ax, scatter_kws={"s": 1, "zorder": 10}, line_kws={"color": sns.color_palette()[1]}, robust=robust, label="medium")

    
                sns.despine(ax=ax)
                ax.grid()

                if not (dist_idx == 0 and didx == 0):
                    ax.legend([],[], frameon=False)
                else:
                    ax.legend(title='Dataset size', labelspacing=0.2, handletextpad=0.2, borderpad=0.2, loc='best')

            print("\n")
            print(tab_correlation_str)
            print("\n")

        os.makedirs("figures", exist_ok=True)
        if coco_type == "i2t":
            plt.savefig(f"figures/performance_vs_modality_gap.pdf")
        plt.savefig(f"figures/modality_gap_vs_performance_{coco_type}.pdf")
        plt.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/performance_vs_modality_gap")
    args = parser.parse_args()
    main(args)