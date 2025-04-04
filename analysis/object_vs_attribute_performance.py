import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from tueplots import bundles

from utils import read_vals_from_json_files, get_small_and_large_indices

datasets = ["imagenet", "coco", "mit-states", "ut-zappos"]
robust=True
correlation_measure = kendalltau

def main(args):
    for dataset in datasets:
        if dataset == "imagenet" or dataset == "coco":
            base_path = args.output_dir_gap
        elif dataset == "mit-states" or dataset == "ut-zappos":
            base_path = args.output_dir_obj
        else:
            raise NotImplementedError
        
        json_files = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if ".json" in f and dataset in f and f"{dataset}_" == f[:len(dataset)+1] and f"{dataset}_old" not in f])
        assert len(json_files) == 115, "Expected 115 models, got " + str(len(json_files))
        
        if dataset == "imagenet" or dataset == "mit-states" or dataset == "ut-zappos":
            keys = ["acc"]
        elif dataset == "coco":
            keys = ["img2txt.R@1", "txt2img.R@1"]
        else:
            raise NotImplementedError
        vals_dict = read_vals_from_json_files(json_files, keys)
        
        if dataset == "imagenet":
            imagenet = deepcopy(vals_dict["acc"])
        elif dataset == "mit-states":
            mit_states = deepcopy(vals_dict["acc"])
        elif dataset == "ut-zappos":
            ut_zappos = deepcopy(vals_dict["acc"])
        elif dataset == "coco":
            coco_both = {"i2t": deepcopy(vals_dict["img2txt.R@1"]), "t2i": deepcopy(vals_dict["txt2img.R@1"])}
        else:
            raise NotImplementedError

    model_names = [os.path.basename(f)[len("ut_zappos_"):-len(".json")] for f in json_files]    
    coco_i2t_performance = coco_both["i2t"]
    small_allowed_indices, large_allowed_indices = get_small_and_large_indices(model_names, coco_i2t_performance)
    assert len(small_allowed_indices) + len(large_allowed_indices) == 101

    for coco_type in ["i2t", "t2i"]:
        coco = coco_both[coco_type]
        plt.rcParams.update(bundles.cvpr2024(column="half", nrows=2, ncols=2))
        fig, axes = plt.subplots(2, 2, sharey=False, sharex=False)

        for idx, (obj_perf, attr_perf) in tqdm(enumerate([[coco, mit_states], [imagenet, mit_states], [coco, ut_zappos], [imagenet, ut_zappos]]), leave=False, total=4):
            ax = axes[idx//2, idx%2]

            if idx % 2 == 0:
                xlabel = "MS COCO\nR@1"
            else:
                xlabel = "ImageNet\naccuracy"

            if idx // 2 == 0:
                ylabel = "MIT-States\nattr accuracy"
            else:
                ylabel = "UT-Zappos\nattr accuracy"

            plot_df = pd.DataFrame({
                xlabel: np.array([d for i, d in enumerate(obj_perf) if i in large_allowed_indices]),
                ylabel: [p for i, p in enumerate(attr_perf) if i in large_allowed_indices],
            })
            sns.regplot(plot_df, x=xlabel, y=ylabel, ax=ax, scatter_kws={"s": 1, "zorder": 10,  "color": sns.color_palette()[0]}, line_kws={"color": sns.color_palette()[0]}, robust=robust, label="large")
            plot_df = pd.DataFrame({
                xlabel: np.array([d for i, d in enumerate(obj_perf) if i in small_allowed_indices]),
                ylabel: [p for i, p in enumerate(attr_perf) if i in small_allowed_indices],
            })
            sns.regplot(plot_df, x=xlabel, y=ylabel, ax=ax, scatter_kws={"s": 1, "zorder": 10,  "color": sns.color_palette()[1]}, line_kws={"color": sns.color_palette()[1]}, robust=robust, label="medium")
            ax.grid()
            sns.despine(ax=ax)

            if idx < 2:
                ax.set_xlabel("")

            if idx % 2 == 1:
                ax.set_ylabel("")

            text = r"Kendall's $\tau$: " + f"{round(correlation_measure(obj_perf, attr_perf).statistic*100, 1)}"
            
            if idx//2 == 0 and idx%2 == 0: # coco-mit-states
                ax.text(12, 23, text)
            elif idx//2 == 0 and idx%2 == 1: # imagenet-mit-states
                ax.text(16, 23, text)
            elif idx//2 == 1 and idx%2 == 0: # coco-ut-zappos
                ax.text(12, 40, text)
            elif idx//2 == 1 and idx%2 == 1: # imagenet-ut-zappos
                ax.text(16, 40, text)
            else:
                raise NotImplementedError

            if not (idx == 0):
                ax.legend([],[], frameon=False)
            else:
                ax.legend(title='Dataset size', labelspacing=0.2, handletextpad=0.2, borderpad=0.2, loc='lower right')


        os.makedirs("figures", exist_ok=True)
        if coco_type == "i2t":
            plt.savefig(f"figures/obj_vs_attr_performance.pdf")
        plt.savefig(f"figures/obj_vs_attr_performance_{coco_type}.pdf")
        plt.close()

        print("#"*5, coco_type, "#"*5)
        print("imagenet", "mit-states", correlation_measure(imagenet, mit_states))
        print("imagenet", "ut zappos", correlation_measure(imagenet, ut_zappos))
        print("coco", "mit-states", correlation_measure(coco, mit_states))
        print("coco", "ut-zappos", correlation_measure(coco, ut_zappos))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir_gap", type=str, default="results/performance_vs_modality_gap")
    parser.add_argument("--output_dir_obj", type=str, default="results/performance_vs_object_bias")
    args = parser.parse_args()
    main(args)