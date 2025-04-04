import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tueplots import bundles
import torch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os

datasets = ["imagenet", "coco"]
model = ["ViT-B-16__openai", "ViT-B-16-SigLIP__webli"]

def main(args):
    assert os.path.exists(args.output_dir), f"Path {args.output_dir} does not exist"
    output_dir = Path(args.output_dir)

    for dataset in datasets:
        dimensions_sorted = []
        diffs = []
        models = []

        dimensions_sorted_inset = []
        diffs_inset = []
        models_inset = []
        plt.rcParams.update(bundles.iclr2024(rel_width=3/4, nrows=1, ncols=1))
        fig, ax = plt.subplots(1, 1, sharey=False, sharex=False)
        axins = ax.inset_axes([0.3,0.2,0.7,0.6]) # Create zoomed-in inset
        for model_idx, model in enumerate(models):
            txt_features = torch.load(output_dir / f"{dataset}_{model}_txt_feats.pth", map_location="cpu").float()
            img_features = torch.load(output_dir / f"{dataset}_{model}_img_feats.pth", map_location="cpu").float()

            txt_features /= txt_features.norm(dim=-1, keepdim=True)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            txt_features_mean = txt_features.mean(dim=0)
            img_features_mean = img_features.mean(dim=0)

            diff_mean = img_features_mean - txt_features_mean
            abs_diff_mean = diff_mean.abs()

            dimensions_sorted += list(range(txt_features.shape[1]))
            diffs += list(abs_diff_mean.sort(descending=True).values.numpy())
            d = "ImageNet" if "imagenet" == dataset else "MS COCO"
            m = "CLIP ViT-B/16" if "openai" in model else "SigLIP ViT-B/16"
            models += [f"{m}" for _ in range(txt_features.shape[1])]

            dimensions_sorted_inset += list(range(txt_features.shape[1]))[:10]
            diffs_inset += list(abs_diff_mean.sort(descending=True).values.numpy())[:10]
            models_inset += [f"{m}" for _ in range(txt_features.shape[1])][:10]

            indices = abs_diff_mean.argsort(descending=True)[:5]
            print("#"*5, f"{m} on {d}", "#"*5)
            print("Dims", indices)
            print("Abs diff", abs_diff_mean[indices])
            print("Mean txt", txt_features_mean[indices])
            print("Max txt dims", txt_features_mean.abs().argsort(descending=True)[:5])
            print("Max txt vals", txt_features_mean.abs().sort(descending=True).values[:5])
            print("Max txt percentage", txt_features_mean.abs().sort(descending=True).values[:5]/txt_features_mean.abs().sort(descending=True).values.sum())
            print("Var txt", txt_features.var(dim=0)[indices])
            print("Mean img", img_features_mean[indices])
            print("Max img dims", img_features_mean.abs().argsort(descending=True)[:5])
            print("Max img vals", img_features_mean.abs().sort(descending=True).values[:5])
            print("Max img vals percentage", img_features_mean.abs().sort(descending=True).values[:5]/img_features_mean.abs().sort(descending=True).values.sum())
            print("Var img", img_features.var(dim=0)[indices])

            print("Var img total", img_features.var(dim=0).var())
            print("Var txt total", txt_features.var(dim=0).var())

            print("Mean img total", img_features.mean(dim=0).var())
            print("Mean txt total", txt_features.mean(dim=0).var())

            print("Mean cossim img", torch.triu(img_features @ img_features.T, diagonal=1).mean())
            print("Mean cossim txt", torch.triu(txt_features @ txt_features.T, diagonal=1).mean())

        xlabel = "Embedding dimensions (sorted)"
        plot_df = pd.DataFrame({
            xlabel: dimensions_sorted,
            "Abs. difference of means": diffs,
            "Model": models,
        })      
        sns.lineplot(
            plot_df,
            x=xlabel,
            y="Abs. difference of means",
            ax=ax,
            hue="Model",
            style="Model",
            zorder=20,
        )

        plot_df = pd.DataFrame({
            xlabel: dimensions_sorted_inset,
            "Abs. difference of means": diffs_inset,
            "Model": models_inset,
        })
        inset = sns.lineplot(
            plot_df,
            x=xlabel,
            y="Abs. difference of means",
            hue="Model",
            style="Model",
            zorder=20,
            ax=axins,
        )
        inset.legend_.remove()
        axins.set_xlim(0,9)
        axins.grid()

        # Hide ticks in the zoomed-in area
        axins.set_xticks([0,2,4,6,8])
        axins.set_ylabel("")
        axins.set_xlabel("")

        # Mark the region corresponding to the zoomed-in area in the main plot
        mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.0", zorder=19)

        sns.despine(fig, right=True)
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/mean_differences_{dataset}.pdf")
        plt.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/performance_vs_modality_gap")
    args = parser.parse_args()
    main(args)