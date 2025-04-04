import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import os

from tueplots import bundles

datasets = ["imagenet", "coco"]
models = ["ViT-B-16__openai", "ViT-B-16-SigLIP__webli"]

def main(args):
    args.output_dir = Path(args.output_dir)

    for dataset in datasets:
        plt.rcParams.update(bundles.iclr2024(rel_width=1/3, nrows=2, ncols=1))
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)

        for model_idx, model in tqdm(enumerate(models), leave=False, total=len(models)):
            ax = axes[model_idx]
            
            img_features = torch.load(args.output_dir / f"{dataset}_{model}_img_feats.pth", map_location="cpu")
            img_features /= img_features.norm(dim=-1, keepdim=True)
            
            text_features = torch.load(args.output_dir / f"{dataset}_{model}_txt_feats.pth", map_location="cpu")
            text_features /= text_features.norm(dim=-1, keepdim=True)

            if args.dim_selection == "largest_mean_per_modality":
                dims = [
                    img_features.mean(dim=0).argmax().item(),
                    text_features.mean(dim=0).argmax().item(),
                ]
            elif args.dim_selection == "largest_mean_diff":
                diff = (img_features.mean(dim=0)-text_features.mean(dim=0)).abs()
                dims = diff.topk(2).indices.tolist()
            else:
                raise NotImplementedError

            dims = sorted(dims)
            text_features = torch.index_select(text_features, dim=1, index=torch.tensor(dims))
            img_features = torch.index_select(img_features, dim=1, index=torch.tensor(dims))

            plot_df = pd.DataFrame(
                {
                    "modality": ["Text" for _ in range(text_features.shape[0])] + ["Image" for _ in range(img_features.shape[0])],
                    f"Dim {dims[0]}": list(text_features[:, 0].numpy())+list(img_features[:, 0].numpy()),
                    f"Dim {dims[1]}": list(text_features[:, 1].numpy())+list(img_features[:, 1].numpy()),
                }
            )

            scatter = sns.scatterplot(
                plot_df, 
                x=f"Dim {dims[0]}", 
                y=f"Dim {dims[1]}",
                hue="modality",
                style="modality",
                ax=ax,
                s=1,
                zorder=10,
            )
            scatter.legend_.set_title(None)

            if model_idx == 1:
                ax.legend([],[], frameon=False)
            else:
                handles, labels = scatter.get_legend_handles_labels()
                for dot in handles:
                    dot._markersize *= 4
                ax.legend(handles=handles, labels=labels)
            ax.grid()

            if "openai" in model:
                ax.set_title("CLIP ViT-B/16")
            elif "SigLIP" in model:
                ax.set_title("SigLIP ViT-B/16")

            sns.despine(ax=ax)

        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/linear_pairs_{args.dim_selection}_{dataset}.png", dpi=600)
        plt.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/performance_vs_modality_gap")
    parser.add_argument("--dim_selection", type=str, default="largest_mean_per_modality", choices=["largest_mean_per_modality", "largest_mean_diff"])
    args = parser.parse_args()
    main(args)