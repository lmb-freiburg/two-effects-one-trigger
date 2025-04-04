import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tueplots import bundles
import torch
from tqdm import tqdm
from functools import partial
import os

from gap_precompute import compute_accuracy, recall_at_k, compute_l2m, compute_rmg, compute_l2i

datasets = ["coco_i2t", "coco_t2i", "imagenet"]
models = ["ViT-B-16__openai", "ViT-B-16-SigLIP__webli"]

def main(args):
    output_dir = Path(args.output_dir)

    for dataset in datasets:
        plt.rcParams.update(bundles.iclr2024(rel_width=1/3, nrows=2, ncols=1))
        fig, axes = plt.subplots(2, 1, sharey=False, sharex=True)
        for model_idx, model in enumerate(models):
            ax = axes[model_idx]

            if "coco" in dataset:
                folder_name_of_dataset = "coco"
            elif "imagenet" == dataset:
                folder_name_of_dataset = "imagenet"
            else:
                raise NotImplementedError
            text_features = torch.load(output_dir / f"{folder_name_of_dataset}_{model}_txt_feats.pth", map_location=args.device).float()
            img_features = torch.load(output_dir / f"{folder_name_of_dataset}_{model}_img_feats.pth", map_location=args.device).float()
            if dataset == "imagenet":
                labels = torch.load(output_dir / f"{folder_name_of_dataset}_{model}_obj_labels.pth", map_location=args.device)
            elif "coco" in dataset:
                image_to_text_map = torch.load(output_dir / f"{folder_name_of_dataset}_{model}_img2txt.pth", map_location=args.device)
                text_to_image_map = torch.load(output_dir / f"{folder_name_of_dataset}_{model}_txt2img.pth", map_location=args.device)
            else:
                raise NotImplementedError
            
            if args.modality_gap_metric == "RMG":
                if dataset == "imagenet":
                    compute_gap_distance = partial(compute_rmg, dataset=folder_name_of_dataset, labels=labels)
                elif "coco" in dataset:
                    compute_gap_distance = partial(compute_rmg, dataset=folder_name_of_dataset, text_to_image_map=text_to_image_map, image_to_text_map=image_to_text_map)
                else:
                    raise NotImplementedError
            elif args.modality_gap_metric == "L2M":
                compute_gap_distance = compute_l2m
            elif args.modality_gap_metric == "L2I":
                if dataset == "imagenet":
                    compute_gap_distance = partial(compute_l2i, dataset=folder_name_of_dataset, labels=labels)
                elif "coco" in dataset:
                    compute_gap_distance = partial(compute_l2i, dataset=folder_name_of_dataset, text_to_image_map=text_to_image_map)
            else:
                raise NotImplementedError

            img_features /= img_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            txt_features_mean = text_features.mean(dim=0)
            img_features_mean = img_features.mean(dim=0)

            diff_mean = img_features_mean - txt_features_mean
            abs_diff_mean = diff_mean.abs()

            ablated_dims = []
            performance = []
            if dataset == "imagenet":
                performance.append(compute_accuracy((img_features@text_features.T).argmax(dim=-1),labels))
            elif "coco" in dataset:
                k_vals=[1]
                t2i, i2t = recall_at_k(image_encodings=img_features, text_encodings=text_features, text_to_image_map=text_to_image_map, image_to_text_map=image_to_text_map, k_vals=k_vals, device=args.device)
                if "coco_i2t" == dataset:
                    performance.append(i2t[0]*100)
                elif "coco_t2i" == dataset:
                    performance.append(t2i[0]*100)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            gap_distance = [compute_gap_distance(img_features, text_features)]
            for dim in tqdm(abs_diff_mean.argsort(descending=True)[:args.n_ablated], leave=False):
                ablated_dims.append(dim)
                tmp_img_embeds = torch.index_select(img_features, dim=1, index=torch.tensor([m for m in range(text_features.shape[1]) if m not in ablated_dims]).to(args.device))
                tmp_txt_embeds = torch.index_select(text_features, dim=1, index=torch.tensor([m for m in range(text_features.shape[1]) if m not in ablated_dims]).to(args.device))
                tmp_img_embeds = torch.nn.functional.normalize(tmp_img_embeds, p=2, dim=-1)
                tmp_txt_embeds = torch.nn.functional.normalize(tmp_txt_embeds, p=2, dim=-1)

                if dataset == "imagenet":
                    performance.append(compute_accuracy((tmp_img_embeds@tmp_txt_embeds.T).argmax(dim=-1),labels))
                elif "coco" in dataset:
                    k_vals=[1]
                    t2i, i2t = recall_at_k(image_encodings=tmp_img_embeds, text_encodings=tmp_txt_embeds, text_to_image_map=text_to_image_map, image_to_text_map=image_to_text_map, k_vals=k_vals, device=args.device)
                    if "coco_i2t" == dataset:
                        performance.append(i2t[0]*100)
                    elif "coco_t2i" == dataset:
                        performance.append(t2i[0]*100)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                gap_distance.append(compute_gap_distance(tmp_img_embeds, tmp_txt_embeds))
            
            plot_df = pd.DataFrame(
                {
                    "Ablated dimensions": list(range(args.n_ablated+1)),
                    ("MS COCO R@1" if "coco" in dataset else "ImageNet\naccuracy"): performance,
                    args.modality_gap_metric: gap_distance,
                }
            )

            sns.lineplot(
                plot_df,
                x="Ablated dimensions",
                y="MS COCO R@1" if "coco" in dataset else "ImageNet\naccuracy",
                ax=ax,
                color=sns.color_palette()[0],
                zorder=20,
            )
            ax.set_ylabel("MS COCO R@1" if "coco" in dataset else "ImageNet\naccuracy", color=sns.color_palette()[0])

            ax2 = ax.twinx()
            sns.lineplot(
                plot_df,
                x="Ablated dimensions",
                y=args.modality_gap_metric,
                ax=ax2,
                color=sns.color_palette()[1],
                linestyle="dashed",
                zorder=5,
            )
            ax2.set_ylabel(args.modality_gap_metric, color=sns.color_palette()[1])

            if "openai" in model:
                ax.set_title("CLIP ViT-B/16")
            elif "SigLIP" in model:
                ax.set_title("SigLIP ViT-B/16")

            ax.tick_params(axis='y', colors=sns.color_palette()[0])
            ax.grid()

            ax2.tick_params(axis='y', colors=sns.color_palette()[1])
            ax2.set_xticks([0, args.n_ablated//2, args.n_ablated])

        sns.despine(fig, right=False)
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/ablated_dims_{dataset}_{args.modality_gap_metric}.pdf")
        plt.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/performance_vs_modality_gap")
    parser.add_argument("--modality_gap_metric", type=str, default="RMG", choices=["RMG", "L2M", "L2I"])
    parser.add_argument("--n_ablated", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)