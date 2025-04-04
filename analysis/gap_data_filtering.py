import json
import os
import numpy as np
from scipy.stats import kendalltau
import json

from rich.table import Table, Column
from rich import print as rprint

from utils import read_vals_from_json_files, get_small_and_large_indices, dataset_to_size, exclude_model_name_list

datasets = ["coco", "imagenet"]

commonpools = [
    "commonpool_s_s",
    "commonpool_s_basic",
    "commonpool_s_text",
    "commonpool_s_image",
    "commonpool_s_laion",
    "commonpool_s_clip",
    "commonpool_m_s",
    "commonpool_m_basic",
    "commonpool_m_text",
    "commonpool_m_image",
    "commonpool_m_laion",
    "commonpool_m_clip",
    "commonpool_l_s",
    "commonpool_l_basic",
    "commonpool_l_text",
    "commonpool_l_image",
    "commonpool_l_laion",
    "commonpool_l_clip",
    "commonpool_xl_s",
    "commonpool_xl_laion",
    "commonpool_xl_clip",
]

def main(args):
    json_files = sorted([os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if ".json" in f and "coco" in f and f"coco_" == f[:len("coco")+1] and f"coco_old" not in f])
    assert len(json_files) == 115, "Expected 115 models, got " + str(len(json_files))
    coco_performance = read_vals_from_json_files(json_files, ["img2txt.R@1"])["img2txt.R@1"]
    model_names = [os.path.basename(f)[len("coco_"):-len(".json")] for f in json_files]    

    for didx, dataset in enumerate(datasets):
        if dataset == "coco":
            cols = [
                "Commonpool", 
                Column("Dataset size"), 
                Column("I2T"), 
                Column("T2I"), 
                Column("L2M"),
                Column("RMG")
            ]
        elif dataset == "imagenet":
            cols = [
                "Commonpool", 
                Column("Dataset size"), 
                Column("Performance"), 
                Column("L2M"),
                Column("RMG")
            ]
        else:
            raise NotImplementedError

        table = Table(*cols, title=dataset)


        json_files = sorted([os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if ".json" in f and dataset in f and f"{dataset}_" == f[:len(dataset)+1] and f"{dataset}_old" not in f])
        assert len(json_files) == 115, "Expected 115 models, got " + str(len(json_files))
        

        shared_keys = ["emb_dim", "model_size", "train_dataset", "l2_means", "rmg"]
        if dataset == "imagenet":
            keys = ["acc"]
        elif dataset == "coco":
            keys = ["img2txt.R@1", "txt2img.R@1"]
        else:
            raise NotImplementedError
        data = read_vals_from_json_files(json_files, shared_keys + keys)

        train_datasets = []
        for d in data["train_dataset"]:
            if "merged2b" in d:
                train_datasets.append("merged2b")
            elif "laion2b" in d:
                train_datasets.append("laion2b")
            elif "laion400m" in d:
                train_datasets.append("laion400m")
            elif "laion_aesthetic" in d:
                train_datasets.append("laion_aesthetic")
            else:
                train_datasets.append(d)

        dataset_sizes = []
        for d in data["train_dataset"]:
            tmp = [k if d == "v1" and "nllb" in d else k for k in dataset_to_size.keys() if k in d]
            if len(tmp) != 1:
                print(d)
            dataset_sizes.append(dataset_to_size[tmp[0]])

        model_sizes = data["model_size"]
        embed_dims = data["emb_dim"]
        train_datasets = data["train_dataset"]
        # dataset_sizes = data["dataset_sizes"]
        performance = [data["acc"]] if dataset == "imagenet" else [data["img2txt.R@1"], data["txt2img.R@1"]]
        # exclude_indices = data["exclude_indices"]
        l2m = data["l2_means"]
        rmg = data["rmg"]

        allowed_indices, _ = get_small_and_large_indices(model_names, coco_performance)

        for commonpool in commonpools:
            indices = [i for i, name in enumerate(model_names) if commonpool in name]
            assert len(indices) == 1
            idx = indices[0]
            # print(commonpool, dataset_sizes[idx], performance[idx], l2m[idx], rmg[idx])

            if dataset == "imagenet":
                table.add_row(commonpool, repr(dataset_sizes[idx]), repr(round(performance[0][idx], 3)), f"{round(l2m[idx], 3)}", f"{round(rmg[idx], 3)}")
            elif dataset == "coco":
                table.add_row(commonpool, repr(dataset_sizes[idx]), repr(round(performance[0][idx], 3)), repr(round(performance[1][idx], 3)), f"{round(l2m[idx], 3)}", f"{round(rmg[idx], 3)}")
            else:
                raise NotImplementedError

        rprint(table)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/performance_vs_modality_gap")
    args = parser.parse_args()

    main(args)