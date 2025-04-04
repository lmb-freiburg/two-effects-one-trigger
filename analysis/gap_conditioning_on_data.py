from collections import Counter
from scipy.stats import kendalltau
import math
import numpy as np
import os

from utils import read_vals_from_json_files, dataset_to_size

correlation_metric = kendalltau

def main(args):
    json_files = sorted([os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if ".json" in f and args.dataset in f and f"{args.dataset}_" == f[:len(args.dataset)+1] and f"{args.dataset}_old" not in f])
    assert len(json_files) == 115, "Expected 115 models, got " + str(len(json_files))
    
    if args.gap_metric == "L2M":
        dist_metric_key = "l2_means"
    elif args.gap_metric == "RMG":
        dist_metric_key = "rmg"
    else:
        raise NotImplementedError
    shared_keys = ["emb_dim", "model_size", "train_dataset", dist_metric_key]
    if args.dataset == "imagenet" or args.dataset == "mit-states" or args.dataset == "ut-zappos":
        keys = ["acc"]
    elif args.dataset == "coco":
        keys = ["img2txt.R@1", "txt2img.R@1"]
    else:
        raise NotImplementedError
    data = read_vals_from_json_files(json_files, shared_keys + keys)

    train_datasets = []
    for dataset in data["train_dataset"]:
        if "merged2b" in dataset:
            train_datasets.append("merged2b")
        elif "laion2b" in dataset:
            train_datasets.append("laion2b")
        elif "laion400m" in dataset:
            train_datasets.append("laion400m")
        elif "laion_aesthetic" in dataset:
            train_datasets.append("laion_aesthetic")
        else:
            train_datasets.append(dataset)

    all_dataset_sizes = []
    for d in data["train_dataset"]:
        tmp = [k if d == "v1" and "nllb" in d else k for k in dataset_to_size.keys() if k in d]
        if len(tmp) != 1:
            print(d)
        all_dataset_sizes.append(dataset_to_size[tmp[0]])

    all_model_names = [os.path.basename(f)[:-5] for f in json_files]

    print("Dataset size vs. gap distance", correlation_metric(all_dataset_sizes, data[dist_metric_key]))

    occurrences = Counter(train_datasets)
    top_k = occurrences.most_common(args.k)

    for search_string, count in top_k:
        
        indices = [i for i, string in enumerate(train_datasets) if string == search_string]
        assert len(indices) == count

        model_names = np.array([all_model_names[i] for i in indices])
        model_sizes = np.array([data["model_size"][i] for i in indices])
        emb_dims = np.array([data["emb_dim"][i] for i in indices])
        dataset_sizes = np.array([all_dataset_sizes[i] for i in indices])
        if args.dataset == "coco":
            performance_i2t = np.array([data["img2txt.R@1"][i] for i in indices])
            performance_t2i = np.array([data["txt2img.R@1"][i] for i in indices])
            performances = [performance_i2t, performance_t2i]
        elif args.dataset == "imagenet":
            performances = [np.array([data["acc"][i] for i in indices])]
        gap_distance = np.array([data[dist_metric_key][i] for i in indices])

        if "laion2b" == search_string:
            use_indices = [0, 2, 3, 4, 5, 6, 8, 9, 11, 14, 16, 17, 20, 21]
        elif "laion400m" == search_string:
            use_indices = [0, 2, 4, 6, 8, 10, 11]
            # use_indices = [0, 4, 6, 8, 10, 11]
        elif "openai" == search_string:
            use_indices = [0, 2, 4, 5, 6, 7, 8, 10, 11]
            # use_indices = [7, 8, 10, 11] # only ViT
            # use_indices = [7, 8, 10, 11] # only conv
        elif "webli" == search_string:
            use_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            # use_indices = [4, 5, 8]
        else:
            raise NotImplementedError
        
        print("#"*5, search_string, count, math.log10(dataset_sizes[0]), "#"*5)
        
        # print("Removed models:", model_names[[i for i in range(len(model_names)) if i not in use_indices]])
        # print("Used models:", model_names[use_indices])

        # print(len(use_indices))

        model_sizes = model_sizes[use_indices]
        emb_dims = emb_dims[use_indices]
        dataset_sizes = dataset_sizes[use_indices]
        performances = [p[use_indices] for p in performances]
        gap_distance = gap_distance[use_indices]

        for i, performance in enumerate(performances):
            if args.dataset == "coco":
                print("#"*5, "COCO", "i2t" if i==0 else "t2i", "#"*5)
            elif args.dataset == "imagenet":
                print("#"*5, "Imagenet", "#"*5)
            # print("Model sizes vs. performance", correlation_metric(model_sizes, performance))
            # print("Emb dims vs. performance", correlation_metric(emb_dims, performance))
            # print("Model sizes vs. gap distance", correlation_metric(model_sizes, gap_distance))
            # print("Emb dims vs. gap distance", correlation_metric(emb_dims, gap_distance))
            print("Performance vs. gap distance", correlation_metric(performance, gap_distance))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/performance_vs_modality_gap")
    parser.add_argument("--dataset", type=str, default="coco", choices=["imagenet", "coco"])
    parser.add_argument("--gap_metric", type=str, default="RMG", choices=["RMG", "L2M"])
    parser.add_argument("--k", type=int, default=4, help="Number of models that are at least needed to be trained on the dataset")
    args = parser.parse_args()
    main(args)