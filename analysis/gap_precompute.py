import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functools import partial
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import open_clip
from tqdm import tqdm
import json
import torchvision
from typing import List
import math

from data_loaders import ImageNet, openai_imagenet_classes
from settings import IMAGENET_DIR, COCO_DATA_DIR, COCO_ANNOTATIONS_FILE

# Performance metrics
@torch.no_grad()
def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (torch.sum(preds==labels) / preds.size(0)).item()*100

@torch.no_grad()
def recall_at_k(image_encodings, text_encodings, text_to_image_map, image_to_text_map, k_vals: List[int], device):
    # Adopted code from https://github.com/openai/CLIP/issues/115#issuecomment-1493074399 
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    # text-to-image recall
    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text
    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    # dist_matrix = dist_matrix.cpu()
    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    text_to_image_recall = []
    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]
        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    # image-to-text recall
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image
    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    image_to_text_recall = []
    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]
        correct = torch.zeros((num_im,), dtype=torch.bool).to(device)
        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)
        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)

    return text_to_image_recall, image_to_text_recall

# Modality gap metrics
@torch.no_grad()
def compute_l2m(image_features: torch.Tensor, text_features: torch.Tensor) -> float:
    return (image_features.mean(axis=0) - text_features.mean(axis=0)).norm().item()

@torch.no_grad()
def compute_l2i(image_features: torch.Tensor, text_features: torch.Tensor, dataset, labels=None, text_to_image_map=None) -> float:
    if dataset == "imagenet":
        image_features_per_text = image_features
        text_feature_per_image = torch.stack([text_features[l] for l in labels], dim=0)
    elif dataset == "coco":
        image_features_per_text = torch.stack([image_features[i] for i in text_to_image_map], dim=0)
        text_feature_per_image = text_features
    else:
        raise NotImplementedError
    return (image_features_per_text - text_feature_per_image).norm(dim=-1).mean().item()

@torch.no_grad()
def compute_rmg(image_features, text_features, dataset, labels=None, text_to_image_map=None, image_to_text_map=None):
    if dataset == "imagenet":
        text_feature_per_image = torch.stack([text_features[l] for l in labels], dim=0)
        labels_to_idx_map = labels[None,:] != labels[:,None]
    elif dataset == "coco":
        image_features_original = image_features.clone()
        image_features = torch.stack([image_features[l] for l in text_to_image_map], dim=0)
        text_feature_per_image = text_features
        labels_to_idx_map = torch.ones((text_features.size(0),text_features.size(0))).bool()
        for txt_idcs in image_to_text_map:
            for i,j in [(x.item(), y.item()) for x in txt_idcs for y in txt_idcs]:
                labels_to_idx_map[i,j] = False
    else:
        raise NotImplementedError
    
    
    image_features_matching = torch.sum(image_features*text_feature_per_image, dim=1).mean()
    image_features_matching = 1-(image_features_matching+1)/2 # [0, 1] & flip
    image_features_matching = torch.where(image_features_matching > 0, image_features_matching, torch.ones_like(image_features_matching)*1e-3) # [1e-3, 1]

    if dataset == "coco":
        i_x_i = image_features_original @ image_features_original.T
    else:
        i_x_i = image_features @ image_features.T
    i_x_i.fill_diagonal_(0)
    mean_img_similarity = i_x_i.sum() / (math.prod(i_x_i.shape)-i_x_i.shape[0]) # [-1, 1]
    mean_img_similarity = 1-(mean_img_similarity+1)/2 # [0,1] & flip

    t_x_t = text_features @ text_features.T
    t_x_t.fill_diagonal_(0)
    mean_txt_similarity = t_x_t.sum() / (math.prod(t_x_t.shape)-t_x_t.shape[0]) # [-1, 1]
    mean_txt_similarity = 1-(mean_txt_similarity+1)/2 # [0,1] & flip

    normalizer = image_features_matching.mean() + (mean_img_similarity.mean() + mean_txt_similarity.mean()) / 2
    dist = image_features_matching.mean().item() / normalizer.item()

    return dist

@torch.no_grad()
def encode_clip(model, loader, tokenizer, prompts=None):
    device = next(model.parameters()).device

    with torch.amp.autocast(device_type=device.type):
        all_image_features, labels, all_text_features = [], [], []
        image_to_text_map, text_to_image_map = [], []
        captions_per_image = []
        text_index, image_index = 0, 0
        for batch in tqdm(loader, leave=False, total=len(loader)):
            if len(batch) == 2:
                inputs, label = batch
            else:
                raise NotImplementedError
            if isinstance(inputs, tuple):
                inputs = torch.stack(inputs, dim=0)
            
            image_features = model.encode_image(inputs.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features.cpu())

            if prompts is None: # coco
                text = label # shape B x 5 x 77
                batch_size, captions_per_image, _ = text.shape

                # Update text_to_image_map and image_to_text_map for this batch
                for i in range(batch_size):
                    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                    text_indices = list(range(text_index, text_index + captions_per_image))
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image

                    # Each of the next captions_per_image text captions correspond to the same image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1

                # B x 5 x 77 -> (B*5) x 77
                text = torch.flatten(text, start_dim=0, end_dim=1)
                try:
                    text_features = model.encode_text(text.to(device))
                except Exception:
                    text_features = []
                    for t in text:
                        text_features.append(model.encode_text(t.unsqueeze(0).to(device)).cpu())
                    text_features = torch.cat(text_features, dim=0)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                all_text_features.append(text_features.cpu())
            else: # imagenet
                labels.append(label)
        
        all_image_features = torch.cat(all_image_features, dim=0)

        if prompts is not None: # imagenet
            labels = torch.cat(labels, dim=0)
            text = tokenizer(prompts).to(device)
            try:
                text_features = model.encode_text(text.to(device)).cpu()
            except Exception:
                text_features = []
                for t in text:
                    text_features.append(model.encode_text(t.unsqueeze(0)).cpu())
                text_features = torch.cat(text_features, dim=0)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return all_image_features, text_features, labels
        else:
            text_features = torch.cat(all_text_features, dim=0)
            text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
            image_to_text_map = torch.LongTensor(image_to_text_map).to(device)
            return all_image_features, text_features, text_to_image_map, image_to_text_map

def main(args):
    args.output_dir = Path(args.output_dir) / "performance_vs_modality_gap"
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name, pretrain_dataset = args.model.split("__")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain_dataset, cache_dir=args.cache_dir)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(args.device)
    model.eval()
    tokenizer = partial(tokenizer, context_length=model.context_length)

    datasets = ["imagenet", "coco"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        print(5*"#", dataset, 5*"#")
        if dataset == "imagenet":
            data = ImageNet(IMAGENET_DIR, transform=preprocess)
            prompts = [f"A photo of a {obj}." for obj in openai_imagenet_classes]
            loader = DataLoader(
                dataset=data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=None,
            )
        elif dataset == "coco":
            def prepend(text: list) -> list:
                return [f"a photo of {t[0].lower()}{t[1:]}" for t in text]
            data = torchvision.datasets.CocoCaptions(root=COCO_DATA_DIR, annFile=COCO_ANNOTATIONS_FILE, transform=preprocess, target_transform=lambda texts: tokenizer(prepend(texts[:5])))
            loader = DataLoader(
                dataset=data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
            )
            prompts = None
        else:
            raise NotImplementedError
        
        dataset_model_str = f"{dataset}_{args.model}"

        if not args.regenerate and os.path.isfile(args.output_dir / f"{dataset_model_str}.json"):
            if dataset == datasets[-1]:
                return
            continue
        if not args.regenerate and os.path.isfile(args.output_dir / f"{dataset_model_str}_img_feats.pth") and os.path.isfile(args.output_dir / f"{dataset_model_str}_txt_feats.pth"):
            image_features = torch.load(args.output_dir / f"{dataset_model_str}_img_feats.pth", map_location="cpu")
            text_features = torch.load(args.output_dir / f"{dataset_model_str}_txt_feats.pth", map_location="cpu")
            if os.path.isfile(args.output_dir / f"{dataset_model_str}_obj_labels.pth"):
                obj_labels = torch.load(args.output_dir / f"{dataset_model_str}_obj_labels.pth", map_location="cpu")
            elif dataset == "coco":
                obj_labels = None
            if os.path.isfile(args.output_dir / f"{dataset_model_str}_txt2img.pth") and os.path.isfile(args.output_dir / f"{dataset_model_str}_img2txt.pth"):
                text_to_image_map = torch.load(args.output_dir / f"{dataset_model_str}_txt2img.pth", map_location="cpu")
                image_to_text_map = torch.load(args.output_dir / f"{dataset_model_str}_img2txt.pth", map_location="cpu")
            if os.path.isfile(args.output_dir / f"{dataset_model_str}_captions_per_img.pth"):
                captions_per_image = torch.load(args.output_dir / f"{dataset_model_str}_captions_per_img.pth", map_location="cpu")
        else:
            if prompts is None and dataset == "coco":
                image_features, text_features, text_to_image_map, image_to_text_map = encode_clip(model, loader, tokenizer=tokenizer, prompts=None)
                if args.save:
                    torch.save(text_to_image_map, args.output_dir / f"{dataset_model_str}_txt2img.pth")
                    torch.save(image_to_text_map, args.output_dir / f"{dataset_model_str}_img2txt.pth")
            else:
                image_features, text_features, obj_labels = encode_clip(model, loader, tokenizer=tokenizer, prompts=prompts)
                if args.save:
                    torch.save(obj_labels, args.output_dir / f"{dataset_model_str}_obj_labels.pth")
            if args.save:
                torch.save(image_features, args.output_dir / f"{dataset_model_str}_img_feats.pth")
                torch.save(text_features, args.output_dir / f"{dataset_model_str}_txt_feats.pth")

        if "imagenet" == dataset:
            labels = obj_labels
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_features = image_features.to(args.device).float()
        text_features = text_features.to(args.device).float()

        # evaluation
        res = {}

        res["nof_images"] = image_features.size(0)
        res["nof_texts"] = text_features.size(0)


        # modality gap metrics
        res["l2_means"] = compute_l2m(image_features, text_features)
        # l2-instancewise -> l2 distance for matched instances
        # rmg -> relative modality gap
        if dataset == "imagenet":
            # res["l2_instancewise"] = compute_l2i(image_features=image_features, text_features=torch.stack([text_features[l] for l in labels], dim=0))
            res["l2_instancewise"] = compute_l2i(image_features=image_features, text_features=text_features, dataset=dataset, labels=labels)
            res["rmg"] = compute_rmg(image_features=image_features.cpu(), text_features=text_features.cpu(), dataset=dataset, labels=labels.cpu())
        elif dataset == "coco":
            # res["l2_instancewise"] = compute_l2i(image_features=torch.stack([image_features[i] for i in text_to_image_map], dim=0), text_features=text_features)
            res["l2_instancewise"] = compute_l2i(image_features=image_features, text_features=text_features, dataset=dataset, text_to_image_map=text_to_image_map)
            res["rmg"] = compute_rmg(image_features=image_features.cpu(), text_features=text_features.cpu(), dataset=dataset, text_to_image_map=text_to_image_map, image_to_text_map=image_to_text_map)
        else:
            raise NotImplementedError
        
        
        # performance metrics
        if dataset == "imagenet":
            preds = torch.argmax(image_features @ text_features.T, dim=1).cpu()
            res["acc"] = compute_accuracy(preds, labels)
        elif dataset == "coco":
            k_vals=[1, 5, 10, 50]
            t2i, i2t = recall_at_k(image_encodings=image_features, text_encodings=text_features, text_to_image_map=text_to_image_map.to(args.device), image_to_text_map=image_to_text_map.to(args.device), k_vals=k_vals, device=args.device)
            res["txt2img"] = {f"R@{k}":100*x for k,x in zip(k_vals, t2i)}
            res["img2txt"] = {f"R@{k}":100*x for k,x in zip(k_vals, i2t)}
        else:
            raise NotImplementedError
        
        # further model information
        res["emb_dim"] = image_features.size(1)
        res["model_size"] = sum(p.numel() for p in model.parameters())
        res["train_dataset"] = pretrain_dataset
        
        with open(args.output_dir / f"{dataset_model_str}.json", mode="w", encoding="utf-8") as f:
            json.dump(res, f, indent=4)

        print(res)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Performance vs Modality Gap")
    parser.add_argument(
        "--model",
        type=str,
        default="RN50__openai",
        help="CLIP model to evaluate",
        choices=[f"{model_name}__{pretrain_dataset}" for model_name, pretrain_dataset in open_clip.list_pretrained()],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="dataset to evaluate on",
        choices=["all", "imagenet", "coco"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for evaluation",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save the computed features (faster if re-running the script)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="regenerate the features even if they are already saved",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/clip",
        help="cache directory for CLIP models",
    )
    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="run in debug mode",
    )
    args = parser.parse_args()
    main(args)