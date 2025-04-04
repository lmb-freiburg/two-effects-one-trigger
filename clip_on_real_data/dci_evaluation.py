import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
import open_clip
from functools import partial
import torchvision

from analysis.gap_precompute import encode_clip, compute_accuracy, recall_at_k, compute_rmg, compute_l2m
from data_loaders import ImageNet, openai_imagenet_classes
from settings import IMAGENET_DIR, COCO_DATA_DIR, COCO_ANNOTATIONS_FILE

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--models", nargs="+", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--cache_dir", type=str, default="~/.cache/clip")
args = parser.parse_args()

datasets = ["imagenet", "coco"]

for model_name in args.models:
    print("#"*5, model_name, "#"*5)

    if "webli" in model_name:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli', cache_dir=args.cache_dir)
        tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    elif "openai" in model_name:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/16', pretrained='openai', cache_dir=args.cache_dir)
        tokenizer = open_clip.get_tokenizer("ViT-B/16")
    model = model.eval().to(args.device)
    tokenizer = partial(tokenizer, context_length=model.context_length)
    
    if model_name is not None:
        ckpt = f"{model_name}/checkpoints/epoch_10.pt"
        ckpt = torch.load(ckpt, map_location=args.device)
        state_dict = {k.replace("module.", ""):v for k,v in ckpt["state_dict"].items()}
        model.load_state_dict(state_dict)

    for dataset in datasets:
        print(3*"#", dataset, 3*"#")
        if dataset == "imagenet":
            data = ImageNet(IMAGENET_DIR, transform=preprocess)
            prompts = [f"A photo of a {obj}." for obj in openai_imagenet_classes]
            loader = DataLoader(
                dataset=data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=20,
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
                num_workers=20,
            )
            prompts = None
        else:
            raise NotImplementedError

        if prompts is None and dataset == "coco":
            image_features, text_features, text_to_image_map, image_to_text_map = encode_clip(model, loader, tokenizer=tokenizer, prompts=None)
        else:
            image_features, text_features, obj_labels = encode_clip(model, loader, tokenizer=tokenizer, prompts=prompts)

        if "imagenet" == dataset:
            labels = obj_labels
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.to(args.device).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.to(args.device).float()

        res = {}

        res["l2_means"] = compute_l2m(image_features, text_features)
        if dataset == "imagenet":
            res["rmg"] = compute_rmg(image_features, text_features, dataset, labels=labels)
        elif dataset == "coco":
            res["rmg"] = compute_rmg(image_features, text_features, dataset, text_to_image_map=text_to_image_map, image_to_text_map=image_to_text_map)
        else:
            pass
        
        # downstream performance metrics
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

        print(res)