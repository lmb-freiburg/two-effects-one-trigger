import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import open_clip
from pathlib import Path
from functools import partial
from tqdm import tqdm, trange
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

from data_loaders import CompositionDataset
from settings import MIT_STATES_DIR, UT_ZAPPOS_DIR

def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (torch.sum(preds==labels) / preds.size(0)).item()*100

@torch.no_grad()
def encode(model, loader, tokenizer, prompts):
    device = next(model.parameters()).device
    with torch.amp.autocast(device_type=device.type):
        all_image_features, attr_labels, obj_labels = [], [], []
        for inputs, attr_label, obj_label in tqdm(loader, leave=False, total=len(loader)):
            image_features = model.encode_image(inputs.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features.cpu())
            attr_labels.append(attr_label)
            obj_labels.append(obj_label)

        all_image_features = torch.cat(all_image_features, dim=0)
        attr_labels = torch.cat(attr_labels, dim=0).long()
        obj_labels = torch.cat(obj_labels, dim=0).long()
        text = tokenizer(prompts).to(device)
        try:
            text_features = model.encode_text(text).cpu()
        except Exception:
            text_features = []
            for t in text:
                text_features.append(model.encode_text(t.unsqueeze(0)).cpu())
            text_features = torch.cat(text_features, dim=0)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return all_image_features, text_features, attr_labels, obj_labels

@torch.no_grad()
def compute_ideal_words(model, loader, tokenizer):
    device = next(model.parameters()).device
    with torch.amp.autocast(device_type=device.type):
        all_ideal_words = []
        for attr in tqdm(loader.dataset.attr2idx.keys(), leave=False):
            prompts = [f"an image of a {attr.lower().replace('.', ' ')} {ut_zappos_obj_preprocessing(obj)}." for obj in loader.dataset.obj2idx.keys()]
            text = tokenizer(prompts).to(device)
            text_features = model.encode_text(text).cpu()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            u_a = text_features.mean(dim=0)
            all_ideal_words.append(u_a)
        for obj in tqdm(loader.dataset.obj2idx.keys(), leave=False):
            prompts = [f"an image of a {attr.lower().replace('.', ' ')} {ut_zappos_obj_preprocessing(obj)}." for attr in loader.dataset.attr2idx.keys()]
            text = tokenizer(prompts).to(device)
            text_features = model.encode_text(text).cpu()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            u_o = text_features.mean(dim=0)
            all_ideal_words.append(u_o) 

        all_ideal_words = torch.stack(all_ideal_words, dim=0)

        try:
            u0 = torch.zeros(model.text_projection.shape[1]).cpu()
        except:
            u0 = torch.zeros(model.text.text_projection.out_features).cpu()
        for attr in tqdm(loader.dataset.attr2idx.keys(), leave=False):
            prompts = [f"an image of a {attr.lower().replace('.', ' ')} {ut_zappos_obj_preprocessing(obj)}." for obj in loader.dataset.obj2idx.keys()]
            text = tokenizer(prompts).to(device)
            text_features = model.encode_text(text).cpu()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            u0 += text_features.sum(dim=0)
        u0 /= len(loader.dataset.attr2idx.keys())*len(loader.dataset.obj2idx.keys())
    
    return all_ideal_words, u0

@torch.no_grad()
def compute_ideal_images(image_features, obj_labels, attr_labels):
    all_ideal_images = []

    normalized_image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    for attr_id in trange(attr_labels.max().item()+1, leave=False):
        index_map = attr_labels == attr_id
        all_image_features_with_attr = normalized_image_features[index_map].cpu()
        all_image_features_with_attr /= all_image_features_with_attr.norm(dim=-1, keepdim=True)
        u_a = all_image_features_with_attr.mean(dim=0)
        all_ideal_images.append(u_a)

    for obj_id in trange(obj_labels.max().item()+1, leave=False):
        index_map = obj_labels == obj_id
        all_image_features_with_obj = normalized_image_features[index_map].cpu()
        all_image_features_with_obj /= all_image_features_with_obj.norm(dim=-1, keepdim=True)
        u_o = all_image_features_with_obj.mean(dim=0)
        all_ideal_images.append(u_o)

    all_ideal_images = torch.stack(all_ideal_images, dim=0)

    u0 = torch.mean(normalized_image_features, dim=0).cpu()

    return all_ideal_images, u0

def ut_zappos_obj_preprocessing(obj):
    obj = obj.lower()
    words = obj.split(".")
    if len(words) > 1:
        if words[0] == words[-1]:
            obj = " ".join(words[1:])
        else:
            obj = " ".join(words[1:] + [words[0]])
    return obj 

def main(args):
    args.output_dir = Path(args.output_dir) / "ideal_words_exps"
    args.output_dir.mkdir(exist_ok=True)

    if args.dataset == "mit-states":
        root = MIT_STATES_DIR
    elif args.dataset == "ut-zappos":
        root = UT_ZAPPOS_DIR
    else:
        raise NotImplementedError

    model_name, pretrain_dataset = args.model.split("__")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain_dataset, cache_dir=args.cache_dir)
    model = model.to(args.device)
    tokenizer = open_clip.get_tokenizer(model_name)
    tokenizer = partial(tokenizer, context_length=model.context_length)

    data = CompositionDataset(root=root, split="test", transform=preprocess, also_return_obj_label=True)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=20,
        pin_memory=False,
    )
    attr_prompts = [f"an image of a {attr.lower().replace('.', ' ')} object" for attr in data.attr2idx.keys()]
    if args.dataset == "mit-states":
        obj_prompts = [f"image of a {obj}" for obj in data.obj2idx.keys()]
    elif args.dataset == "ut-zappos":
        obj_prompts = [f"image of a {ut_zappos_obj_preprocessing(obj)}" for obj in data.obj2idx.keys()]
    else:
        raise NotImplementedError
    prompts = attr_prompts + obj_prompts
    n_attr = len(data.attr2idx.keys())

    dataset_model_str = f"{args.dataset}_{args.model}"

    # compute image and text features on test set
    if not args.regenerate and os.path.isfile(args.output_dir / f"{dataset_model_str}_image_features.pth"):
        image_features = torch.load(args.output_dir / f"{dataset_model_str}_image_features.pth")
        text_features = torch.load(args.output_dir / f"{dataset_model_str}_text_features.pth")
        attr_labels = torch.load(args.output_dir / f"{dataset_model_str}_attr_labels.pth")
        obj_labels = torch.load(args.output_dir / f"{dataset_model_str}_obj_labels.pth")
    else:
        print("Compute image and text features on test set...")
        image_features, text_features, attr_labels, obj_labels = encode(model, loader, tokenizer, prompts)
        if args.save:
            torch.save(image_features, args.output_dir / f"{dataset_model_str}_image_features.pth")
            torch.save(text_features, args.output_dir / f"{dataset_model_str}_text_features.pth")
            torch.save(attr_labels, args.output_dir / f"{dataset_model_str}_attr_labels.pth")
            torch.save(obj_labels, args.output_dir / f"{dataset_model_str}_obj_labels.pth")

    # compute ideal words
    if not args.regenerate and os.path.isfile(args.output_dir / f"{dataset_model_str}_ideal_words.pth"):
        all_ideal_words = torch.load(args.output_dir / f"{dataset_model_str}_ideal_words.pth")
        u0_text = torch.load(args.output_dir / f"{dataset_model_str}_ideal_words_u0.pth")
        all_ideal_words_minus_u0 = all_ideal_words - u0_text
    else:
        print("Compute ideal words...")
        all_ideal_words, u0_text = compute_ideal_words(model, loader, tokenizer)
        torch.save(all_ideal_words, args.output_dir / f"{dataset_model_str}_ideal_words.pth")
        torch.save(u0_text, args.output_dir / f"{dataset_model_str}_ideal_words_u0.pth")
        all_ideal_words_minus_u0 = all_ideal_words - u0_text
    all_ideal_words_minus_u0 = all_ideal_words_minus_u0.to(torch.float16)
    
    # precompute train image features and save object and attribute labels
    if not args.regenerate and os.path.isfile(args.output_dir / f"{dataset_model_str}_train_image_features.pth"):
        train_image_features = torch.load(args.output_dir / f"{dataset_model_str}_train_image_features.pth")
        train_attr_labels = torch.load(args.output_dir / f"{dataset_model_str}_train_attr_labels.pth")
        train_obj_labels = torch.load(args.output_dir / f"{dataset_model_str}_train_obj_labels.pth")
    else:
        print("Compute train images...")
        train_data = CompositionDataset(root=root, split="train", transform=preprocess, also_return_obj_label=True)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=20,
            pin_memory=False,
        )
        train_image_features, _, train_attr_labels, train_obj_labels = encode(model, train_loader, tokenizer, prompts)
        if args.save:
            torch.save(train_image_features, args.output_dir / f"{dataset_model_str}_train_image_features.pth")
            torch.save(train_attr_labels, args.output_dir / f"{dataset_model_str}_train_attr_labels.pth")
            torch.save(train_obj_labels, args.output_dir / f"{dataset_model_str}_train_obj_labels.pth")
    
    # compute ideal images
    if not args.regenerate and os.path.isfile(args.output_dir / f"{dataset_model_str}_ideal_images.pth"):
        all_ideal_images = torch.load(args.output_dir / f"{dataset_model_str}_ideal_images.pth")
        u0_img = torch.load(args.output_dir / f"{dataset_model_str}_ideal_images_u0.pth")
        all_ideal_images_minus_u0 = all_ideal_images - u0_img
    else:
        print("Compute ideal images...")
        all_ideal_images, u0_img = compute_ideal_images(train_image_features, train_obj_labels, train_attr_labels)
        torch.save(all_ideal_images, args.output_dir / f"{dataset_model_str}_ideal_images.pth")
        torch.save(u0_img, args.output_dir / f"{dataset_model_str}_ideal_images_u0.pth")
        all_ideal_images_minus_u0 = all_ideal_images - u0_img
    
    tmp_data = {
        "Category": ["attribute"  for _ in range(n_attr)] + ["object" for _ in range(all_ideal_images.size(0)-n_attr)] + ["attribute" for _ in range(n_attr)] + ["object" for _ in range(all_ideal_images.size(0)-n_attr)],
        "modality": ["image" for _ in range(all_ideal_images.size(0))] + ["text" for _ in range(all_ideal_words.size(0))],
        "Ideal Word Magnitude": list(all_ideal_images.norm(dim=-1).cpu().numpy()) + list(all_ideal_words.norm(dim=-1).cpu().numpy()),
    }
    df = pd.DataFrame(tmp_data)
    ax = sns.violinplot(data=df, x="Category", y="Ideal Word Magnitude", hue="modality", split=True)
    sns.stripplot(data=df, jitter=True, size=5, alpha=1.0, x="Category", y="Ideal Word Magnitude", hue="modality", edgecolor="k", linewidth=1.3)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(args.output_dir / f"{dataset_model_str}_ideal_words_mag.png")
    plt.close()
    
    train_image_features = train_image_features.to(args.device)
    image_features = image_features.to(args.device)
    text_features = text_features.to(args.device)
    all_ideal_words = all_ideal_words.to(args.device)
    all_ideal_images = all_ideal_images.to(args.device)
    all_ideal_words_minus_u0 = all_ideal_words_minus_u0.to(args.device)
    all_ideal_images_minus_u0 = all_ideal_images_minus_u0.to(args.device)

    train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    mean_word = text_features.mean(dim=0)
    mean_image = image_features.mean(dim=0)

    ideal_oracle_objs = torch.stack([all_ideal_images[n_attr:][l] for l in obj_labels], dim=0)

    print("#"*5, "real words", "#"*5)
    print("attr acc", compute_accuracy(torch.argmax(image_features @ text_features[:n_attr].T, dim=1).cpu(), attr_labels))
    print("obj acc", compute_accuracy(torch.argmax(image_features @ text_features[n_attr:].T, dim=1).cpu(), obj_labels))
    text_features_per_attr = torch.stack([text_features[l] for l in attr_labels], dim=0)
    text_features_per_obj = torch.stack([text_features[l] for l in obj_labels], dim=0)
    mean_cossim = F.cosine_similarity(torch.concat([image_features, image_features]), torch.concat([text_features_per_attr, text_features_per_obj]), dim=-1).mean().item()
    print("cos sim", mean_cossim)


    print("#"*5, "ideal words", "#"*5)
    print("attr acc", compute_accuracy(torch.argmax(image_features @ all_ideal_words[:n_attr].T, dim=1).cpu(), attr_labels))
    print("obj acc", compute_accuracy(torch.argmax(image_features @ all_ideal_words[n_attr:].T, dim=1).cpu(), obj_labels))
    mean_cossim = F.cosine_similarity(all_ideal_images, all_ideal_words, dim=-1).mean().item()
    print("cos sim", mean_cossim)

    
    print("#"*5, "ideal words minus u0", "#"*5)
    print("attr acc", compute_accuracy(torch.argmax(image_features @ all_ideal_words_minus_u0[:n_attr].T, dim=1).cpu(), attr_labels))
    print("obj acc", compute_accuracy(torch.argmax(image_features @ all_ideal_words_minus_u0[n_attr:].T, dim=1).cpu(), obj_labels))
    mean_cossim = F.cosine_similarity(all_ideal_images_minus_u0, all_ideal_words_minus_u0, dim=-1).mean().item()
    print("cos sim", mean_cossim)

    print("#"*5, "ideal words & mean gap vector", "#"*5)
    gap_vector = train_image_features.mean(dim=0) - text_features.mean(dim=0)
    all_ideal_words_with_mean_gap_vector = all_ideal_words + gap_vector
    print("attr acc", compute_accuracy(torch.argmax(image_features @ all_ideal_words_with_mean_gap_vector[:n_attr].T, dim=1).cpu(), attr_labels))
    print("obj acc", compute_accuracy(torch.argmax(image_features @ all_ideal_words_with_mean_gap_vector[n_attr:].T, dim=1).cpu(), obj_labels))
    mean_cossim = F.cosine_similarity(all_ideal_images, all_ideal_words_with_mean_gap_vector, dim=-1).mean().item()
    print("cos sim", mean_cossim)
    
    print("#"*5, "ideal words minus u0 & mean gap vector", "#"*5)
    gap_vector = train_image_features.mean(dim=0) - text_features.mean(dim=0)
    all_ideal_words_minus_u0_with_mean_gap_vector = all_ideal_words_minus_u0 + gap_vector
    print("attr acc", compute_accuracy(torch.argmax(image_features @ all_ideal_words_minus_u0_with_mean_gap_vector[:n_attr].T, dim=1).cpu(), attr_labels))
    print("obj acc", compute_accuracy(torch.argmax(image_features @ all_ideal_words_minus_u0_with_mean_gap_vector[n_attr:].T, dim=1).cpu(), obj_labels))
    mean_cossim = F.cosine_similarity(all_ideal_images_minus_u0, all_ideal_words_minus_u0_with_mean_gap_vector, dim=-1).mean().item()
    print("cos sim", mean_cossim)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modality gap vs. performance")
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14__openai",
        help="clip model",
        choices=[f"{model_name}__{pretrain_dataset}" for model_name, pretrain_dataset in open_clip.list_pretrained()],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mit-states",
        choices=["mit-states", "ut-zappos"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/clip",
    )
    parser.add_argument(
        "--DEBUG",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)