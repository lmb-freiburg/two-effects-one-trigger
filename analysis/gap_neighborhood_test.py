import torch
from tqdm import tqdm, trange
import random
import numpy as np
from torch.utils.data import DataLoader
import open_clip
from functools import partial
import torchvision
import os
import itertools
from pathlib import Path

from cifar_class_names import cifar10_classes, cifar100_classes
from data_loaders import ImageNet, openai_imagenet_classes, get_name2_id
from settings import IMAGENET100_S1, IMAGENET100_S2, IMAGENET100_S3

@torch.no_grad()
def encode_clip(model, loader, tokenizer, prompts=None):
    device = next(model.parameters()).device
    all_image_features, labels, all_text_features = [], [], []
    for inputs, label in tqdm(loader, leave=False, total=len(loader)):
        image_features = model.encode_image(inputs.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        all_image_features.append(image_features.cpu())

        if isinstance(label, list):  # is ovad
            label = label[1] - 1
        labels.append(label)

    all_image_features = torch.cat(all_image_features, dim=0)

    if prompts is not None:
        labels = torch.cat(labels, dim=0)
        text = tokenizer(prompts).to(device)
        text_features = model.encode_text(text).cpu()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = torch.cat(all_text_features, dim=0)

    return all_image_features.float(), text_features.float(), labels


def compute_ranks(arr):
    # Create a sorted list of tuples (element, index)
    sorted_arr = sorted(enumerate(arr), key=lambda x: x[1])

    # Create a dictionary to store the rank of each element
    rank_dict = {element[0]: i + 1 for i, element in enumerate(sorted_arr)}

    # Create an array to store the ranks
    ranks = [rank_dict[element] for element in range(len(arr))]

    return np.array(ranks)


def kendall_tau_distance(order_a, order_b):
    pairs = itertools.combinations(range(1, len(order_a) + 1), 2)
    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    return distance / (0.5 * len(order_a) * (len(order_a) - 1))


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = Path(args.output_dir)

    device = 'cuda'

    model_name, pretrain_dataset = args.model.split('__')
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain_dataset,
                                                                 cache_dir=args.cache_dir)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    tokenizer = partial(tokenizer, context_length=model.context_length)

    if args.dataset == 'cifar10':
        data = torchvision.datasets.CIFAR10('datasets/cifar', train=False, transform=preprocess,
                                            download=True)
        prompts = [f'A photo of a {l}' for l in cifar10_classes]
    elif args.dataset == 'cifar100':
        data = torchvision.datasets.CIFAR100('datasets/cifar', train=False, transform=preprocess,
                                             download=True)
        prompts = [f'A photo of a {l}' for l in cifar100_classes]
    elif 'imagenet100' in args.dataset:
        if args.dataset == 'imagenet100_1':
            imagenet100 = IMAGENET100_S1
        elif args.dataset == 'imagenet100_2':
            imagenet100 = IMAGENET100_S2
        elif args.dataset == 'imagenet100_3':
            imagenet100 = IMAGENET100_S3
        data = ImageNet(imagenet100, transform=preprocess)
        wnet_id = os.listdir(imagenet100)
        n2id = get_name2_id()
        prompts = [f'A photo of a {openai_imagenet_classes[int(n2id[obj])]}.' for obj in wnet_id]

    dataset_model_str = f'{args.dataset}_{args.model}'

    if os.path.isfile(args.output_dir / f'{dataset_model_str}_image_features.pth'):
        image_features = torch.load(args.output_dir / f'{dataset_model_str}_image_features.pth').to(device)
        text_features = torch.load(args.output_dir / f'{dataset_model_str}_text_features.pth').to(device)
        obj_labels = torch.load(args.output_dir / f'{dataset_model_str}_obj_labels.pth').to(device)
    else:
        loader = DataLoader(
            dataset=data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=None,
        )
        image_features, text_features, obj_labels = encode_clip(model, loader, tokenizer, prompts=prompts)
        torch.save(image_features, args.output_dir / f'{dataset_model_str}_image_features.pth')
        torch.save(text_features, args.output_dir / f'{dataset_model_str}_text_features.pth')
        torch.save(obj_labels, args.output_dir / f'{dataset_model_str}_obj_labels.pth')

        image_features = image_features.to(device)
        text_features = text_features.to(device)
        obj_labels = obj_labels.to(device)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    dist_metric = lambda vec1, vec2: (vec1 @ vec2.T).mean()

    img_mat = np.empty((len(obj_labels.unique()), len(obj_labels.unique())))
    txt_mat = np.empty((len(obj_labels.unique()), len(obj_labels.unique())))
    for l1 in tqdm(obj_labels.unique(), leave=False):
        for l2 in obj_labels.unique():
            img_mat[l1, l2] = dist_metric(image_features[obj_labels == l1], image_features[obj_labels == l2]).item()
            txt_mat[l1, l2] = dist_metric(text_features[l1, None], text_features[l2, None]).item()

    kendalls = []
    for l1 in trange(len(obj_labels.unique()), leave=False):
        img_vec = img_mat[l1]
        txt_vec = txt_mat[l1]

        img_vec_rank = compute_ranks(img_vec)
        txt_vec_rank = compute_ranks(txt_vec)

        dist = kendall_tau_distance(list(img_vec_rank), list(txt_vec_rank))
        kendalls.append(dist)

    print(np.mean(kendalls))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='(Un)order in the embedding spaces')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100', 'imagenet', 'imagenet100_1', 'imagenet100_2', 'imagenet100_3']
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ViT-B-16__openai',
        help='clip model',
        choices=[f'{model_name}__{pretrain_dataset}' for model_name, pretrain_dataset in open_clip.list_pretrained()],
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/neighborhood_test',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache/clip',
    )
    parser.add_argument(
        '--DEBUG',
        action='store_true'
    )
    args = parser.parse_args()
    main(args)