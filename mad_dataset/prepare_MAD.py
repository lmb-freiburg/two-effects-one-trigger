import os
from pathlib import Path
import torchvision
from tqdm import tqdm
import torch
import numpy as np
import random
from morphoMNIST.morphomnist import morpho, perturb
from itertools import product
from copy import deepcopy
from skimage import transform
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Any

thickthinning = ["no thickthinning", "thickening", "thinning"]
swelling = ["no swelling", "swelling"]
fracture = ["no fracture", "fracture"]
scaling = ["large", "small"]
colors = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"]

def rescale(image: np.ndarray) -> np.ndarray:
    new_image = np.zeros_like(image)
    image_rescaled = (transform.rescale(image.astype(np.float32)/255, 0.75, anti_aliasing=False)*255).astype(np.uint8)
    y = random.choice([0, 1])
    x = random.choice([0, 1])
    new_image[3 if y == 0 else 4:-3 if y==1 else -4, 3 if x == 0 else 4:-3 if x==1 else -4] = image_rescaled
    return new_image

class ImageCaptionDatasetCreator(Dataset):
    def __init__(self, root, train: bool = True):
        self.image_data = torchvision.datasets.MNIST(root=root, train=train, download=True)
        self.aug_combinations = list(product(*[thickthinning, swelling, fracture, scaling]))

    def __len__(self) -> int:
        return len(self.image_data)*len(self.aug_combinations)
    
    def __getitem__(self, index) -> Any:
        image, cls_label = self.image_data.__getitem__(index//len(self.aug_combinations))
        combo = self.aug_combinations[index%len(self.aug_combinations)]

        morphology = morpho.ImageMorphology(deepcopy(image), scale=4)
            
        if combo[0] == "thickening":
            morphology = perturb.Thickening(amount=.7)(morphology)
        elif combo[0] == "thinning":
            morphology = perturb.Thinning(amount=.7)(morphology)

        if combo[1] == "swelling":
            morphology = perturb.Swelling(strength=3, radius=7)(morphology)

        if combo[2] == "fracture":
            morphology = perturb.Fracture(num_frac=3)(morphology)

        perturbed_image = morphology.downscale(morphology.binary_image)

        if combo[3] == "small":
            perturbed_image = rescale(perturbed_image)

        return torch.from_numpy(perturbed_image), cls_label, "-".join(combo)

def main(args):
    args.out_dir = Path(args.out_dir + f"_{args.seed}")
    print(args.out_dir)
    args.out_dir.mkdir(exist_ok=True)
    print(args.out_dir)

    mmap_filepath = args.out_dir / f"{'test' if args.test else 'train'}.mmap"
    json_filepath = args.out_dir / f"{'test' if args.test else 'train'}.json"
    if os.path.isfile(mmap_filepath):
        os.remove(mmap_filepath)
        os.remove(json_filepath)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = ImageCaptionDatasetCreator(root=args.root, train=not args.test)

    loader = DataLoader(
        dataset=data,
        batch_size=2 if args.DEBUG else 24,
        shuffle=False,
        num_workers=20,
    )

    out_json = {}
    image_file = np.memmap(mmap_filepath, dtype=np.uint8, mode="w+", shape=(len(data), 28, 28))
    counter_idx = 0
    for image, cls_label, combo in tqdm(loader, leave=False, total=len(loader)):
        image_file[counter_idx:counter_idx+image.size(0)] = deepcopy(image.numpy())
        for label, comb in zip(cls_label, combo):
            comb_split = comb.split("-")
            out_json[counter_idx] = {
                "mmap_idx": counter_idx,
                "cls_label": label.item(),
                "thickthinnig": comb_split[0],
                "swelling": comb_split[1],
                "fracture": comb_split[2],
                "scaling": comb_split[3],
            }
            counter_idx += 1
    
    with open(json_filepath, mode="w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-generate data")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(os.getcwd()) / "datasets" / "morphoMNIST")
    )
    parser.add_argument(
        "--root",
        type=str,
        default="."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--test",
        action="store_true"
    )
    parser.add_argument(
        "--DEBUG",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)