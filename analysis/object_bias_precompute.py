from functools import partial
import torch
from pathlib import Path
import open_clip
import os
from tqdm import tqdm
import json

from data_loaders import CompositionDataset
from settings import MIT_STATES_DIR, UT_ZAPPOS_DIR

@torch.no_grad()
def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (torch.sum(preds==labels) / preds.size(0)).item()*100

@torch.no_grad()
def inter_intra_cossine_sims(all_features, all_cls_labels=None, all_att_mat_bool=None):
    '''
    all_features: (batch, dunn_idx_img_cls)
    all_cls_labels: (batch, ) class labels
    all_att_mat_bool: (batch, nr_individual attributes ) (ie. sinlge column for example red and green, not a general color column)

    use either all_cls_labels or all_att_mat_bool
    '''

    def cos_sim_matrix(outputs, outputs2=None):
        if outputs2 is None:
            outputs2 = outputs
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        outputs2 = torch.nn.functional.normalize(outputs2, p=2, dim=1)
        cos_sim = torch.mm(outputs, outputs2.T)
        return cos_sim

    assert not (all_cls_labels is None and all_att_mat_bool is None) and (not all_cls_labels is None or not all_att_mat_bool is None) # only one should be set

    intra_sims_batched = []
    cross_sims_batched = []
    intra_sims_min = []
    cross_sims_max = []
    intra_var = []
    inter_var = []
    intra_n = 0
    cross_n = 0
    if not all_cls_labels is None:
        labels = all_cls_labels
    else:
        labels = torch.arange(0, all_att_mat_bool.shape[1])

    all_features = torch.nn.functional.normalize(all_features.float(), p=2, dim=1)
    for label in labels.unique():
        if not all_cls_labels is None:
            idxs = labels == label
        else:
            idxs = all_att_mat_bool[:, label] > 0
        
        if len(idxs.nonzero()) == 0:
            continue

        # upper_tria_idxs = np.triu_indices(n=sum(idxs), k=1)
        # sims = cos_sim_matrix((all_features[idxs, :],
        #                       all_features[idxs, :])[upper_tria_idxs[0], upper_tria_idxs[1]]).cpu().reshape(-1, 1)
        # intra_n += upper_tria_idxs[0].shape[0]

        cossim = all_features[idxs] @ all_features[idxs].T
        sims = cossim[torch.triu_indices(cossim.size(0), cossim.size(1), offset=1).unbind()].cpu()
        intra_n += sims.size(0)

        # min and max switched because of similarity/dissimilarity
        intra_sims_min.append(torch.min(sims).item())
        intra_sims_batched.append(torch.sum(sims).item())
        intra_var.append(torch.var(all_features[idxs, :], dim=0).cpu())
        
        for label2 in labels.unique():
            if label2 == label:
                continue
            if not all_cls_labels is None:
                idxs2 = labels == label2
            else:
                idxs2 = all_att_mat_bool[:, label2] > 0
            
            if len(idxs2.nonzero()) == 0:
                continue

            # upper_tria_idxs = np.triu_indices(n=sum(idxs), k=1, m=sum(idxs2))
            # sims = (cos_sim_matrix(all_features[idxs, :],
            #                        all_features[idxs2, :])[upper_tria_idxs]).cpu().reshape(-1, 1)
            # cross_n += upper_tria_idxs[0].shape[0]

            cossim = all_features[idxs] @ all_features[idxs2].T
            sims = cossim[torch.triu_indices(cossim.size(0), cossim.size(1), offset=1).unbind()].cpu()
            cross_n += sims.size(0)

            # min and max switched because of similarity/dissimilarity
            cross_sims_max.append(torch.max(sims).item())
            cross_sims_batched.append(torch.sum(sims).item())

    # go to dissimilarity to apply the dunn indx
    dunn_DELTA = torch.min(1-((torch.Tensor(cross_sims_max)+1)/2))
    dunn_delta = torch.max(1-((torch.Tensor(intra_sims_min)+1)/2))

    mean_intra_sim = torch.sum(torch.Tensor(intra_sims_batched)) / intra_n
    mean_cross_sim = torch.sum(torch.Tensor(cross_sims_batched)) / cross_n
    intra_var = torch.mean(torch.cat(intra_var, dim=0)).cpu()

    return {'mean_intra_sim': mean_intra_sim.item(),
            'mean_cross_sim': mean_cross_sim.item(),
            'intra_var': intra_var.item(),
            'dunn_idx': (dunn_DELTA / dunn_delta).item()}

@torch.no_grad()
def encode_clip(model, loader, tokenizer, prompts=None):
    device = next(model.parameters()).device

    with torch.amp.autocast(device_type=device.type):
        all_image_features, labels = [], []
        attr_labels, attr_label = [], None
        for batch in tqdm(loader, leave=False, total=len(loader)):
            if len(batch) == 3:
                inputs, attr_label, label = batch
            else:
                raise NotImplementedError
            if isinstance(inputs, tuple):
                inputs = torch.stack(inputs, dim=0)
            
            image_features = model.encode_image(inputs.to(device))        
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features.cpu())

            labels.append(label)
            attr_labels.append(attr_label)
        
        all_image_features = torch.cat(all_image_features, dim=0)

        if prompts is not None:
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
            attr_labels = torch.cat(attr_labels, dim=0)
            return all_image_features, text_features, labels, attr_labels
        else:
            raise NotImplementedError

def main(args):
    args.output_dir = Path(args.output_dir) / "performance_vs_object_bias"
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name, pretrain_dataset = args.model.split("__")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain_dataset, cache_dir=args.cache_dir)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(args.device)
    model.eval()
    tokenizer = partial(tokenizer, context_length=model.context_length)

    datasets = ["mit-states", "ut-zappos"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        print(5*"#", dataset, 5*"#")
        if "mit-states" in dataset or "ut-zappos" == dataset:   
            if "mit-states" in dataset:
                root = MIT_STATES_DIR
            elif "ut-zappos" == dataset:
                root = UT_ZAPPOS_DIR
            else:
                raise NotImplementedError
            data = CompositionDataset(root=root, split="test", transform=preprocess, also_return_obj_label=True)
            loader = torch.utils.data.DataLoader(
                data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
            )
            prompts = [f"an image of a {attr.lower().replace('.', ' ')} object" for attr in data.attr2idx.keys()]
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
            if os.path.isfile(args.output_dir / f"{dataset_model_str}_attr_labels.pth"):
                attr_labels = torch.load(args.output_dir / f"{dataset_model_str}_attr_labels.pth", map_location="cpu")
        else:
            if "mit-states" in dataset or "ut-zappos" == dataset:
                image_features, text_features, obj_labels, attr_labels = encode_clip(model, loader, tokenizer=tokenizer, prompts=prompts)
                if args.save:
                    torch.save(attr_labels, args.output_dir / f"{dataset_model_str}_attr_labels.pth")
            else:
                raise NotImplementedError
            if args.save:
                torch.save(obj_labels, args.output_dir / f"{dataset_model_str}_obj_labels.pth")
            if args.save:
                torch.save(image_features, args.output_dir / f"{dataset_model_str}_img_feats.pth")
                torch.save(text_features, args.output_dir / f"{dataset_model_str}_txt_feats.pth")

        if "mit-states" in dataset or dataset == "ut-zappos":
            labels = attr_labels
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_features = image_features.to(args.device).float()
        text_features = text_features.to(args.device).float()

        # evaluation
        res = {}

        res["nof_images"] = image_features.size(0)
        res["nof_texts"] = text_features.size(0)
                
        # object bias metrics
        all_att_mat = torch.nn.functional.one_hot(attr_labels, num_classes=attr_labels.max()+1)
        attrs = list(data.attr2idx.keys())
        objs = list(data.obj2idx.keys())
        matching_prompts = [f"an image of a {attrs[attr_labels[idx].item()]} {objs[obj_labels[idx].item()]}" for idx in range(image_features.size(0))]
        matching_text_features = []
        with torch.no_grad():
            text = tokenizer(matching_prompts).to(args.device)
            try:
                matching_text_features = model.encode_text(text).cpu()
            except Exception:
                matching_text_features = []
                for t in tqdm(text, leave=False):
                    matching_text_features.append(model.encode_text(t.unsqueeze(0)).cpu())
                matching_text_features = torch.cat(matching_text_features, dim=0)
            matching_text_features /= matching_text_features.norm(dim=-1, keepdim=True)
            matching_text_features = matching_text_features.to(args.device)
        
        res["obj_bias"] = {}
        res["obj_bias"]["img_obj"] = inter_intra_cossine_sims(image_features.float(), all_cls_labels=obj_labels)
        res["obj_bias"]["img_attr"] = inter_intra_cossine_sims(image_features.float(), all_att_mat_bool=all_att_mat)
        res["obj_bias"]["txt_obj"] = inter_intra_cossine_sims(matching_text_features.float(), all_cls_labels=obj_labels)
        res["obj_bias"]["txt_attr"] = inter_intra_cossine_sims(matching_text_features.float(), all_att_mat_bool=all_att_mat)
        
        # downstream performance metrics
        preds = torch.argmax(image_features @ text_features.T, dim=1).cpu()
        res["acc"] = compute_accuracy(preds, labels)
        
        with open(args.output_dir / f"{dataset_model_str}.json", mode="w", encoding="utf-8") as f:
            json.dump(res, f, indent=4)

        print(res)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Performance vs Object Bias")
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
        choices=["all", "mit-states", "ut-zappos"],
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
    args = parser.parse_args()
    main(args)