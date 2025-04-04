from copy import deepcopy
import json

longer_trained_model_exists_for = [
    "ViT-B-16-plus-240__laion400m_e31",
    "ViT-B-16__laion400m_e31",
    "ViT-B-32-quickgelu__laion400m_e31",
    "ViT-B-32__laion400m_e31",
    "ViT-L-14__laion400m_e31",
    # "ViT-g-14__laion2b_s12b_b42k",
]
finetuned = [
    "coca_ViT-B-32__mscoco_finetuned_laion2b_s13b_b90k",
    "coca_ViT-L-14__mscoco_finetuned_laion2b_s13b_b90k",
]
exclude_model_name_list = longer_trained_model_exists_for + finetuned

precise_size = True
dataset_to_size = {
    "openai": 400*1e6,
    "laion400m": 413*1e6 if precise_size else 400*1e6, # https://laion.ai/blog/laion-400-open-dataset/; Tab. 1 in https://arxiv.org/pdf/2111.02114
    "metaclip_400m": 400*1e6,
    "laion2b": 2.3*1e9 if precise_size else 2*1e9, # https://arxiv.org/pdf/2210.08402
    "laion5b": 5.85*1e9 if precise_size else 5*1e9, # https://arxiv.org/pdf/2210.08402
    "yfcc15m": 14829396, # 15M; https://github.com/openai/CLIP/blob/main/data/yfcc100m.md
    "cc12m": 12423374, # 12M; https://arxiv.org/pdf/2102.08981
    "laion_aesthetic": 900*1e6, # 900M; https://huggingface.co/laion/CLIP-convnext_base_w-laion_aesthetic-s13B-b82K
    "metaclip_fullcc": 2.5*1e9, # 2.5B, https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/pretrained.py#L201
    "merged2b": 2*1e9, # https://arxiv.org/pdf/2303.15389
    "v1": 106246, # 106246, # https://arxiv.org/abs/2309.01859

    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/docs/datacomp_models.md
    "datacomp_s_": 1.4*1e6, # 1.4M
    "datacomp_m_": 14*1e6, # 14M
    "datacomp_l_": 140*1e6, # 140M
    "datacomp_s34b": 1.4*1e9, # 1.4B
    "datacomp1b": 1.4*1e9, # 1.4B; https://arxiv.org/pdf/2305.07017
    "datacomp_xl": 1.4*1e9, # 1.4B; https://github.com/mlfoundations/datacomp

    # Tab. 3 in https://arxiv.org/pdf/2304.14108 & https://github.com/mlfoundations/open_clip/blob/main/docs/datacomp_models.md
    "commonpool_s_basic": 3*1e6, # 3M
    "commonpool_s_text": 3.2*1e6, # 3.2M
    "commonpool_s_image": 3*1e6, # 3M
    "commonpool_s_laion": 1.3*1e6, # 1.3M
    "commonpool_s_clip": 3.8*1e6, # 3.8M
    "commonpool_s_s": 12.8*1e6, # 12.8M

    "commonpool_m_s": 128*1e6, # 128M
    "commonpool_m_basic": 30*1e6, # 30M
    "commonpool_m_text": 31*1e6, # 31M
    "commonpool_m_image": 29*1e6, # 29M
    "commonpool_m_laion": 13*1e6, # 13M
    "commonpool_m_clip": 38*1e6, # 38M

    "commonpool_l_s": 1.28*1e9, # 1.28B
    "commonpool_l_basic": 298*1e6, # 298M
    "commonpool_l_text": 317*1e6, # 317M
    "commonpool_l_image": 293*1e6, # 293M
    "commonpool_l_laion": 130*1e6, # 130M
    "commonpool_l_clip": 384*1e6, # 384M

    "commonpool_xl_s": 12.8*1e9, # 12.8B
    "commonpool_xl_laion": 1.3*1e9, # 1.3B
    "commonpool_xl_clip": 3.8*1e9, # 3.8B

    "webli": 400*1e6, # Fig. 4 in https://arxiv.org/pdf/2209.06794; 40.3% english, WebLI complete ca. 1B -> 400M
}

small_to_medium_datasets = [
    k for k,v in dataset_to_size.items() if v <= 140*1e6
]

def get_exclude_indices(datasets: list):
    return [i for i,d in enumerate(datasets) if any(d1 in d for d1 in small_to_medium_datasets)]

def get_small_and_large_indices(model_names: list, coco_i2t_performance: list):
    train_datasets = []
    for model_name in model_names:
        _, pretrain_dataset = model_name.split("__")
        train_datasets.append(pretrain_dataset)
    exclude_indices = get_exclude_indices(train_datasets)

    small_allowed_indices = [i for i, p in enumerate(coco_i2t_performance) if i in exclude_indices and p >= 5.0 and model_names[i] not in exclude_model_name_list]
    large_allowed_indices = [i for i, p in enumerate(coco_i2t_performance) if i not in exclude_indices and p >= 5.0 and model_names[i] not in exclude_model_name_list]

    return small_allowed_indices, large_allowed_indices


def read_vals_from_json_files(json_filepaths: list, keys: list) -> dict:
    def get_nested_value(d, key_str):
        keys = key_str.split(".")
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                raise KeyError(f"Key '{key}' not found in {d}")
        return d
    
    data = [json.load(open(f, mode="r")) for f in json_filepaths if ".json" in f]
    out_dict = {
        k: [get_nested_value(d, k) for d in data] for k in keys
    }
    return out_dict

if __name__ == "__main__":
    print(exclude_model_name_list)
    print(small_to_medium_datasets)
    print(dataset_to_size)
    print(sum([v for k,v in dataset_to_size.items() if k in small_to_medium_datasets]))