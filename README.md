# Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models (ICLR 2025 Oral)

This repository provides the code for our paper [Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models](https://openreview.net/forum?id=uAFHCZRmXk).

If this work is useful to you, please consider citing our paper:

```
@inproceedings{schrodi2025two,
    title={Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models},
    author={Simon Schrodi and David T. Hoffmann and Max Argus and Volker Fischer and Thomas Brox},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=uAFHCZRmXk}
}
```

## Prerequisites

To setup your Python environment follow the steps below:

1. Clone this repository
2. Create a Python environment 
3. Create the folder `datasets` and add symbolic links to the respective datasets or update the configs in `settings.py`
4. [Optionally]: You can add a folder for `results` and `figures` that are symbolically linked to some workspace
5. Python package installation:
    
    a. For analysis and experiments on MAD: `pip install -r requirements.txt`
    
    b. For experiments on real data: Follow the instructions [here](clip_on_real_data/README.md)

## Analysis 

### Modality gap

This part describes how you can reproduce the results from the analysis of the modality gap.

1. Run `python analyis/gap_precompute.py --model $model --save`, e.g., with `model=RN50__openai`. This will precompute the embedding features and modality gap as well as performance metrics that we will later use.

2. Below, we provide the scripts to reproduce the results (note that you may need to set):
* To re-create Figure 3 and Table 1, run: `python analysis/gap_vs_performance.py`
* To re-create Figure 4a & 11a, run: `python analysis/gap_mean_differences.py`
* To re-create Figure 4b, 11b & 12, run: `python analysis/gap_embedding_dim_pairs.py`
* To re-create Figure 4c & 11c, run: `python analysis/gap_ablate_dims.py`
* To re-create Table 2, run: Follow the instructions [here](https://github.com/boschresearch/rince/tree/imagenet100) to prepare the ImageNet100 splits. Then, run `python analysis/gap_neighborhood_test.py`
* To re-create Table 3, run: `python analysis/gap_conditioning_on_data.py`
* To re-create Table 4, run: `python analysis/gap_data_filtering.py`
* To re-create Table 5, run: `python analysis/gap_ideal_words.py`

### Object bias

1. Precompute features and object bias/performance metrics, via `python analysis/object_bias_precompute.py --model RN50__openai --save`

2. To replicate our analysis, run the scripts below:
* To re-create Figure 5a, run: `python analysis/object_bias_vs_performance.py`
* To re-create Figure 5b, run: `python analysis/object_vs_attribute_performance.py`

## CLIP trainings on synthetic data (MAD)

We provide the dataset implementation in `mad_dataset`. The augmentations are partly based on [Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST).

Unfortunately, we are not allowed to share the training and evaluation code. To reproduce our experiments, you can adopt standard CLIP training pipelines and adapt the provided evaluation protocols.

## CLIP trainings on real data (CC12M and CC3M training, and DCI finetuning)

The code for training of CLIP models on CC12M and CC3M in [clip_on_real_data](clip_on_real_data/) is adopted from [OpenCLIP](https://github.com/mlfoundations/open_clip). We provide setup and run instructions in the [README in clip_on_real_data](clip_on_real_data/README.md).
