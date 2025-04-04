import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import re

from settings import DCI_DIR

class DCI(Dataset):
    def __init__(self, root, short=False):
        self.root = root
        
        self.data = []
        for filename in os.listdir(os.path.join(self.root, "annotations")):
            if ".json" not in filename:
                continue
            
            annotation_json = os.path.join(self.root, "annotations", filename)
            assert os.path.exists(annotation_json), f"Annotation file {annotation_json} does not exist"
            with open(annotation_json, "r") as f:
                annotation = json.load(f)

            if short:
                caption = annotation["short_caption"]
                if caption == "":
                    caption = re.split(r'(?<=[.!?])\s+', annotation["extra_caption"])[0]
            else:
                caption = annotation["short_caption"] + annotation["extra_caption"] # simply concat short and extra caption
            caption = caption.replace("\n", " ").replace("\t", " ").replace("\r", " ")
                
            filepath = os.path.join(self.root, "photos", annotation["image"])
            assert os.path.exists(filepath), f"File {filepath} does not exist"
            
            self.data.append((filepath, caption))

    def __len__(self):
        return len(self.data)

def build_dataset(dataset_name, root="root", transform=None, split="test", download=True, annotation_file=None, language="en", task="zeroshot_classification", wds_cache_dir=None, custom_classname_file=None, custom_template_file=None, **kwargs):
    """
    Main function to use in order to build a dataset instance,

    dataset_name: str
        name of the dataset
    
    root: str
        root folder where the dataset is downloaded and stored. can be shared among datasets.

    transform: torchvision transform applied to images

    split: str
        split to use, depending on the dataset can have different options.
        In general, `train` and `test` are available.
        For specific splits, please look at the corresponding dataset.
    
    annotation_file: str or None
        only for datasets with captions (used for retrieval) such as COCO
        and Flickr.
    
    custom_classname_file: str or None
        Custom classname file where keys are dataset names and values are list of classnames.

    custom_template_file: str or None
        Custom template file where keys are dataset names and values are list of prompts, or dicts
        where keys are classnames and values are class-specific prompts.

    """
    use_classnames_and_templates = task in ('zeroshot_classification', 'linear_probe')
    if use_classnames_and_templates:  # Only load templates and classnames if we have to
        current_folder = os.path.dirname(__file__)

        # Load <LANG>_classnames.json (packaged with CLIP benchmark that are used by default)
        default_classname_file = os.path.join(current_folder, language + "_classnames.json")
        if os.path.exists(default_classname_file):
            with open(default_classname_file, "r") as f:
                default_classnames = json.load(f)
        else:
            default_classnames = None
        
        # Load <LANG>_zeroshot_classification_templates.json  (packaged with CLIP benchmark that are used by default)
        default_template_file = os.path.join(current_folder, language + "_zeroshot_classification_templates.json")
        if os.path.exists(default_template_file):
            with open(default_template_file, "r") as f:
                default_templates = json.load(f)
        else:
            default_templates = None
        
        # Load custom classnames file if --custom_classname_file is specified
        if custom_classname_file:
            if not os.path.exists(custom_classname_file):
                custom_classname_file = os.path.join(current_folder, custom_classname_file)
            assert os.path.exists(custom_classname_file), f"Custom classname file '{custom_classname_file}' does not exist"
            with open(custom_classname_file, "r") as f:
                custom_classnames = json.load(f)
        else:
            custom_classnames = None
        
        # Load custom template file if --custom_template_file is specified
        if custom_template_file:
            if not os.path.exists(custom_template_file):
                # look at current_folder
                custom_template_file = os.path.join(current_folder, custom_template_file)
            assert os.path.exists(custom_template_file), f"Custom template file '{custom_template_file}' does not exist"
            with open(custom_template_file, "r") as f:
                custom_templates = json.load(f)
        else:
            custom_templates = None

    if dataset_name == "dci":
        ds = DCI(root=root, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

    default_dataset_for_templates = "imagenet1k"
    if dataset_name.startswith("tfds/") or dataset_name.startswith("vtab/") or dataset_name.startswith("wds/"):
        prefix, *rest = dataset_name.split("/")
        short_name = "/".join(rest)
        # if it's a vtab/tfds/wds/ dataset, we look for e.g. vtab/<name>  
        # as well as <name> in the custom template file/classname file,
        # whichever is found.
        keys_to_lookup = [dataset_name, short_name]
    else:
        keys_to_lookup = [dataset_name]
    
    if use_classnames_and_templates:
        # Specify templates for the dataset (if needed)
        if custom_templates:
            # We override with custom templates ONLY if they are provided,
            # which is the case when `custom_templates` is loaded.
            ds.templates = value_from_first_key_found(
                custom_templates, keys=keys_to_lookup + [default_dataset_for_templates]
            )
            assert ds.templates is not None, f"Templates not specified for {dataset_name}"          
        elif not hasattr(ds, "templates"):
            # No templates specified by the dataset itself, 
            # so we use  templates are packaged with CLIP benchmark 
            # (loaded from <LANG>_zeroshot_classification_templates.json).
            ds.templates = value_from_first_key_found(default_templates, keys=keys_to_lookup + [default_dataset_for_templates])
            assert ds.templates is not None, f"Templates not specified for {dataset_name}"            
        else:
            # dataset has templates already (e.g., WDS case), so we keep it as is.
            pass

         # We override with custom classnames ONLY if they are provided.
        if custom_classnames:
            ds.classes = value_from_first_key_found(custom_classnames, keys=keys_to_lookup)
        
        assert ds.classes is not None, f"Classes not specified for {dataset_name}"
        assert ds.templates is not None, f"Templates not specified for {dataset_name}"
    return ds


def value_from_first_key_found(dic, keys):
    for k in keys:
        if k in dic:
            return dic[k]

ds = build_dataset("dci", root=DCI_DIR, task="captioning", short=True) # this downloads the dataset if it is not there already
future_df = {"filepath":[], "title":[]}
for img_filepath, caption in tqdm(ds.data, leave=False):
    future_df["filepath"].append(img_filepath)
    future_df["title"].append(caption)
pd.DataFrame.from_dict(future_df).to_csv(
  "dci_short.csv", index=False, sep="\t"
)