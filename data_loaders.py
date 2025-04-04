import linecache
import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any
from torchvision.datasets.folder import ImageFolder

from imagenet_classnames import name_map, folder_label_map

class EmbeddingAttr(Dataset):
    def __init__(self, embeddings, attrs) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.attrs = attrs

    def __getitem__(self, index) -> Any:
        return self.embeddings[index], self.attrs[index]

    def __len__(self):
        return len(self.embeddings)

openai_imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "eft",
                           "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier",
                           "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo",
                           "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat",
                           "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox","maillot",
                           "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial",
                           "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

# class ImageNet(ImageFolder):
#     classes = [name_map[i] for i in range(1000)]
#     name_map = name_map

#     def __init__(self, root:str, split:str="train", transform=None, target_transform=None, class_idcs=None,
#                 **kwargs):
#         _ = kwargs  # Just for consistency with other datasets.
#         assert split in ["train", "val"]
#         path = os.path.join(root, split)
#         super().__init__(path, transform=transform, target_transform=target_transform)
#         if class_idcs is not None:
#             class_idcs = list(sorted(class_idcs))
#             tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
#             self.classes = [self.classes[c] for c in class_idcs]
#             self.samples = [(p, tgt_to_tgt_map[t]) for p, t in self.samples if t in tgt_to_tgt_map]
#             self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}


#         self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
#         self.targets = np.array(self.samples)[:, 1]

class ImageNet(ImageFolder):
    classes = [name_map[i] for i in range(1000)]
    name_map = name_map

    def __init__(
            self,
            root:str,
            split:str="val",
            transform=None,
            target_transform=None,
            class_idcs=None,
            **kwargs
    ):
        path = root if "val" in root or root[-5:] == "train" else os.path.join(root, split)
        super().__init__(path, transform=transform, target_transform=target_transform)

        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            samples = []
            idx_to_tgt_cls = []
            for i, (p, t) in enumerate(self.samples):
                if t in tgt_to_tgt_map:
                    samples.append((p, tgt_to_tgt_map[t]))
                    idx_to_tgt_cls.append(self.idx_to_tgt_cls[i])

            self.idx_to_tgt_cls = idx_to_tgt_cls
            # self.samples = [(p, tgt_to_tgt_map[t]) for i, (p, t) in enumerate(self.samples) if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}

        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
        self.targets = np.array(self.samples)[:, 1]

    # def __getitem__(self, index):
    #     sample = super().__getitem__(index)
    #     return sample


class CompositionDataset(Dataset):
    # joint data loader to mit-states and ut-zappos

    def __init__(self, root, split, transform, target_transform=None, antonym_prompts: bool = False,
                 also_return_obj_label: bool = False):
        self.root = root
        self.split = split

        # Load metadata
        self.metadata = torch.load(os.path.join(root, "metadata_compositional-split-natural.t7"))

        # Load attribute-noun pairs for each split
        all_info, split_info = self.parse_split()
        self.attrs, self.objs, self.pairs = all_info
        self.train_pairs, self.valid_pairs, self.test_pairs = split_info

        # Get obj/attr/pair to indices mappings
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.idx2obj = {idx: obj for obj, idx in self.obj2idx.items()}
        self.idx2attr = {idx: attr for attr, idx in self.attr2idx.items()}
        self.idx2pair = {idx: pair for pair, idx in self.pair2idx.items()}
        self.unique_objs = list(set([noun for _, noun in self.pairs]))
        self.unique_attrs = list(set([attr for attr, _ in self.pairs]))
        self.antonym_data = load_antonym_data(root)

        assert (antonym_prompts and len(self.antonym_data) > 0) or not antonym_prompts

        # Get all data
        self.train_data, self.valid_data, self.test_data = self.get_split_info()
        if self.split == "train":
            self.data = self.train_data
        elif self.split == "valid":
            self.data = self.valid_data
        else:
            self.data = self.test_data

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        self.transform = transform
        self.target_transform = target_transform
        self.antonym_prompts = antonym_prompts
        self.also_return_obj_label = also_return_obj_label

    def parse_split(self):
        def parse_pairs(pair_path):
            with open(pair_path, "r") as f:
                pairs = f.read().strip().split("\n")
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            os.path.join(self.root, "compositional-split-natural", "train_pairs.txt"))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            os.path.join(self.root, "compositional-split-natural", "val_pairs.txt"))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            os.path.join(self.root, "compositional-split-natural", "test_pairs.txt"))

        all_attrs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs)))
        all_objs = sorted(list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return (all_attrs, all_objs, all_pairs), (tr_pairs, vl_pairs, ts_pairs)

    def get_split_info(self):
        train_data, val_data, test_data = [], [], []
        for instance in self.metadata:
            image, attr, obj, settype = instance["image"], instance["attr"], instance["obj"], instance["set"]
            image = image.split("/")[1]  # Get the image name without (attr, obj) folder
            image = os.path.join(self.root, "images", " ".join([attr, obj]), image)

            if (
                    (attr == "NA") or
                    ((attr, obj) not in self.pairs) or
                    (settype == "NA")
            ):
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = {
                "image_path": image,
                "attr": attr,
                "obj": obj,
                "pair": (attr, obj),
                "attr_id": self.attr2idx[attr],
                "obj_id": self.obj2idx[obj],
                "pair_id": self.pair2idx[(attr, obj)],
            }
            if settype == "train":
                train_data.append(data_i)
            elif settype == "val":
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index = self.sample_indices[index]
        data_dict = self.data[index]

        img = self.transform(Image.open(os.path.join(data_dict["image_path"])))

        if self.target_transform:
            if self.antonym_prompts:
                captions = self.target_transform(data_dict["pair"], self.antonym_data[data_dict["attr"]],
                                                 self.unique_objs)
            else:
                captions = self.target_transform(data_dict["pair"], self.unique_attrs, self.unique_objs)
            return img, (captions, self.attr2idx[data_dict["pair"][0]])
        if self.also_return_obj_label:
            return img, self.attr2idx[data_dict["pair"][0]], data_dict["obj_id"]
        return img, self.attr2idx[data_dict["pair"][0]]


def softmax(inputs):
    res = torch.tensor(inputs).float()
    res = res.softmax(dim=-1)
    return res.numpy()


def normalize(inputs):
    res = torch.tensor(inputs).float()
    res /= res.norm(dim=-1, keepdim=True)
    return res.numpy()


def get_gt_primitives(split, data, ):
    """ Get groundtruth primtiive concepts. """
    data_dict = {
        "train": data.train_data,
        "valid": data.valid_data,
        "test": data.test_data,
    }
    split_data = data_dict[split]
    labels_attr = [sample["attr_id"] for sample in split_data]
    labels_obj = [sample["obj_id"] for sample in split_data]
    gt_features_attr = np.zeros((len(split_data), len(data.attrs)))
    gt_features_obj = np.zeros((len(split_data), len(data.objs)))
    gt_features_attr[np.arange(len(labels_attr)), labels_attr] = 1
    gt_features_obj[np.arange(len(labels_obj)), labels_obj] = 1
    gt_features_concat = np.concatenate([gt_features_attr, gt_features_obj], axis=-1)
    return gt_features_concat


def get_precomputed_features(feature, args, is_softmax=False):
    """ Get precomputed CLIP image/pair/attr/obj features """
    data_root = args.precomputed_data_root
    feature_name = "image_features" if feature == "image" else f"{feature}_activations"
    feature_train = np.load(os.path.join(data_root, f"{feature_name}_train.npy"))
    feature_valid = np.load(os.path.join(data_root, f"{feature_name}_valid.npy"))
    feature_test = np.load(os.path.join(data_root, f"{feature_name}_test.npy"))
    if is_softmax:
        feature_train = softmax(feature_train)
        feature_valid = softmax(feature_valid)
        feature_test = softmax(feature_test)
    return feature_train, feature_valid, feature_test


def load_antonym_data(data_root):
    antonym_dict = {}
    antonym_path = os.path.join(data_root, "adj_ants.csv")
    if not os.path.isfile(antonym_path):
        return antonym_dict
    with open(antonym_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip(",\n").split(",")
            antonym_dict[words[0]] = words[1:] if len(words) > 1 else []
    return antonym_dict



id_2_classid_and_name  = {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"],
                               "2": ["n01484850", "great_white_shark"],
                               "3": ["n01491361", "tiger_shark"], "4": ["n01494475", "hammerhead"],
                               "5": ["n01496331", "electric_ray"],
                               "6": ["n01498041", "stingray"], "7": ["n01514668", "cock"], "8": ["n01514859", "hen"],
                               "9": ["n01518878", "ostrich"], "10": ["n01530575", "brambling"],
                               "11": ["n01531178", "goldfinch"],
                               "12": ["n01532829", "house_finch"], "13": ["n01534433", "junco"],
                               "14": ["n01537544", "indigo_bunting"],
                               "15": ["n01558993", "robin"], "16": ["n01560419", "bulbul"], "17": ["n01580077", "jay"],
                               "18": ["n01582220", "magpie"], "19": ["n01592084", "chickadee"],
                               "20": ["n01601694", "water_ouzel"],
                               "21": ["n01608432", "kite"], "22": ["n01614925", "bald_eagle"],
                               "23": ["n01616318", "vulture"],
                               "24": ["n01622779", "great_grey_owl"], "25": ["n01629819", "European_fire_salamander"],
                               "26": ["n01630670", "common_newt"], "27": ["n01631663", "eft"],
                               "28": ["n01632458", "spotted_salamander"],
                               "29": ["n01632777", "axolotl"], "30": ["n01641577", "bullfrog"],
                               "31": ["n01644373", "tree_frog"],
                               "32": ["n01644900", "tailed_frog"], "33": ["n01664065", "loggerhead"],
                               "34": ["n01665541", "leatherback_turtle"], "35": ["n01667114", "mud_turtle"],
                               "36": ["n01667778", "terrapin"],
                               "37": ["n01669191", "box_turtle"], "38": ["n01675722", "banded_gecko"],
                               "39": ["n01677366", "common_iguana"],
                               "40": ["n01682714", "American_chameleon"], "41": ["n01685808", "whiptail"],
                               "42": ["n01687978", "agama"],
                               "43": ["n01688243", "frilled_lizard"], "44": ["n01689811", "alligator_lizard"],
                               "45": ["n01692333", "Gila_monster"], "46": ["n01693334", "green_lizard"],
                               "47": ["n01694178", "African_chameleon"], "48": ["n01695060", "Komodo_dragon"],
                               "49": ["n01697457", "African_crocodile"], "50": ["n01698640", "American_alligator"],
                               "51": ["n01704323", "triceratops"], "52": ["n01728572", "thunder_snake"],
                               "53": ["n01728920", "ringneck_snake"], "54": ["n01729322", "hognose_snake"],
                               "55": ["n01729977", "green_snake"], "56": ["n01734418", "king_snake"],
                               "57": ["n01735189", "garter_snake"],
                               "58": ["n01737021", "water_snake"], "59": ["n01739381", "vine_snake"],
                               "60": ["n01740131", "night_snake"],
                               "61": ["n01742172", "boa_constrictor"], "62": ["n01744401", "rock_python"],
                               "63": ["n01748264", "Indian_cobra"], "64": ["n01749939", "green_mamba"],
                               "65": ["n01751748", "sea_snake"],
                               "66": ["n01753488", "horned_viper"], "67": ["n01755581", "diamondback"],
                               "68": ["n01756291", "sidewinder"],
                               "69": ["n01768244", "trilobite"], "70": ["n01770081", "harvestman"],
                               "71": ["n01770393", "scorpion"],
                               "72": ["n01773157", "black_and_gold_garden_spider"], "73": ["n01773549", "barn_spider"],
                               "74": ["n01773797", "garden_spider"], "75": ["n01774384", "black_widow"],
                               "76": ["n01774750", "tarantula"],
                               "77": ["n01775062", "wolf_spider"], "78": ["n01776313", "tick"],
                               "79": ["n01784675", "centipede"],
                               "80": ["n01795545", "black_grouse"], "81": ["n01796340", "ptarmigan"],
                               "82": ["n01797886", "ruffed_grouse"],
                               "83": ["n01798484", "prairie_chicken"], "84": ["n01806143", "peacock"],
                               "85": ["n01806567", "quail"],
                               "86": ["n01807496", "partridge"], "87": ["n01817953", "African_grey"],
                               "88": ["n01818515", "macaw"],
                               "89": ["n01819313", "sulphur-crested_cockatoo"], "90": ["n01820546", "lorikeet"],
                               "91": ["n01824575", "coucal"], "92": ["n01828970", "bee_eater"],
                               "93": ["n01829413", "hornbill"],
                               "94": ["n01833805", "hummingbird"], "95": ["n01843065", "jacamar"],
                               "96": ["n01843383", "toucan"],
                               "97": ["n01847000", "drake"], "98": ["n01855032", "red-breasted_merganser"],
                               "99": ["n01855672", "goose"],
                               "100": ["n01860187", "black_swan"], "101": ["n01871265", "tusker"],
                               "102": ["n01872401", "echidna"],
                               "103": ["n01873310", "platypus"], "104": ["n01877812", "wallaby"],
                               "105": ["n01882714", "koala"],
                               "106": ["n01883070", "wombat"], "107": ["n01910747", "jellyfish"],
                               "108": ["n01914609", "sea_anemone"],
                               "109": ["n01917289", "brain_coral"], "110": ["n01924916", "flatworm"],
                               "111": ["n01930112", "nematode"],
                               "112": ["n01943899", "conch"], "113": ["n01944390", "snail"],
                               "114": ["n01945685", "slug"],
                               "115": ["n01950731", "sea_slug"], "116": ["n01955084", "chiton"],
                               "117": ["n01968897", "chambered_nautilus"],
                               "118": ["n01978287", "Dungeness_crab"], "119": ["n01978455", "rock_crab"],
                               "120": ["n01980166", "fiddler_crab"], "121": ["n01981276", "king_crab"],
                               "122": ["n01983481", "American_lobster"], "123": ["n01984695", "spiny_lobster"],
                               "124": ["n01985128", "crayfish"], "125": ["n01986214", "hermit_crab"],
                               "126": ["n01990800", "isopod"],
                               "127": ["n02002556", "white_stork"], "128": ["n02002724", "black_stork"],
                               "129": ["n02006656", "spoonbill"],
                               "130": ["n02007558", "flamingo"], "131": ["n02009229", "little_blue_heron"],
                               "132": ["n02009912", "American_egret"], "133": ["n02011460", "bittern"],
                               "134": ["n02012849", "crane"],
                               "135": ["n02013706", "limpkin"], "136": ["n02017213", "European_gallinule"],
                               "137": ["n02018207", "American_coot"], "138": ["n02018795", "bustard"],
                               "139": ["n02025239", "ruddy_turnstone"], "140": ["n02027492", "red-backed_sandpiper"],
                               "141": ["n02028035", "redshank"], "142": ["n02033041", "dowitcher"],
                               "143": ["n02037110", "oystercatcher"],
                               "144": ["n02051845", "pelican"], "145": ["n02056570", "king_penguin"],
                               "146": ["n02058221", "albatross"],
                               "147": ["n02066245", "grey_whale"], "148": ["n02071294", "killer_whale"],
                               "149": ["n02074367", "dugong"],
                               "150": ["n02077923", "sea_lion"], "151": ["n02085620", "Chihuahua"],
                               "152": ["n02085782", "Japanese_spaniel"],
                               "153": ["n02085936", "Maltese_dog"], "154": ["n02086079", "Pekinese"],
                               "155": ["n02086240", "Shih-Tzu"],
                               "156": ["n02086646", "Blenheim_spaniel"], "157": ["n02086910", "papillon"],
                               "158": ["n02087046", "toy_terrier"], "159": ["n02087394", "Rhodesian_ridgeback"],
                               "160": ["n02088094", "Afghan_hound"], "161": ["n02088238", "basset"],
                               "162": ["n02088364", "beagle"],
                               "163": ["n02088466", "bloodhound"], "164": ["n02088632", "bluetick"],
                               "165": ["n02089078", "black-and-tan_coonhound"], "166": ["n02089867", "Walker_hound"],
                               "167": ["n02089973", "English_foxhound"], "168": ["n02090379", "redbone"],
                               "169": ["n02090622", "borzoi"],
                               "170": ["n02090721", "Irish_wolfhound"], "171": ["n02091032", "Italian_greyhound"],
                               "172": ["n02091134", "whippet"], "173": ["n02091244", "Ibizan_hound"],
                               "174": ["n02091467", "Norwegian_elkhound"], "175": ["n02091635", "otterhound"],
                               "176": ["n02091831", "Saluki"],
                               "177": ["n02092002", "Scottish_deerhound"], "178": ["n02092339", "Weimaraner"],
                               "179": ["n02093256", "Staffordshire_bullterrier"],
                               "180": ["n02093428", "American_Staffordshire_terrier"],
                               "181": ["n02093647", "Bedlington_terrier"], "182": ["n02093754", "Border_terrier"],
                               "183": ["n02093859", "Kerry_blue_terrier"], "184": ["n02093991", "Irish_terrier"],
                               "185": ["n02094114", "Norfolk_terrier"], "186": ["n02094258", "Norwich_terrier"],
                               "187": ["n02094433", "Yorkshire_terrier"],
                               "188": ["n02095314", "wire-haired_fox_terrier"],
                               "189": ["n02095570", "Lakeland_terrier"], "190": ["n02095889", "Sealyham_terrier"],
                               "191": ["n02096051", "Airedale"], "192": ["n02096177", "cairn"],
                               "193": ["n02096294", "Australian_terrier"],
                               "194": ["n02096437", "Dandie_Dinmont"], "195": ["n02096585", "Boston_bull"],
                               "196": ["n02097047", "miniature_schnauzer"], "197": ["n02097130", "giant_schnauzer"],
                               "198": ["n02097209", "standard_schnauzer"], "199": ["n02097298", "Scotch_terrier"],
                               "200": ["n02097474", "Tibetan_terrier"], "201": ["n02097658", "silky_terrier"],
                               "202": ["n02098105", "soft-coated_wheaten_terrier"],
                               "203": ["n02098286", "West_Highland_white_terrier"],
                               "204": ["n02098413", "Lhasa"], "205": ["n02099267", "flat-coated_retriever"],
                               "206": ["n02099429", "curly-coated_retriever"], "207": ["n02099601", "golden_retriever"],
                               "208": ["n02099712", "Labrador_retriever"],
                               "209": ["n02099849", "Chesapeake_Bay_retriever"],
                               "210": ["n02100236", "German_short-haired_pointer"], "211": ["n02100583", "vizsla"],
                               "212": ["n02100735", "English_setter"], "213": ["n02100877", "Irish_setter"],
                               "214": ["n02101006", "Gordon_setter"], "215": ["n02101388", "Brittany_spaniel"],
                               "216": ["n02101556", "clumber"], "217": ["n02102040", "English_springer"],
                               "218": ["n02102177", "Welsh_springer_spaniel"], "219": ["n02102318", "cocker_spaniel"],
                               "220": ["n02102480", "Sussex_spaniel"], "221": ["n02102973", "Irish_water_spaniel"],
                               "222": ["n02104029", "kuvasz"], "223": ["n02104365", "schipperke"],
                               "224": ["n02105056", "groenendael"],
                               "225": ["n02105162", "malinois"], "226": ["n02105251", "briard"],
                               "227": ["n02105412", "kelpie"],
                               "228": ["n02105505", "komondor"], "229": ["n02105641", "Old_English_sheepdog"],
                               "230": ["n02105855", "Shetland_sheepdog"], "231": ["n02106030", "collie"],
                               "232": ["n02106166", "Border_collie"], "233": ["n02106382", "Bouvier_des_Flandres"],
                               "234": ["n02106550", "Rottweiler"], "235": ["n02106662", "German_shepherd"],
                               "236": ["n02107142", "Doberman"],
                               "237": ["n02107312", "miniature_pinscher"],
                               "238": ["n02107574", "Greater_Swiss_Mountain_dog"],
                               "239": ["n02107683", "Bernese_mountain_dog"], "240": ["n02107908", "Appenzeller"],
                               "241": ["n02108000", "EntleBucher"], "242": ["n02108089", "boxer"],
                               "243": ["n02108422", "bull_mastiff"],
                               "244": ["n02108551", "Tibetan_mastiff"], "245": ["n02108915", "French_bulldog"],
                               "246": ["n02109047", "Great_Dane"], "247": ["n02109525", "Saint_Bernard"],
                               "248": ["n02109961", "Eskimo_dog"],
                               "249": ["n02110063", "malamute"], "250": ["n02110185", "Siberian_husky"],
                               "251": ["n02110341", "dalmatian"],
                               "252": ["n02110627", "affenpinscher"], "253": ["n02110806", "basenji"],
                               "254": ["n02110958", "pug"],
                               "255": ["n02111129", "Leonberg"], "256": ["n02111277", "Newfoundland"],
                               "257": ["n02111500", "Great_Pyrenees"],
                               "258": ["n02111889", "Samoyed"], "259": ["n02112018", "Pomeranian"],
                               "260": ["n02112137", "chow"],
                               "261": ["n02112350", "keeshond"], "262": ["n02112706", "Brabancon_griffon"],
                               "263": ["n02113023", "Pembroke"],
                               "264": ["n02113186", "Cardigan"], "265": ["n02113624", "toy_poodle"],
                               "266": ["n02113712", "miniature_poodle"],
                               "267": ["n02113799", "standard_poodle"], "268": ["n02113978", "Mexican_hairless"],
                               "269": ["n02114367", "timber_wolf"], "270": ["n02114548", "white_wolf"],
                               "271": ["n02114712", "red_wolf"],
                               "272": ["n02114855", "coyote"], "273": ["n02115641", "dingo"],
                               "274": ["n02115913", "dhole"],
                               "275": ["n02116738", "African_hunting_dog"], "276": ["n02117135", "hyena"],
                               "277": ["n02119022", "red_fox"],
                               "278": ["n02119789", "kit_fox"], "279": ["n02120079", "Arctic_fox"],
                               "280": ["n02120505", "grey_fox"],
                               "281": ["n02123045", "tabby"], "282": ["n02123159", "tiger_cat"],
                               "283": ["n02123394", "Persian_cat"],
                               "284": ["n02123597", "Siamese_cat"], "285": ["n02124075", "Egyptian_cat"],
                               "286": ["n02125311", "cougar"],
                               "287": ["n02127052", "lynx"], "288": ["n02128385", "leopard"],
                               "289": ["n02128757", "snow_leopard"],
                               "290": ["n02128925", "jaguar"], "291": ["n02129165", "lion"],
                               "292": ["n02129604", "tiger"],
                               "293": ["n02130308", "cheetah"], "294": ["n02132136", "brown_bear"],
                               "295": ["n02133161", "American_black_bear"], "296": ["n02134084", "ice_bear"],
                               "297": ["n02134418", "sloth_bear"], "298": ["n02137549", "mongoose"],
                               "299": ["n02138441", "meerkat"],
                               "300": ["n02165105", "tiger_beetle"], "301": ["n02165456", "ladybug"],
                               "302": ["n02167151", "ground_beetle"],
                               "303": ["n02168699", "long-horned_beetle"], "304": ["n02169497", "leaf_beetle"],
                               "305": ["n02172182", "dung_beetle"], "306": ["n02174001", "rhinoceros_beetle"],
                               "307": ["n02177972", "weevil"],
                               "308": ["n02190166", "fly"], "309": ["n02206856", "bee"], "310": ["n02219486", "ant"],
                               "311": ["n02226429", "grasshopper"], "312": ["n02229544", "cricket"],
                               "313": ["n02231487", "walking_stick"],
                               "314": ["n02233338", "cockroach"], "315": ["n02236044", "mantis"],
                               "316": ["n02256656", "cicada"],
                               "317": ["n02259212", "leafhopper"], "318": ["n02264363", "lacewing"],
                               "319": ["n02268443", "dragonfly"],
                               "320": ["n02268853", "damselfly"], "321": ["n02276258", "admiral"],
                               "322": ["n02277742", "ringlet"],
                               "323": ["n02279972", "monarch"], "324": ["n02280649", "cabbage_butterfly"],
                               "325": ["n02281406", "sulphur_butterfly"], "326": ["n02281787", "lycaenid"],
                               "327": ["n02317335", "starfish"],
                               "328": ["n02319095", "sea_urchin"], "329": ["n02321529", "sea_cucumber"],
                               "330": ["n02325366", "wood_rabbit"],
                               "331": ["n02326432", "hare"], "332": ["n02328150", "Angora"],
                               "333": ["n02342885", "hamster"],
                               "334": ["n02346627", "porcupine"], "335": ["n02356798", "fox_squirrel"],
                               "336": ["n02361337", "marmot"],
                               "337": ["n02363005", "beaver"], "338": ["n02364673", "guinea_pig"],
                               "339": ["n02389026", "sorrel"],
                               "340": ["n02391049", "zebra"], "341": ["n02395406", "hog"],
                               "342": ["n02396427", "wild_boar"],
                               "343": ["n02397096", "warthog"], "344": ["n02398521", "hippopotamus"],
                               "345": ["n02403003", "ox"],
                               "346": ["n02408429", "water_buffalo"], "347": ["n02410509", "bison"],
                               "348": ["n02412080", "ram"],
                               "349": ["n02415577", "bighorn"], "350": ["n02417914", "ibex"],
                               "351": ["n02422106", "hartebeest"],
                               "352": ["n02422699", "impala"], "353": ["n02423022", "gazelle"],
                               "354": ["n02437312", "Arabian_camel"],
                               "355": ["n02437616", "llama"], "356": ["n02441942", "weasel"],
                               "357": ["n02442845", "mink"],
                               "358": ["n02443114", "polecat"], "359": ["n02443484", "black-footed_ferret"],
                               "360": ["n02444819", "otter"],
                               "361": ["n02445715", "skunk"], "362": ["n02447366", "badger"],
                               "363": ["n02454379", "armadillo"],
                               "364": ["n02457408", "three-toed_sloth"], "365": ["n02480495", "orangutan"],
                               "366": ["n02480855", "gorilla"],
                               "367": ["n02481823", "chimpanzee"], "368": ["n02483362", "gibbon"],
                               "369": ["n02483708", "siamang"],
                               "370": ["n02484975", "guenon"], "371": ["n02486261", "patas"],
                               "372": ["n02486410", "baboon"],
                               "373": ["n02487347", "macaque"], "374": ["n02488291", "langur"],
                               "375": ["n02488702", "colobus"],
                               "376": ["n02489166", "proboscis_monkey"], "377": ["n02490219", "marmoset"],
                               "378": ["n02492035", "capuchin"],
                               "379": ["n02492660", "howler_monkey"], "380": ["n02493509", "titi"],
                               "381": ["n02493793", "spider_monkey"],
                               "382": ["n02494079", "squirrel_monkey"], "383": ["n02497673", "Madagascar_cat"],
                               "384": ["n02500267", "indri"],
                               "385": ["n02504013", "Indian_elephant"], "386": ["n02504458", "African_elephant"],
                               "387": ["n02509815", "lesser_panda"], "388": ["n02510455", "giant_panda"],
                               "389": ["n02514041", "barracouta"],
                               "390": ["n02526121", "eel"], "391": ["n02536864", "coho"],
                               "392": ["n02606052", "rock_beauty"],
                               "393": ["n02607072", "anemone_fish"], "394": ["n02640242", "sturgeon"],
                               "395": ["n02641379", "gar"],
                               "396": ["n02643566", "lionfish"], "397": ["n02655020", "puffer"],
                               "398": ["n02666196", "abacus"],
                               "399": ["n02667093", "abaya"], "400": ["n02669723", "academic_gown"],
                               "401": ["n02672831", "accordion"],
                               "402": ["n02676566", "acoustic_guitar"], "403": ["n02687172", "aircraft_carrier"],
                               "404": ["n02690373", "airliner"], "405": ["n02692877", "airship"],
                               "406": ["n02699494", "altar"],
                               "407": ["n02701002", "ambulance"], "408": ["n02704792", "amphibian"],
                               "409": ["n02708093", "analog_clock"],
                               "410": ["n02727426", "apiary"], "411": ["n02730930", "apron"],
                               "412": ["n02747177", "ashcan"],
                               "413": ["n02749479", "assault_rifle"], "414": ["n02769748", "backpack"],
                               "415": ["n02776631", "bakery"],
                               "416": ["n02777292", "balance_beam"], "417": ["n02782093", "balloon"],
                               "418": ["n02783161", "ballpoint"],
                               "419": ["n02786058", "Band_Aid"], "420": ["n02787622", "banjo"],
                               "421": ["n02788148", "bannister"],
                               "422": ["n02790996", "barbell"], "423": ["n02791124", "barber_chair"],
                               "424": ["n02791270", "barbershop"],
                               "425": ["n02793495", "barn"], "426": ["n02794156", "barometer"],
                               "427": ["n02795169", "barrel"],
                               "428": ["n02797295", "barrow"], "429": ["n02799071", "baseball"],
                               "430": ["n02802426", "basketball"],
                               "431": ["n02804414", "bassinet"], "432": ["n02804610", "bassoon"],
                               "433": ["n02807133", "bathing_cap"],
                               "434": ["n02808304", "bath_towel"], "435": ["n02808440", "bathtub"],
                               "436": ["n02814533", "beach_wagon"],
                               "437": ["n02814860", "beacon"], "438": ["n02815834", "beaker"],
                               "439": ["n02817516", "bearskin"],
                               "440": ["n02823428", "beer_bottle"], "441": ["n02823750", "beer_glass"],
                               "442": ["n02825657", "bell_cote"],
                               "443": ["n02834397", "bib"], "444": ["n02835271", "bicycle-built-for-two"],
                               "445": ["n02837789", "bikini"],
                               "446": ["n02840245", "binder"], "447": ["n02841315", "binoculars"],
                               "448": ["n02843684", "birdhouse"],
                               "449": ["n02859443", "boathouse"], "450": ["n02860847", "bobsled"],
                               "451": ["n02865351", "bolo_tie"],
                               "452": ["n02869837", "bonnet"], "453": ["n02870880", "bookcase"],
                               "454": ["n02871525", "bookshop"],
                               "455": ["n02877765", "bottlecap"], "456": ["n02879718", "bow"],
                               "457": ["n02883205", "bow_tie"],
                               "458": ["n02892201", "brass"], "459": ["n02892767", "brassiere"],
                               "460": ["n02894605", "breakwater"],
                               "461": ["n02895154", "breastplate"], "462": ["n02906734", "broom"],
                               "463": ["n02909870", "bucket"],
                               "464": ["n02910353", "buckle"], "465": ["n02916936", "bulletproof_vest"],
                               "466": ["n02917067", "bullet_train"],
                               "467": ["n02927161", "butcher_shop"], "468": ["n02930766", "cab"],
                               "469": ["n02939185", "caldron"],
                               "470": ["n02948072", "candle"], "471": ["n02950826", "cannon"],
                               "472": ["n02951358", "canoe"],
                               "473": ["n02951585", "can_opener"], "474": ["n02963159", "cardigan"],
                               "475": ["n02965783", "car_mirror"],
                               "476": ["n02966193", "carousel"], "477": ["n02966687", "carpenter's_kit"],
                               "478": ["n02971356", "carton"],
                               "479": ["n02974003", "car_wheel"], "480": ["n02977058", "cash_machine"],
                               "481": ["n02978881", "cassette"],
                               "482": ["n02979186", "cassette_player"], "483": ["n02980441", "castle"],
                               "484": ["n02981792", "catamaran"],
                               "485": ["n02988304", "CD_player"], "486": ["n02992211", "cello"],
                               "487": ["n02992529", "cellular_telephone"],
                               "488": ["n02999410", "chain"], "489": ["n03000134", "chainlink_fence"],
                               "490": ["n03000247", "chain_mail"],
                               "491": ["n03000684", "chain_saw"], "492": ["n03014705", "chest"],
                               "493": ["n03016953", "chiffonier"],
                               "494": ["n03017168", "chime"], "495": ["n03018349", "china_cabinet"],
                               "496": ["n03026506", "Christmas_stocking"], "497": ["n03028079", "church"],
                               "498": ["n03032252", "cinema"],
                               "499": ["n03041632", "cleaver"], "500": ["n03042490", "cliff_dwelling"],
                               "501": ["n03045698", "cloak"],
                               "502": ["n03047690", "clog"], "503": ["n03062245", "cocktail_shaker"],
                               "504": ["n03063599", "coffee_mug"],
                               "505": ["n03063689", "coffeepot"], "506": ["n03065424", "coil"],
                               "507": ["n03075370", "combination_lock"],
                               "508": ["n03085013", "computer_keyboard"], "509": ["n03089624", "confectionery"],
                               "510": ["n03095699", "container_ship"], "511": ["n03100240", "convertible"],
                               "512": ["n03109150", "corkscrew"],
                               "513": ["n03110669", "cornet"], "514": ["n03124043", "cowboy_boot"],
                               "515": ["n03124170", "cowboy_hat"],
                               "516": ["n03125729", "cradle"], "517": ["n03126707", "crane"],
                               "518": ["n03127747", "crash_helmet"],
                               "519": ["n03127925", "crate"], "520": ["n03131574", "crib"],
                               "521": ["n03133878", "Crock_Pot"],
                               "522": ["n03134739", "croquet_ball"], "523": ["n03141823", "crutch"],
                               "524": ["n03146219", "cuirass"],
                               "525": ["n03160309", "dam"], "526": ["n03179701", "desk"],
                               "527": ["n03180011", "desktop_computer"],
                               "528": ["n03187595", "dial_telephone"], "529": ["n03188531", "diaper"],
                               "530": ["n03196217", "digital_clock"],
                               "531": ["n03197337", "digital_watch"], "532": ["n03201208", "dining_table"],
                               "533": ["n03207743", "dishrag"],
                               "534": ["n03207941", "dishwasher"], "535": ["n03208938", "disk_brake"],
                               "536": ["n03216828", "dock"],
                               "537": ["n03218198", "dogsled"], "538": ["n03220513", "dome"],
                               "539": ["n03223299", "doormat"],
                               "540": ["n03240683", "drilling_platform"], "541": ["n03249569", "drum"],
                               "542": ["n03250847", "drumstick"],
                               "543": ["n03255030", "dumbbell"], "544": ["n03259280", "Dutch_oven"],
                               "545": ["n03271574", "electric_fan"],
                               "546": ["n03272010", "electric_guitar"], "547": ["n03272562", "electric_locomotive"],
                               "548": ["n03290653", "entertainment_center"], "549": ["n03291819", "envelope"],
                               "550": ["n03297495", "espresso_maker"], "551": ["n03314780", "face_powder"],
                               "552": ["n03325584", "feather_boa"], "553": ["n03337140", "file"],
                               "554": ["n03344393", "fireboat"],
                               "555": ["n03345487", "fire_engine"], "556": ["n03347037", "fire_screen"],
                               "557": ["n03355925", "flagpole"],
                               "558": ["n03372029", "flute"], "559": ["n03376595", "folding_chair"],
                               "560": ["n03379051", "football_helmet"],
                               "561": ["n03384352", "forklift"], "562": ["n03388043", "fountain"],
                               "563": ["n03388183", "fountain_pen"],
                               "564": ["n03388549", "four-poster"], "565": ["n03393912", "freight_car"],
                               "566": ["n03394916", "French_horn"],
                               "567": ["n03400231", "frying_pan"], "568": ["n03404251", "fur_coat"],
                               "569": ["n03417042", "garbage_truck"],
                               "570": ["n03424325", "gasmask"], "571": ["n03425413", "gas_pump"],
                               "572": ["n03443371", "goblet"],
                               "573": ["n03444034", "go-kart"], "574": ["n03445777", "golf_ball"],
                               "575": ["n03445924", "golfcart"],
                               "576": ["n03447447", "gondola"], "577": ["n03447721", "gong"],
                               "578": ["n03450230", "gown"],
                               "579": ["n03452741", "grand_piano"], "580": ["n03457902", "greenhouse"],
                               "581": ["n03459775", "grille"],
                               "582": ["n03461385", "grocery_store"], "583": ["n03467068", "guillotine"],
                               "584": ["n03476684", "hair_slide"],
                               "585": ["n03476991", "hair_spray"], "586": ["n03478589", "half_track"],
                               "587": ["n03481172", "hammer"],
                               "588": ["n03482405", "hamper"], "589": ["n03483316", "hand_blower"],
                               "590": ["n03485407", "hand-held_computer"], "591": ["n03485794", "handkerchief"],
                               "592": ["n03492542", "hard_disc"], "593": ["n03494278", "harmonica"],
                               "594": ["n03495258", "harp"],
                               "595": ["n03496892", "harvester"], "596": ["n03498962", "hatchet"],
                               "597": ["n03527444", "holster"],
                               "598": ["n03529860", "home_theater"], "599": ["n03530642", "honeycomb"],
                               "600": ["n03532672", "hook"],
                               "601": ["n03534580", "hoopskirt"], "602": ["n03535780", "horizontal_bar"],
                               "603": ["n03538406", "horse_cart"],
                               "604": ["n03544143", "hourglass"], "605": ["n03584254", "iPod"],
                               "606": ["n03584829", "iron"],
                               "607": ["n03590841", "jack-o'-lantern"], "608": ["n03594734", "jean"],
                               "609": ["n03594945", "jeep"],
                               "610": ["n03595614", "jersey"], "611": ["n03598930", "jigsaw_puzzle"],
                               "612": ["n03599486", "jinrikisha"],
                               "613": ["n03602883", "joystick"], "614": ["n03617480", "kimono"],
                               "615": ["n03623198", "knee_pad"],
                               "616": ["n03627232", "knot"], "617": ["n03630383", "lab_coat"],
                               "618": ["n03633091", "ladle"],
                               "619": ["n03637318", "lampshade"], "620": ["n03642806", "laptop"],
                               "621": ["n03649909", "lawn_mower"],
                               "622": ["n03657121", "lens_cap"], "623": ["n03658185", "letter_opener"],
                               "624": ["n03661043", "library"],
                               "625": ["n03662601", "lifeboat"], "626": ["n03666591", "lighter"],
                               "627": ["n03670208", "limousine"],
                               "628": ["n03673027", "liner"], "629": ["n03676483", "lipstick"],
                               "630": ["n03680355", "Loafer"],
                               "631": ["n03690938", "lotion"], "632": ["n03691459", "loudspeaker"],
                               "633": ["n03692522", "loupe"],
                               "634": ["n03697007", "lumbermill"], "635": ["n03706229", "magnetic_compass"],
                               "636": ["n03709823", "mailbag"],
                               "637": ["n03710193", "mailbox"], "638": ["n03710637", "maillot"],
                               "639": ["n03710721", "maillot"],
                               "640": ["n03717622", "manhole_cover"], "641": ["n03720891", "maraca"],
                               "642": ["n03721384", "marimba"],
                               "643": ["n03724870", "mask"], "644": ["n03729826", "matchstick"],
                               "645": ["n03733131", "maypole"],
                               "646": ["n03733281", "maze"], "647": ["n03733805", "measuring_cup"],
                               "648": ["n03742115", "medicine_chest"],
                               "649": ["n03743016", "megalith"], "650": ["n03759954", "microphone"],
                               "651": ["n03761084", "microwave"],
                               "652": ["n03763968", "military_uniform"], "653": ["n03764736", "milk_can"],
                               "654": ["n03769881", "minibus"],
                               "655": ["n03770439", "miniskirt"], "656": ["n03770679", "minivan"],
                               "657": ["n03773504", "missile"],
                               "658": ["n03775071", "mitten"], "659": ["n03775546", "mixing_bowl"],
                               "660": ["n03776460", "mobile_home"],
                               "661": ["n03777568", "Model_T"], "662": ["n03777754", "modem"],
                               "663": ["n03781244", "monastery"],
                               "664": ["n03782006", "monitor"], "665": ["n03785016", "moped"],
                               "666": ["n03786901", "mortar"],
                               "667": ["n03787032", "mortarboard"], "668": ["n03788195", "mosque"],
                               "669": ["n03788365", "mosquito_net"],
                               "670": ["n03791053", "motor_scooter"], "671": ["n03792782", "mountain_bike"],
                               "672": ["n03792972", "mountain_tent"], "673": ["n03793489", "mouse"],
                               "674": ["n03794056", "mousetrap"],
                               "675": ["n03796401", "moving_van"], "676": ["n03803284", "muzzle"],
                               "677": ["n03804744", "nail"],
                               "678": ["n03814639", "neck_brace"], "679": ["n03814906", "necklace"],
                               "680": ["n03825788", "nipple"],
                               "681": ["n03832673", "notebook"], "682": ["n03837869", "obelisk"],
                               "683": ["n03838899", "oboe"],
                               "684": ["n03840681", "ocarina"], "685": ["n03841143", "odometer"],
                               "686": ["n03843555", "oil_filter"],
                               "687": ["n03854065", "organ"], "688": ["n03857828", "oscilloscope"],
                               "689": ["n03866082", "overskirt"],
                               "690": ["n03868242", "oxcart"], "691": ["n03868863", "oxygen_mask"],
                               "692": ["n03871628", "packet"],
                               "693": ["n03873416", "paddle"], "694": ["n03874293", "paddlewheel"],
                               "695": ["n03874599", "padlock"],
                               "696": ["n03876231", "paintbrush"], "697": ["n03877472", "pajama"],
                               "698": ["n03877845", "palace"],
                               "699": ["n03884397", "panpipe"], "700": ["n03887697", "paper_towel"],
                               "701": ["n03888257", "parachute"],
                               "702": ["n03888605", "parallel_bars"], "703": ["n03891251", "park_bench"],
                               "704": ["n03891332", "parking_meter"], "705": ["n03895866", "passenger_car"],
                               "706": ["n03899768", "patio"],
                               "707": ["n03902125", "pay-phone"], "708": ["n03903868", "pedestal"],
                               "709": ["n03908618", "pencil_box"],
                               "710": ["n03908714", "pencil_sharpener"], "711": ["n03916031", "perfume"],
                               "712": ["n03920288", "Petri_dish"],
                               "713": ["n03924679", "photocopier"], "714": ["n03929660", "pick"],
                               "715": ["n03929855", "pickelhaube"],
                               "716": ["n03930313", "picket_fence"], "717": ["n03930630", "pickup"],
                               "718": ["n03933933", "pier"],
                               "719": ["n03935335", "piggy_bank"], "720": ["n03937543", "pill_bottle"],
                               "721": ["n03938244", "pillow"],
                               "722": ["n03942813", "ping-pong_ball"], "723": ["n03944341", "pinwheel"],
                               "724": ["n03947888", "pirate"],
                               "725": ["n03950228", "pitcher"], "726": ["n03954731", "plane"],
                               "727": ["n03956157", "planetarium"],
                               "728": ["n03958227", "plastic_bag"], "729": ["n03961711", "plate_rack"],
                               "730": ["n03967562", "plow"],
                               "731": ["n03970156", "plunger"], "732": ["n03976467", "Polaroid_camera"],
                               "733": ["n03976657", "pole"],
                               "734": ["n03977966", "police_van"], "735": ["n03980874", "poncho"],
                               "736": ["n03982430", "pool_table"],
                               "737": ["n03983396", "pop_bottle"], "738": ["n03991062", "pot"],
                               "739": ["n03992509", "potter's_wheel"],
                               "740": ["n03995372", "power_drill"], "741": ["n03998194", "prayer_rug"],
                               "742": ["n04004767", "printer"],
                               "743": ["n04005630", "prison"], "744": ["n04008634", "projectile"],
                               "745": ["n04009552", "projector"],
                               "746": ["n04019541", "puck"], "747": ["n04023962", "punching_bag"],
                               "748": ["n04026417", "purse"],
                               "749": ["n04033901", "quill"], "750": ["n04033995", "quilt"],
                               "751": ["n04037443", "racer"],
                               "752": ["n04039381", "racket"], "753": ["n04040759", "radiator"],
                               "754": ["n04041544", "radio"],
                               "755": ["n04044716", "radio_telescope"], "756": ["n04049303", "rain_barrel"],
                               "757": ["n04065272", "recreational_vehicle"], "758": ["n04067472", "reel"],
                               "759": ["n04069434", "reflex_camera"], "760": ["n04070727", "refrigerator"],
                               "761": ["n04074963", "remote_control"], "762": ["n04081281", "restaurant"],
                               "763": ["n04086273", "revolver"],
                               "764": ["n04090263", "rifle"], "765": ["n04099969", "rocking_chair"],
                               "766": ["n04111531", "rotisserie"],
                               "767": ["n04116512", "rubber_eraser"], "768": ["n04118538", "rugby_ball"],
                               "769": ["n04118776", "rule"],
                               "770": ["n04120489", "running_shoe"], "771": ["n04125021", "safe"],
                               "772": ["n04127249", "safety_pin"],
                               "773": ["n04131690", "saltshaker"], "774": ["n04133789", "sandal"],
                               "775": ["n04136333", "sarong"],
                               "776": ["n04141076", "sax"], "777": ["n04141327", "scabbard"],
                               "778": ["n04141975", "scale"],
                               "779": ["n04146614", "school_bus"], "780": ["n04147183", "schooner"],
                               "781": ["n04149813", "scoreboard"],
                               "782": ["n04152593", "screen"], "783": ["n04153751", "screw"],
                               "784": ["n04154565", "screwdriver"],
                               "785": ["n04162706", "seat_belt"], "786": ["n04179913", "sewing_machine"],
                               "787": ["n04192698", "shield"],
                               "788": ["n04200800", "shoe_shop"], "789": ["n04201297", "shoji"],
                               "790": ["n04204238", "shopping_basket"],
                               "791": ["n04204347", "shopping_cart"], "792": ["n04208210", "shovel"],
                               "793": ["n04209133", "shower_cap"],
                               "794": ["n04209239", "shower_curtain"], "795": ["n04228054", "ski"],
                               "796": ["n04229816", "ski_mask"],
                               "797": ["n04235860", "sleeping_bag"], "798": ["n04238763", "slide_rule"],
                               "799": ["n04239074", "sliding_door"],
                               "800": ["n04243546", "slot"], "801": ["n04251144", "snorkel"],
                               "802": ["n04252077", "snowmobile"],
                               "803": ["n04252225", "snowplow"], "804": ["n04254120", "soap_dispenser"],
                               "805": ["n04254680", "soccer_ball"],
                               "806": ["n04254777", "sock"], "807": ["n04258138", "solar_dish"],
                               "808": ["n04259630", "sombrero"],
                               "809": ["n04263257", "soup_bowl"], "810": ["n04264628", "space_bar"],
                               "811": ["n04265275", "space_heater"],
                               "812": ["n04266014", "space_shuttle"], "813": ["n04270147", "spatula"],
                               "814": ["n04273569", "speedboat"],
                               "815": ["n04275548", "spider_web"], "816": ["n04277352", "spindle"],
                               "817": ["n04285008", "sports_car"],
                               "818": ["n04286575", "spotlight"], "819": ["n04296562", "stage"],
                               "820": ["n04310018", "steam_locomotive"],
                               "821": ["n04311004", "steel_arch_bridge"], "822": ["n04311174", "steel_drum"],
                               "823": ["n04317175", "stethoscope"], "824": ["n04325704", "stole"],
                               "825": ["n04326547", "stone_wall"],
                               "826": ["n04328186", "stopwatch"], "827": ["n04330267", "stove"],
                               "828": ["n04332243", "strainer"],
                               "829": ["n04335435", "streetcar"], "830": ["n04336792", "stretcher"],
                               "831": ["n04344873", "studio_couch"],
                               "832": ["n04346328", "stupa"], "833": ["n04347754", "submarine"],
                               "834": ["n04350905", "suit"],
                               "835": ["n04355338", "sundial"], "836": ["n04355933", "sunglass"],
                               "837": ["n04356056", "sunglasses"],
                               "838": ["n04357314", "sunscreen"], "839": ["n04366367", "suspension_bridge"],
                               "840": ["n04367480", "swab"],
                               "841": ["n04370456", "sweatshirt"], "842": ["n04371430", "swimming_trunks"],
                               "843": ["n04371774", "swing"],
                               "844": ["n04372370", "switch"], "845": ["n04376876", "syringe"],
                               "846": ["n04380533", "table_lamp"],
                               "847": ["n04389033", "tank"], "848": ["n04392985", "tape_player"],
                               "849": ["n04398044", "teapot"],
                               "850": ["n04399382", "teddy"], "851": ["n04404412", "television"],
                               "852": ["n04409515", "tennis_ball"],
                               "853": ["n04417672", "thatch"], "854": ["n04418357", "theater_curtain"],
                               "855": ["n04423845", "thimble"],
                               "856": ["n04428191", "thresher"], "857": ["n04429376", "throne"],
                               "858": ["n04435653", "tile_roof"],
                               "859": ["n04442312", "toaster"], "860": ["n04443257", "tobacco_shop"],
                               "861": ["n04447861", "toilet_seat"],
                               "862": ["n04456115", "torch"], "863": ["n04458633", "totem_pole"],
                               "864": ["n04461696", "tow_truck"],
                               "865": ["n04462240", "toyshop"], "866": ["n04465501", "tractor"],
                               "867": ["n04467665", "trailer_truck"],
                               "868": ["n04476259", "tray"], "869": ["n04479046", "trench_coat"],
                               "870": ["n04482393", "tricycle"],
                               "871": ["n04483307", "trimaran"], "872": ["n04485082", "tripod"],
                               "873": ["n04486054", "triumphal_arch"],
                               "874": ["n04487081", "trolleybus"], "875": ["n04487394", "trombone"],
                               "876": ["n04493381", "tub"],
                               "877": ["n04501370", "turnstile"], "878": ["n04505470", "typewriter_keyboard"],
                               "879": ["n04507155", "umbrella"], "880": ["n04509417", "unicycle"],
                               "881": ["n04515003", "upright"],
                               "882": ["n04517823", "vacuum"], "883": ["n04522168", "vase"],
                               "884": ["n04523525", "vault"],
                               "885": ["n04525038", "velvet"], "886": ["n04525305", "vending_machine"],
                               "887": ["n04532106", "vestment"],
                               "888": ["n04532670", "viaduct"], "889": ["n04536866", "violin"],
                               "890": ["n04540053", "volleyball"],
                               "891": ["n04542943", "waffle_iron"], "892": ["n04548280", "wall_clock"],
                               "893": ["n04548362", "wallet"],
                               "894": ["n04550184", "wardrobe"], "895": ["n04552348", "warplane"],
                               "896": ["n04553703", "washbasin"],
                               "897": ["n04554684", "washer"], "898": ["n04557648", "water_bottle"],
                               "899": ["n04560804", "water_jug"],
                               "900": ["n04562935", "water_tower"], "901": ["n04579145", "whiskey_jug"],
                               "902": ["n04579432", "whistle"],
                               "903": ["n04584207", "wig"], "904": ["n04589890", "window_screen"],
                               "905": ["n04590129", "window_shade"],
                               "906": ["n04591157", "Windsor_tie"], "907": ["n04591713", "wine_bottle"],
                               "908": ["n04592741", "wing"],
                               "909": ["n04596742", "wok"], "910": ["n04597913", "wooden_spoon"],
                               "911": ["n04599235", "wool"],
                               "912": ["n04604644", "worm_fence"], "913": ["n04606251", "wreck"],
                               "914": ["n04612504", "yawl"],
                               "915": ["n04613696", "yurt"], "916": ["n06359193", "web_site"],
                               "917": ["n06596364", "comic_book"],
                               "918": ["n06785654", "crossword_puzzle"], "919": ["n06794110", "street_sign"],
                               "920": ["n06874185", "traffic_light"], "921": ["n07248320", "book_jacket"],
                               "922": ["n07565083", "menu"],
                               "923": ["n07579787", "plate"], "924": ["n07583066", "guacamole"],
                               "925": ["n07584110", "consomme"],
                               "926": ["n07590611", "hot_pot"], "927": ["n07613480", "trifle"],
                               "928": ["n07614500", "ice_cream"],
                               "929": ["n07615774", "ice_lolly"], "930": ["n07684084", "French_loaf"],
                               "931": ["n07693725", "bagel"],
                               "932": ["n07695742", "pretzel"], "933": ["n07697313", "cheeseburger"],
                               "934": ["n07697537", "hotdog"],
                               "935": ["n07711569", "mashed_potato"], "936": ["n07714571", "head_cabbage"],
                               "937": ["n07714990", "broccoli"],
                               "938": ["n07715103", "cauliflower"], "939": ["n07716358", "zucchini"],
                               "940": ["n07716906", "spaghetti_squash"], "941": ["n07717410", "acorn_squash"],
                               "942": ["n07717556", "butternut_squash"], "943": ["n07718472", "cucumber"],
                               "944": ["n07718747", "artichoke"],
                               "945": ["n07720875", "bell_pepper"], "946": ["n07730033", "cardoon"],
                               "947": ["n07734744", "mushroom"],
                               "948": ["n07742313", "Granny_Smith"], "949": ["n07745940", "strawberry"],
                               "950": ["n07747607", "orange"],
                               "951": ["n07749582", "lemon"], "952": ["n07753113", "fig"],
                               "953": ["n07753275", "pineapple"],
                               "954": ["n07753592", "banana"], "955": ["n07754684", "jackfruit"],
                               "956": ["n07760859", "custard_apple"],
                               "957": ["n07768694", "pomegranate"], "958": ["n07802026", "hay"],
                               "959": ["n07831146", "carbonara"],
                               "960": ["n07836838", "chocolate_sauce"], "961": ["n07860988", "dough"],
                               "962": ["n07871810", "meat_loaf"],
                               "963": ["n07873807", "pizza"], "964": ["n07875152", "potpie"],
                               "965": ["n07880968", "burrito"],
                               "966": ["n07892512", "red_wine"], "967": ["n07920052", "espresso"],
                               "968": ["n07930864", "cup"],
                               "969": ["n07932039", "eggnog"], "970": ["n09193705", "alp"],
                               "971": ["n09229709", "bubble"],
                               "972": ["n09246464", "cliff"], "973": ["n09256479", "coral_reef"],
                               "974": ["n09288635", "geyser"],
                               "975": ["n09332890", "lakeside"], "976": ["n09399592", "promontory"],
                               "977": ["n09421951", "sandbar"],
                               "978": ["n09428293", "seashore"], "979": ["n09468604", "valley"],
                               "980": ["n09472597", "volcano"],
                               "981": ["n09835506", "ballplayer"], "982": ["n10148035", "groom"],
                               "983": ["n10565667", "scuba_diver"],
                               "984": ["n11879895", "rapeseed"], "985": ["n11939491", "daisy"],
                               "986": ["n12057211", "yellow_lady's_slipper"],
                               "987": ["n12144580", "corn"], "988": ["n12267677", "acorn"], "989": ["n12620546", "hip"],
                               "990": ["n12768682", "buckeye"], "991": ["n12985857", "coral_fungus"],
                               "992": ["n12998815", "agaric"],
                               "993": ["n13037406", "gyromitra"], "994": ["n13040303", "stinkhorn"],
                               "995": ["n13044778", "earthstar"],
                               "996": ["n13052670", "hen-of-the-woods"], "997": ["n13054560", "bolete"],
                               "998": ["n13133613", "ear"],
                               "999": ["n15075141", "toilet_tissue"]}

def get_name2_id():
    return {'n01440764': '0', 'n01443537': '1', 'n01484850': '2', 'n01491361': '3', 'n01494475': '4', 'n01496331': '5',
            'n01498041': '6', 'n01514668': '7', 'n01514859': '8', 'n01518878': '9', 'n01530575': '10',
            'n01531178': '11', 'n01532829': '12', 'n01534433': '13', 'n01537544': '14', 'n01558993': '15',
            'n01560419': '16', 'n01580077': '17', 'n01582220': '18', 'n01592084': '19', 'n01601694': '20',
            'n01608432': '21', 'n01614925': '22', 'n01616318': '23', 'n01622779': '24', 'n01629819': '25',
            'n01630670': '26', 'n01631663': '27', 'n01632458': '28', 'n01632777': '29', 'n01641577': '30',
            'n01644373': '31', 'n01644900': '32', 'n01664065': '33', 'n01665541': '34', 'n01667114': '35',
            'n01667778': '36', 'n01669191': '37', 'n01675722': '38', 'n01677366': '39', 'n01682714': '40',
            'n01685808': '41', 'n01687978': '42', 'n01688243': '43', 'n01689811': '44', 'n01692333': '45',
            'n01693334': '46', 'n01694178': '47', 'n01695060': '48', 'n01697457': '49', 'n01698640': '50',
            'n01704323': '51', 'n01728572': '52', 'n01728920': '53', 'n01729322': '54', 'n01729977': '55',
            'n01734418': '56', 'n01735189': '57', 'n01737021': '58', 'n01739381': '59', 'n01740131': '60',
            'n01742172': '61', 'n01744401': '62', 'n01748264': '63', 'n01749939': '64', 'n01751748': '65',
            'n01753488': '66', 'n01755581': '67', 'n01756291': '68', 'n01768244': '69', 'n01770081': '70',
            'n01770393': '71', 'n01773157': '72', 'n01773549': '73', 'n01773797': '74', 'n01774384': '75',
            'n01774750': '76', 'n01775062': '77', 'n01776313': '78', 'n01784675': '79', 'n01795545': '80',
            'n01796340': '81', 'n01797886': '82', 'n01798484': '83', 'n01806143': '84', 'n01806567': '85',
            'n01807496': '86', 'n01817953': '87', 'n01818515': '88', 'n01819313': '89', 'n01820546': '90',
            'n01824575': '91', 'n01828970': '92', 'n01829413': '93', 'n01833805': '94', 'n01843065': '95',
            'n01843383': '96', 'n01847000': '97', 'n01855032': '98', 'n01855672': '99', 'n01860187': '100',
            'n01871265': '101', 'n01872401': '102', 'n01873310': '103', 'n01877812': '104', 'n01882714': '105',
            'n01883070': '106', 'n01910747': '107', 'n01914609': '108', 'n01917289': '109', 'n01924916': '110',
            'n01930112': '111', 'n01943899': '112', 'n01944390': '113', 'n01945685': '114', 'n01950731': '115',
            'n01955084': '116', 'n01968897': '117', 'n01978287': '118', 'n01978455': '119', 'n01980166': '120',
            'n01981276': '121', 'n01983481': '122', 'n01984695': '123', 'n01985128': '124', 'n01986214': '125',
            'n01990800': '126', 'n02002556': '127', 'n02002724': '128', 'n02006656': '129', 'n02007558': '130',
            'n02009229': '131', 'n02009912': '132', 'n02011460': '133', 'n02012849': '134', 'n02013706': '135',
            'n02017213': '136', 'n02018207': '137', 'n02018795': '138', 'n02025239': '139', 'n02027492': '140',
            'n02028035': '141', 'n02033041': '142', 'n02037110': '143', 'n02051845': '144', 'n02056570': '145',
            'n02058221': '146', 'n02066245': '147', 'n02071294': '148', 'n02074367': '149', 'n02077923': '150',
            'n02085620': '151', 'n02085782': '152', 'n02085936': '153', 'n02086079': '154', 'n02086240': '155',
            'n02086646': '156', 'n02086910': '157', 'n02087046': '158', 'n02087394': '159', 'n02088094': '160',
            'n02088238': '161', 'n02088364': '162', 'n02088466': '163', 'n02088632': '164', 'n02089078': '165',
            'n02089867': '166', 'n02089973': '167', 'n02090379': '168', 'n02090622': '169', 'n02090721': '170',
            'n02091032': '171', 'n02091134': '172', 'n02091244': '173', 'n02091467': '174', 'n02091635': '175',
            'n02091831': '176', 'n02092002': '177', 'n02092339': '178', 'n02093256': '179', 'n02093428': '180',
            'n02093647': '181', 'n02093754': '182', 'n02093859': '183', 'n02093991': '184', 'n02094114': '185',
            'n02094258': '186', 'n02094433': '187', 'n02095314': '188', 'n02095570': '189', 'n02095889': '190',
            'n02096051': '191', 'n02096177': '192', 'n02096294': '193', 'n02096437': '194', 'n02096585': '195',
            'n02097047': '196', 'n02097130': '197', 'n02097209': '198', 'n02097298': '199', 'n02097474': '200',
            'n02097658': '201', 'n02098105': '202', 'n02098286': '203', 'n02098413': '204', 'n02099267': '205',
            'n02099429': '206', 'n02099601': '207', 'n02099712': '208', 'n02099849': '209', 'n02100236': '210',
            'n02100583': '211', 'n02100735': '212', 'n02100877': '213', 'n02101006': '214', 'n02101388': '215',
            'n02101556': '216', 'n02102040': '217', 'n02102177': '218', 'n02102318': '219', 'n02102480': '220',
            'n02102973': '221', 'n02104029': '222', 'n02104365': '223', 'n02105056': '224', 'n02105162': '225',
            'n02105251': '226', 'n02105412': '227', 'n02105505': '228', 'n02105641': '229', 'n02105855': '230',
            'n02106030': '231', 'n02106166': '232', 'n02106382': '233', 'n02106550': '234', 'n02106662': '235',
            'n02107142': '236', 'n02107312': '237', 'n02107574': '238', 'n02107683': '239', 'n02107908': '240',
            'n02108000': '241', 'n02108089': '242', 'n02108422': '243', 'n02108551': '244', 'n02108915': '245',
            'n02109047': '246', 'n02109525': '247', 'n02109961': '248', 'n02110063': '249', 'n02110185': '250',
            'n02110341': '251', 'n02110627': '252', 'n02110806': '253', 'n02110958': '254', 'n02111129': '255',
            'n02111277': '256', 'n02111500': '257', 'n02111889': '258', 'n02112018': '259', 'n02112137': '260',
            'n02112350': '261', 'n02112706': '262', 'n02113023': '263', 'n02113186': '264', 'n02113624': '265',
            'n02113712': '266', 'n02113799': '267', 'n02113978': '268', 'n02114367': '269', 'n02114548': '270',
            'n02114712': '271', 'n02114855': '272', 'n02115641': '273', 'n02115913': '274', 'n02116738': '275',
            'n02117135': '276', 'n02119022': '277', 'n02119789': '278', 'n02120079': '279', 'n02120505': '280',
            'n02123045': '281', 'n02123159': '282', 'n02123394': '283', 'n02123597': '284', 'n02124075': '285',
            'n02125311': '286', 'n02127052': '287', 'n02128385': '288', 'n02128757': '289', 'n02128925': '290',
            'n02129165': '291', 'n02129604': '292', 'n02130308': '293', 'n02132136': '294', 'n02133161': '295',
            'n02134084': '296', 'n02134418': '297', 'n02137549': '298', 'n02138441': '299', 'n02165105': '300',
            'n02165456': '301', 'n02167151': '302', 'n02168699': '303', 'n02169497': '304', 'n02172182': '305',
            'n02174001': '306', 'n02177972': '307', 'n02190166': '308', 'n02206856': '309', 'n02219486': '310',
            'n02226429': '311', 'n02229544': '312', 'n02231487': '313', 'n02233338': '314', 'n02236044': '315',
            'n02256656': '316', 'n02259212': '317', 'n02264363': '318', 'n02268443': '319', 'n02268853': '320',
            'n02276258': '321', 'n02277742': '322', 'n02279972': '323', 'n02280649': '324', 'n02281406': '325',
            'n02281787': '326', 'n02317335': '327', 'n02319095': '328', 'n02321529': '329', 'n02325366': '330',
            'n02326432': '331', 'n02328150': '332', 'n02342885': '333', 'n02346627': '334', 'n02356798': '335',
            'n02361337': '336', 'n02363005': '337', 'n02364673': '338', 'n02389026': '339', 'n02391049': '340',
            'n02395406': '341', 'n02396427': '342', 'n02397096': '343', 'n02398521': '344', 'n02403003': '345',
            'n02408429': '346', 'n02410509': '347', 'n02412080': '348', 'n02415577': '349', 'n02417914': '350',
            'n02422106': '351', 'n02422699': '352', 'n02423022': '353', 'n02437312': '354', 'n02437616': '355',
            'n02441942': '356', 'n02442845': '357', 'n02443114': '358', 'n02443484': '359', 'n02444819': '360',
            'n02445715': '361', 'n02447366': '362', 'n02454379': '363', 'n02457408': '364', 'n02480495': '365',
            'n02480855': '366', 'n02481823': '367', 'n02483362': '368', 'n02483708': '369', 'n02484975': '370',
            'n02486261': '371', 'n02486410': '372', 'n02487347': '373', 'n02488291': '374', 'n02488702': '375',
            'n02489166': '376', 'n02490219': '377', 'n02492035': '378', 'n02492660': '379', 'n02493509': '380',
            'n02493793': '381', 'n02494079': '382', 'n02497673': '383', 'n02500267': '384', 'n02504013': '385',
            'n02504458': '386', 'n02509815': '387', 'n02510455': '388', 'n02514041': '389', 'n02526121': '390',
            'n02536864': '391', 'n02606052': '392', 'n02607072': '393', 'n02640242': '394', 'n02641379': '395',
            'n02643566': '396', 'n02655020': '397', 'n02666196': '398', 'n02667093': '399', 'n02669723': '400',
            'n02672831': '401', 'n02676566': '402', 'n02687172': '403', 'n02690373': '404', 'n02692877': '405',
            'n02699494': '406', 'n02701002': '407', 'n02704792': '408', 'n02708093': '409', 'n02727426': '410',
            'n02730930': '411', 'n02747177': '412', 'n02749479': '413', 'n02769748': '414', 'n02776631': '415',
            'n02777292': '416', 'n02782093': '417', 'n02783161': '418', 'n02786058': '419', 'n02787622': '420',
            'n02788148': '421', 'n02790996': '422', 'n02791124': '423', 'n02791270': '424', 'n02793495': '425',
            'n02794156': '426', 'n02795169': '427', 'n02797295': '428', 'n02799071': '429', 'n02802426': '430',
            'n02804414': '431', 'n02804610': '432', 'n02807133': '433', 'n02808304': '434', 'n02808440': '435',
            'n02814533': '436', 'n02814860': '437', 'n02815834': '438', 'n02817516': '439', 'n02823428': '440',
            'n02823750': '441', 'n02825657': '442', 'n02834397': '443', 'n02835271': '444', 'n02837789': '445',
            'n02840245': '446', 'n02841315': '447', 'n02843684': '448', 'n02859443': '449', 'n02860847': '450',
            'n02865351': '451', 'n02869837': '452', 'n02870880': '453', 'n02871525': '454', 'n02877765': '455',
            'n02879718': '456', 'n02883205': '457', 'n02892201': '458', 'n02892767': '459', 'n02894605': '460',
            'n02895154': '461', 'n02906734': '462', 'n02909870': '463', 'n02910353': '464', 'n02916936': '465',
            'n02917067': '466', 'n02927161': '467', 'n02930766': '468', 'n02939185': '469', 'n02948072': '470',
            'n02950826': '471', 'n02951358': '472', 'n02951585': '473', 'n02963159': '474', 'n02965783': '475',
            'n02966193': '476', 'n02966687': '477', 'n02971356': '478', 'n02974003': '479', 'n02977058': '480',
            'n02978881': '481', 'n02979186': '482', 'n02980441': '483', 'n02981792': '484', 'n02988304': '485',
            'n02992211': '486', 'n02992529': '487', 'n02999410': '488', 'n03000134': '489', 'n03000247': '490',
            'n03000684': '491', 'n03014705': '492', 'n03016953': '493', 'n03017168': '494', 'n03018349': '495',
            'n03026506': '496', 'n03028079': '497', 'n03032252': '498', 'n03041632': '499', 'n03042490': '500',
            'n03045698': '501', 'n03047690': '502', 'n03062245': '503', 'n03063599': '504', 'n03063689': '505',
            'n03065424': '506', 'n03075370': '507', 'n03085013': '508', 'n03089624': '509', 'n03095699': '510',
            'n03100240': '511', 'n03109150': '512', 'n03110669': '513', 'n03124043': '514', 'n03124170': '515',
            'n03125729': '516', 'n03126707': '517', 'n03127747': '518', 'n03127925': '519', 'n03131574': '520',
            'n03133878': '521', 'n03134739': '522', 'n03141823': '523', 'n03146219': '524', 'n03160309': '525',
            'n03179701': '526', 'n03180011': '527', 'n03187595': '528', 'n03188531': '529', 'n03196217': '530',
            'n03197337': '531', 'n03201208': '532', 'n03207743': '533', 'n03207941': '534', 'n03208938': '535',
            'n03216828': '536', 'n03218198': '537', 'n03220513': '538', 'n03223299': '539', 'n03240683': '540',
            'n03249569': '541', 'n03250847': '542', 'n03255030': '543', 'n03259280': '544', 'n03271574': '545',
            'n03272010': '546', 'n03272562': '547', 'n03290653': '548', 'n03291819': '549', 'n03297495': '550',
            'n03314780': '551', 'n03325584': '552', 'n03337140': '553', 'n03344393': '554', 'n03345487': '555',
            'n03347037': '556', 'n03355925': '557', 'n03372029': '558', 'n03376595': '559', 'n03379051': '560',
            'n03384352': '561', 'n03388043': '562', 'n03388183': '563', 'n03388549': '564', 'n03393912': '565',
            'n03394916': '566', 'n03400231': '567', 'n03404251': '568', 'n03417042': '569', 'n03424325': '570',
            'n03425413': '571', 'n03443371': '572', 'n03444034': '573', 'n03445777': '574', 'n03445924': '575',
            'n03447447': '576', 'n03447721': '577', 'n03450230': '578', 'n03452741': '579', 'n03457902': '580',
            'n03459775': '581', 'n03461385': '582', 'n03467068': '583', 'n03476684': '584', 'n03476991': '585',
            'n03478589': '586', 'n03481172': '587', 'n03482405': '588', 'n03483316': '589', 'n03485407': '590',
            'n03485794': '591', 'n03492542': '592', 'n03494278': '593', 'n03495258': '594', 'n03496892': '595',
            'n03498962': '596', 'n03527444': '597', 'n03529860': '598', 'n03530642': '599', 'n03532672': '600',
            'n03534580': '601', 'n03535780': '602', 'n03538406': '603', 'n03544143': '604', 'n03584254': '605',
            'n03584829': '606', 'n03590841': '607', 'n03594734': '608', 'n03594945': '609', 'n03595614': '610',
            'n03598930': '611', 'n03599486': '612', 'n03602883': '613', 'n03617480': '614', 'n03623198': '615',
            'n03627232': '616', 'n03630383': '617', 'n03633091': '618', 'n03637318': '619', 'n03642806': '620',
            'n03649909': '621', 'n03657121': '622', 'n03658185': '623', 'n03661043': '624', 'n03662601': '625',
            'n03666591': '626', 'n03670208': '627', 'n03673027': '628', 'n03676483': '629', 'n03680355': '630',
            'n03690938': '631', 'n03691459': '632', 'n03692522': '633', 'n03697007': '634', 'n03706229': '635',
            'n03709823': '636', 'n03710193': '637', 'n03710637': '638', 'n03710721': '639', 'n03717622': '640',
            'n03720891': '641', 'n03721384': '642', 'n03724870': '643', 'n03729826': '644', 'n03733131': '645',
            'n03733281': '646', 'n03733805': '647', 'n03742115': '648', 'n03743016': '649', 'n03759954': '650',
            'n03761084': '651', 'n03763968': '652', 'n03764736': '653', 'n03769881': '654', 'n03770439': '655',
            'n03770679': '656', 'n03773504': '657', 'n03775071': '658', 'n03775546': '659', 'n03776460': '660',
            'n03777568': '661', 'n03777754': '662', 'n03781244': '663', 'n03782006': '664', 'n03785016': '665',
            'n03786901': '666', 'n03787032': '667', 'n03788195': '668', 'n03788365': '669', 'n03791053': '670',
            'n03792782': '671', 'n03792972': '672', 'n03793489': '673', 'n03794056': '674', 'n03796401': '675',
            'n03803284': '676', 'n03804744': '677', 'n03814639': '678', 'n03814906': '679', 'n03825788': '680',
            'n03832673': '681', 'n03837869': '682', 'n03838899': '683', 'n03840681': '684', 'n03841143': '685',
            'n03843555': '686', 'n03854065': '687', 'n03857828': '688', 'n03866082': '689', 'n03868242': '690',
            'n03868863': '691', 'n03871628': '692', 'n03873416': '693', 'n03874293': '694', 'n03874599': '695',
            'n03876231': '696', 'n03877472': '697', 'n03877845': '698', 'n03884397': '699', 'n03887697': '700',
            'n03888257': '701', 'n03888605': '702', 'n03891251': '703', 'n03891332': '704', 'n03895866': '705',
            'n03899768': '706', 'n03902125': '707', 'n03903868': '708', 'n03908618': '709', 'n03908714': '710',
            'n03916031': '711', 'n03920288': '712', 'n03924679': '713', 'n03929660': '714', 'n03929855': '715',
            'n03930313': '716', 'n03930630': '717', 'n03933933': '718', 'n03935335': '719', 'n03937543': '720',
            'n03938244': '721', 'n03942813': '722', 'n03944341': '723', 'n03947888': '724', 'n03950228': '725',
            'n03954731': '726', 'n03956157': '727', 'n03958227': '728', 'n03961711': '729', 'n03967562': '730',
            'n03970156': '731', 'n03976467': '732', 'n03976657': '733', 'n03977966': '734', 'n03980874': '735',
            'n03982430': '736', 'n03983396': '737', 'n03991062': '738', 'n03992509': '739', 'n03995372': '740',
            'n03998194': '741', 'n04004767': '742', 'n04005630': '743', 'n04008634': '744', 'n04009552': '745',
            'n04019541': '746', 'n04023962': '747', 'n04026417': '748', 'n04033901': '749', 'n04033995': '750',
            'n04037443': '751', 'n04039381': '752', 'n04040759': '753', 'n04041544': '754', 'n04044716': '755',
            'n04049303': '756', 'n04065272': '757', 'n04067472': '758', 'n04069434': '759', 'n04070727': '760',
            'n04074963': '761', 'n04081281': '762', 'n04086273': '763', 'n04090263': '764', 'n04099969': '765',
            'n04111531': '766', 'n04116512': '767', 'n04118538': '768', 'n04118776': '769', 'n04120489': '770',
            'n04125021': '771', 'n04127249': '772', 'n04131690': '773', 'n04133789': '774', 'n04136333': '775',
            'n04141076': '776', 'n04141327': '777', 'n04141975': '778', 'n04146614': '779', 'n04147183': '780',
            'n04149813': '781', 'n04152593': '782', 'n04153751': '783', 'n04154565': '784', 'n04162706': '785',
            'n04179913': '786', 'n04192698': '787', 'n04200800': '788', 'n04201297': '789', 'n04204238': '790',
            'n04204347': '791', 'n04208210': '792', 'n04209133': '793', 'n04209239': '794', 'n04228054': '795',
            'n04229816': '796', 'n04235860': '797', 'n04238763': '798', 'n04239074': '799', 'n04243546': '800',
            'n04251144': '801', 'n04252077': '802', 'n04252225': '803', 'n04254120': '804', 'n04254680': '805',
            'n04254777': '806', 'n04258138': '807', 'n04259630': '808', 'n04263257': '809', 'n04264628': '810',
            'n04265275': '811', 'n04266014': '812', 'n04270147': '813', 'n04273569': '814', 'n04275548': '815',
            'n04277352': '816', 'n04285008': '817', 'n04286575': '818', 'n04296562': '819', 'n04310018': '820',
            'n04311004': '821', 'n04311174': '822', 'n04317175': '823', 'n04325704': '824', 'n04326547': '825',
            'n04328186': '826', 'n04330267': '827', 'n04332243': '828', 'n04335435': '829', 'n04336792': '830',
            'n04344873': '831', 'n04346328': '832', 'n04347754': '833', 'n04350905': '834', 'n04355338': '835',
            'n04355933': '836', 'n04356056': '837', 'n04357314': '838', 'n04366367': '839', 'n04367480': '840',
            'n04370456': '841', 'n04371430': '842', 'n04371774': '843', 'n04372370': '844', 'n04376876': '845',
            'n04380533': '846', 'n04389033': '847', 'n04392985': '848', 'n04398044': '849', 'n04399382': '850',
            'n04404412': '851', 'n04409515': '852', 'n04417672': '853', 'n04418357': '854', 'n04423845': '855',
            'n04428191': '856', 'n04429376': '857', 'n04435653': '858', 'n04442312': '859', 'n04443257': '860',
            'n04447861': '861', 'n04456115': '862', 'n04458633': '863', 'n04461696': '864', 'n04462240': '865',
            'n04465501': '866', 'n04467665': '867', 'n04476259': '868', 'n04479046': '869', 'n04482393': '870',
            'n04483307': '871', 'n04485082': '872', 'n04486054': '873', 'n04487081': '874', 'n04487394': '875',
            'n04493381': '876', 'n04501370': '877', 'n04505470': '878', 'n04507155': '879', 'n04509417': '880',
            'n04515003': '881', 'n04517823': '882', 'n04522168': '883', 'n04523525': '884', 'n04525038': '885',
            'n04525305': '886', 'n04532106': '887', 'n04532670': '888', 'n04536866': '889', 'n04540053': '890',
            'n04542943': '891', 'n04548280': '892', 'n04548362': '893', 'n04550184': '894', 'n04552348': '895',
            'n04553703': '896', 'n04554684': '897', 'n04557648': '898', 'n04560804': '899', 'n04562935': '900',
            'n04579145': '901', 'n04579432': '902', 'n04584207': '903', 'n04589890': '904', 'n04590129': '905',
            'n04591157': '906', 'n04591713': '907', 'n04592741': '908', 'n04596742': '909', 'n04597913': '910',
            'n04599235': '911', 'n04604644': '912', 'n04606251': '913', 'n04612504': '914', 'n04613696': '915',
            'n06359193': '916', 'n06596364': '917', 'n06785654': '918', 'n06794110': '919', 'n06874185': '920',
            'n07248320': '921', 'n07565083': '922', 'n07579787': '923', 'n07583066': '924', 'n07584110': '925',
            'n07590611': '926', 'n07613480': '927', 'n07614500': '928', 'n07615774': '929', 'n07684084': '930',
            'n07693725': '931', 'n07695742': '932', 'n07697313': '933', 'n07697537': '934', 'n07711569': '935',
            'n07714571': '936', 'n07714990': '937', 'n07715103': '938', 'n07716358': '939', 'n07716906': '940',
            'n07717410': '941', 'n07717556': '942', 'n07718472': '943', 'n07718747': '944', 'n07720875': '945',
            'n07730033': '946', 'n07734744': '947', 'n07742313': '948', 'n07745940': '949', 'n07747607': '950',
            'n07749582': '951', 'n07753113': '952', 'n07753275': '953', 'n07753592': '954', 'n07754684': '955',
            'n07760859': '956', 'n07768694': '957', 'n07802026': '958', 'n07831146': '959', 'n07836838': '960',
            'n07860988': '961', 'n07871810': '962', 'n07873807': '963', 'n07875152': '964', 'n07880968': '965',
            'n07892512': '966', 'n07920052': '967', 'n07930864': '968', 'n07932039': '969', 'n09193705': '970',
            'n09229709': '971', 'n09246464': '972', 'n09256479': '973', 'n09288635': '974', 'n09332890': '975',
            'n09399592': '976', 'n09421951': '977', 'n09428293': '978', 'n09468604': '979', 'n09472597': '980',
            'n09835506': '981', 'n10148035': '982', 'n10565667': '983', 'n11879895': '984', 'n11939491': '985',
            'n12057211': '986', 'n12144580': '987', 'n12267677': '988', 'n12620546': '989', 'n12768682': '990',
            'n12985857': '991', 'n12998815': '992', 'n13037406': '993', 'n13040303': '994', 'n13044778': '995',
            'n13052670': '996', 'n13054560': '997', 'n13133613': '998', 'n15075141': '999'}
