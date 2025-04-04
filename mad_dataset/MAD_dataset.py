import os
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from skimage import transform
import json
from torch.utils.data import Dataset
from typing import Any
from PIL import Image

mean, std = (0.5,), (0.5,)
digit_options = list(map(str, range(10)))
thickthinning_options = ["no thickthinning", "thickening", "thinning"]
swelling_options = ["no swelling", "swelling"]
fracture_options = ["no fracture", "fracture"]
scaling_options = ["large", "small"]
color_options = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"]

class MADDataset(Dataset):
    digits = list(map(str, range(10)))
    thickthinning = ["no thickthinning", "thickening", "thinning"]
    swelling = ["no swelling", "swelling"]
    fracture = ["no fracture", "fracture"]
    scaling = ["large", "small"]
    colors = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"]

    def __init__(self, root, transform, mode: str = "object_only", main_object: str= "digit", 
                 return_att_mat: bool = False, train: bool = True, 
                 return_att_mat_with_cls: bool = False, debug_visu: bool = False) -> None:
        super().__init__()

        self.follow_json = False
        if not train:
            self.follow_json = True

        with open(os.path.join(root, "train.json" if train else "test.json"), "r") as f:
            self.json_data = json.load(f)
        self.image_data = np.memmap(os.path.join(root, "train.mmap" if train else "test.mmap"), 
                                    dtype=np.uint8, mode="r", shape=(len(self.json_data), 28, 28))

        self.json_data = {int(k): v for k,v in self.json_data.items()}

        if train:
            self.block_size = len(self.json_data) // 60000
        else:
            self.block_size = len(self.json_data) // 10000
        self.transform = transform
        self.mode = mode
        self.train = train
        self.return_att_mat = return_att_mat
        self.return_att_mat_with_cls = return_att_mat_with_cls
        self.object = main_object
        self.debug_visu = debug_visu

    def __getitem__(self, index) -> Any:
        if not self.follow_json:
            idx = random.choice(range(self.block_size))
            moved_idx = index * self.block_size + idx
        else:
            moved_idx = index
        pregenerated_dict = self.json_data[moved_idx]

        image = np.array(self.image_data[pregenerated_dict["mmap_idx"]])
        cls_label = pregenerated_dict["cls_label"]
        choice1 = pregenerated_dict["thickthinnig"]
        choice2 = pregenerated_dict["swelling"]
        choice3 = pregenerated_dict["fracture"]
        choice4 = pregenerated_dict["scaling"]
        nr_classes = 10

        attribute_list = []
        # thickthinning
        if choice1 == 'no thickthinning':
            attribute_list.append(0)
        if choice1 == "thickening":
            attribute_list.append(1)
        elif choice1 == "thinning":
            attribute_list.append(2)
        if self.object == 'thickthinning':
            drop_idx = 0
            new_cls_label = attribute_list[-1]
            nr_classes = 3
        # swelling
        if choice2 =='no swelling':
            attribute_list.append(3)
        if choice2 == "swelling":
            attribute_list.append(4)
        if self.object == 'swelling':
            drop_idx = 1
            new_cls_label = attribute_list[-1]
            nr_classes = 2
        # facture
        if choice3 == 'no fracture':
            attribute_list.append(5)
        if choice3 == "fracture":
            attribute_list.append(6)
        if self.object == 'fracture':
            drop_idx = 2
            new_cls_label = attribute_list[-1]
            nr_classes = 2
        # scaling
        if choice4 == "large":
            attribute_list.append(7)
        if choice4 == "small":
            attribute_list.append(8)
        if self.object == 'scaling':
            drop_idx = 3
            new_cls_label = attribute_list[-1]
            nr_classes = 2
        pil_img = Image.fromarray(image).convert("RGB")

        # colorization
        colors = color_options
        if "color" in pregenerated_dict:
            choice5 = pregenerated_dict["color"]
        else:
            choice5 = random.choice(color_options)
        attribute_list.append(np.where([c==choice5 for c in colors])[0][0] + 9)
        if self.object == 'colors':
            drop_idx = 4
            new_cls_label = attribute_list[-1]
            nr_classes = len(colors)

        pil_img = self.colorize(pil_img, choice5)

        if self.transform is not None:
            image = self.transform(pil_img)

        if self.object == "digit":
            text = self.generate_text(cls_label, choice1, choice2, choice3, choice4, choice5)
        elif self.object == "thickthinning":
            text = self.generate_text(choice1, str(cls_label), choice2, choice3, choice4, choice5)
        elif self.object == "swelling":
            text = self.generate_text(choice2, str(cls_label), choice1, choice3, choice4, choice5)
        elif self.object == "fracture":
            text = self.generate_text(choice3, str(cls_label), choice1, choice2, choice4, choice5)
        elif self.object == "scaling":
            text = self.generate_text(choice4, str(cls_label), choice1, choice2, choice3, choice5)
        elif self.object == "colors":
            text = self.generate_text(choice5, str(cls_label), choice1, choice2, choice3, choice4)
        else:
            raise NotImplementedError

        if self.debug_visu:
            save_text = self.generate_text_save(cls_label, choice1, choice2, choice3, choice4, choice5)
            os.makedirs('./debug_imgs', exist_ok=True)
            pil_img.save(os.path.join('./debug_imgs', save_text + '.jpg'))

        if self.return_att_mat:
            if not self.object == 'digit':
                attribute_list.append(cls_label+9+len(colors)) #highest number for attributes, i.e. 9-19 are class labels now
                attribute_list[drop_idx:] = list(np.array(attribute_list[drop_idx:]) - nr_classes)
                attribute_list.pop(drop_idx)
                cls_label = new_cls_label

            if self.return_att_mat_with_cls:
                att_mat, att_mat_bool = self.get_att_mat(attribute_list)
                _, att_mat_bool_with_cls = self.get_att_mat(attribute_list, include_boolean_class_label=True, cls_label=cls_label)

                return image, text, att_mat, att_mat_bool, att_mat_bool_with_cls, cls_label
            elif self.return_att_mat:
                att_mat, att_mat_bool = self.get_att_mat(attribute_list, nr_classes=nr_classes)
                return image, text, att_mat, att_mat_bool, cls_label

        return image, text


    def __len__(self) -> int:
        if self.follow_json:
            return len(self.json_data)
        return len(self.json_data) // self.block_size

    def get_att_mat(self, attribute_list, include_boolean_class_label=False, cls_label=None, nr_classes=0):
        att_mat = np.ones((1,15)) * -1
        att_mat[:,:len(attribute_list)] = attribute_list
        if not include_boolean_class_label:
            if not self.object == 'digit':
                att_mat_bool = np.zeros((1, 26 - nr_classes))
            else:
                att_mat_bool = np.zeros((1, 16))
            att_mat_bool[:, attribute_list] = 1
        else:
            if not self.object == 'digit':
                raise NotImplementedError
            nr_classes = 10
            nr_rows = 16 + nr_classes #
            att_mat_bool = np.zeros((1, nr_rows))
            att_mat_bool[0,cls_label] = 1
            att_mat_bool[:, np.array(attribute_list)+nr_classes] = 1 

        return att_mat, att_mat_bool

    def rescale(self, image: np.ndarray) -> np.ndarray:
        new_image = np.zeros_like(image)
        image_rescaled = (transform.rescale(image.astype(np.float32)/255, 0.75, anti_aliasing=False)*255).astype(np.uint8)
        y = random.choice([0, 1])
        x = random.choice([0, 1])
        new_image[3 if y == 0 else 4:-3 if y==1 else -4, 3 if x == 0 else 4:-3 if x==1 else -4] = image_rescaled
        return new_image

    def colorize(self, image: Image, color: str) -> np.ndarray:
        if color == "gray":
            return image
        image = np.array(image)
        mask = image[...,0] > 0
        if color == "blue":
            image[mask,0] = 0
            image[mask,1] = 0
        elif color == "green":
            image[mask,0] = 0
            image[mask,2] = 0
        elif color == "red":
            image[mask,1] = 0
            image[mask,2] = 0
        elif color == "yellow":
            image[mask,0] = 255
            image[mask,1] = 255
        elif color == "magenta":
            image[mask,0] = 255
            image[mask,2] = 255
        elif color == "cyan":
            image[mask,1] = 255
            image[mask,2] = 255
        else:
            raise NotImplementedError
        return Image.fromarray(image)

    def generate_text(self, cls_label: int, *args):
        if self.mode == "object_only":
            caption = str(cls_label)
        elif "object_attribute_" in self.mode:
            string_content = [str(cls_label)]
            string_content += random.sample(args, int(self.mode[-1]))
            random.shuffle(string_content)
            caption = "-".join(string_content)
        else:
            raise NotImplementedError
        return caption

    def generate_text_save(self, cls_label: int, *args):
        if self.mode == "object_only":
            caption = str(cls_label)
        elif "object_attribute_" in self.mode:
            caption = str(cls_label) + '-' + '-'.join(args)
        else:
            raise NotImplementedError
        return caption


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                            ])
    dataset = MADDataset(root='./datasets/morphoMNIST_0', transform=transform, mode="object_attribute_1",
                         train=True, debug_visu=True)
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    for batch in tqdm.tqdm(train_loader, total=len(train_loader)):
        continue
