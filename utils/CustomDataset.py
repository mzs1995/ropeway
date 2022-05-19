import csv
import os
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset1(Dataset):
    def __init__(self, img_dir, label_path, transform=None):
        self.img_dir = img_dir
        self.all_img_names = []
        self.img_name_to_label = {}
        self.load_img_names(label_path)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.all_img_names[idx])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img.convert('RGB'))
        else:
            img = img.convert('RGB')
        label = self.img_name_to_label[self.all_img_names[idx]]
        return img, label, self.all_img_names[idx]

    def __len__(self):
        return len(self.all_img_names)

    def load_img_names(self, img_name_dir):
        lines = csv.reader(open(img_name_dir, "r"))
        for oneline in lines:
            if oneline[0].startswith("I"):
                self.all_img_names.append(oneline[0])
                self.img_name_to_label[oneline[0]] = int(oneline[1]) - 1


class CustomDataset2(Dataset):
    def __init__(self, adv_dir, sm_label_path, tm_label_path, transform=None):
        self.adv_dir = adv_dir
        self.all_img_names = []
        self.img_name_to_label = {}
        self.load_name_and_label(sm_label_path, tm_label_path)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.adv_dir, self.all_img_names[idx])
        img = Image.open(img_path)
        img = self.transform(img.convert('RGB'))
        label = self.img_name_to_label[self.all_img_names[idx]]
        return img, label, self.all_img_names[idx]

    def __len__(self):
        return len(self.all_img_names)

    def load_name_and_label(self, sm_label_path, tm_label_path):
        sm_all_img_names = []
        tm_all_img_names = []
        sm_img_name_to_label = {}

        with open(sm_label_path, "r") as f:
            allLine = f.readlines()
            for one in allLine:
                img_name = one.split(" ")[0]
                label = one.split(" ")[1].rstrip("\n")
                sm_all_img_names.append(img_name)
                sm_img_name_to_label[img_name] = int(label)

        with open(tm_label_path, "r") as f:
            allLine = f.readlines()
            for one in allLine:
                img_name = one.split(" ")[0]
                tm_all_img_names.append(img_name)

        self.all_img_names = [i for i in sm_all_img_names if i in tm_all_img_names]
        for onename in self.all_img_names:
            self.img_name_to_label[onename] = sm_img_name_to_label[onename]