import os.path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy.stats as st
import numpy as np
import pretrainedmodels

from utils.CustomDataset import CustomDataset1
from utils.kit import TensorToImg
from attack.dim import dim
from attack.sim import sim
from attack.tim import tim
from attack.di_admix import di_admix
from attack.re_dim import re_dim
from attack.re_sim import re_sim
from attack.re_tim import re_tim
from attack.re_di_admix import re_di_admix


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel_np = (kernel_raw / kernel_raw.sum()).astype(np.float32)
    kernel_tensor = torch.unsqueeze(torch.tensor(kernel_np), dim=0)
    stack_kernel = torch.stack([kernel_tensor, kernel_tensor, kernel_tensor])
    return stack_kernel

def generate(attack, batchsize, storage_adv_dir, img_dir, label_path):
    # torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_model_name = "inceptionv3"
    source_model = pretrainedmodels.__dict__["inceptionv3"](num_classes=1000, pretrained='imagenet')
    source_model.to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    if attack.__contains__("DI-Admix"):
        batchsize = 1
    dataset = CustomDataset1(img_dir, label_path, transform)
    dataloader = DataLoader(dataset, batch_size=batchsize)

    to_pil_img = transforms.ToPILImage(mode="RGB")

    translated_kernel = gkern(kernlen=7)
    translated_kernel = translated_kernel.to(device)

    count = 0
    for img, tag, name in dataloader:

        torch.manual_seed(0)
        count = count + batchsize
        img = img.to(device) * 2 - 1.
        tag = tag.to(device)
        source_model.eval()

        # attack
        if attack == "DIM":
            img_adv, N = dim(source_model, img, tag)
        elif attack == "Re-DIM":
            img_adv, N = re_dim(source_model, img, tag)
        elif attack == "SIM":
            img_adv, N = sim(source_model, img, tag)
        elif attack == "Re-SIM":
            img_adv, N = re_sim(source_model, img, tag)
        elif attack == "TIM":
            img_adv, N = tim(source_model, img, tag, translated_kernel)
        elif attack == "Re-TIM":
            img_adv, N = re_tim(source_model, img, tag, translated_kernel)
        elif attack == "DI-Admix":
            img_adv, N = di_admix(source_model, img, tag, transform, img_dir, label_path)
        elif attack == "Re-DI-Admix":
            img_adv, N = re_di_admix(source_model, img, tag, transform, img_dir, label_path)
        else:
            print("Attack name is wrong.")
            break

        for j in range(batchsize):
            imgs_rgb = TensorToImg(img_adv[j], to_pil_img)
            path = os.path.join(storage_adv_dir, name[j])
            imgs_rgb.save(path)

        print("\r", "Attack:[{}] || n in Ropeway:[{}] || Generate AEs from [{}] || No.{}"
              .format(attack, N, source_model_name, count), end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # attack = "DIM"
    attack = "Re-DIM"
    # attack = "SIM"
    # attack = "Re-SIM"
    # attack = "TIM"
    # attack = "Re-TIM"
    # attack = "DI-Admix"
    # attack = "Re-DI-Admix"
    batchsize = 50
    storage_adv_dir = "storage/adv"
    img_dir = "storage/imgs"
    label_path = "storage/label/sim_dataset.csv"
    generate(attack, batchsize, storage_adv_dir, img_dir, label_path)
