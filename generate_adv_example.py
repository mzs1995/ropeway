import os.path
import argparse
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
from attack.r_dim import r_dim
from attack.r_sim import r_sim
from attack.r_tim import r_tim
from attack.r_di_admix import r_di_admix


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

    dataset = CustomDataset1(img_dir, label_path, transform)
    dataloader = DataLoader(dataset, batch_size=batchsize)

    to_pil_img = transforms.ToPILImage(mode="RGB")

    translated_kernel = gkern(kernlen=7)
    translated_kernel = translated_kernel.to(device)

    count = 0
    for img, tag, name in dataloader:

        torch.manual_seed(0)
        count = count + len(tag)
        img = img.to(device) * 2 - 1.
        tag = tag.to(device)
        source_model.eval()

        # attack
        if attack == "DIM":
            img_adv, N = dim(source_model, img, tag)
        elif attack == "R-DIM":
            img_adv, N = r_dim(source_model, img, tag)
        elif attack == "SIM":
            img_adv, N = sim(source_model, img, tag)
        elif attack == "R-SIM":
            img_adv, N = r_sim(source_model, img, tag)
        elif attack == "TIM":
            img_adv, N = tim(source_model, img, tag, translated_kernel)
        elif attack == "R-TIM":
            img_adv, N = r_tim(source_model, img, tag, translated_kernel)
        elif attack == "DI-Admix":
            img_adv, N = di_admix(source_model, img, tag, transform, img_dir, label_path)
        elif attack == "R-DI-Admix":
            img_adv, N = r_di_admix(source_model, img, tag, transform, img_dir, label_path)
        else:
            print("Attack name is wrong.")
            break

        for j in range(len(img_adv)):
            imgs_rgb = TensorToImg(img_adv[j], to_pil_img)
            path = os.path.join(storage_adv_dir, name[j])
            imgs_rgb.save(path)

        print("\r", "Attack:[{}] || n in Ropeway:[{}] || Generate AEs from [{}] || No.{}"
              .format(attack, N, source_model_name, count), end="", flush=True)
    print("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial examples")
    parser.add_argument("--attack", default="R-DIM", type=str, help="Attack name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    attack = args.attack
    batchsize = 20
    storage_adv_dir = "storage/adv"
    img_dir = "storage/imgs"
    label_path = "storage/label/sim_dataset.csv"
    generate(attack, batchsize, storage_adv_dir, img_dir, label_path)
