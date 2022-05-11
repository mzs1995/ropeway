import csv
import os
import torch
from PIL import Image
from torchvision.transforms import Resize
from torch.nn import ConstantPad2d


def transform_with_p(x, p):
    nums = torch.rand(1)[0].item()
    if nums < p:
        size = torch.randint(299, 330, (1, 1))[0].item()
        resize = Resize([size, size])
        x = resize(x)
        final_size = 330 - size
        pad_top = torch.randint(0, final_size + 1, (1, 1))[0].item()
        pad_bottom = final_size - pad_top
        pad_left = torch.randint(0, final_size + 1, (1, 1))[0].item()
        pad_right = final_size - pad_left
        pad = ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.0)
        x = pad(x)
    return x


def ropeway_di_admix(model, lossfn, adv_te, y, g_te, g_sum_di_admix, alpha, N,
                       m1, m2, eta, transform, img_dir, label_path, device):
    for i in range(N+1):
        imgs_sampled = []
        get_m2_random_imgs(m2, y.item(), transform, imgs_sampled, img_dir, label_path)
        for img in imgs_sampled:
            img = img.to(device) * 2 - 1.
            for j in range(m1):
                gamma = 1 / (2**j)
                loss = lossfn(model(transform_with_p(gamma * (adv_te + eta * img), 0.5)), y)
                loss.backward()
                g_sum_di_admix = g_sum_di_admix + adv_te.grad
                adv_te.grad.zero_()
        g_ave = g_sum_di_admix / (m1 * m2)
        g_te = g_te + g_ave
        g_sum_di_admix.zero_()
        adv_te.data = adv_te.data + alpha * torch.sign(g_ave)
    return g_te


def get_m2_random_img_name(m2, tag, all_img_names, img_name_to_label, names):
    for i in range(m2):
        lab = tag
        num = None
        while lab == tag:
            num = torch.randint(0, 1000, (1, 1)).item()
            lab = int(img_name_to_label[all_img_names[num]])
        names.append(all_img_names[num])


def load_img_names(img_name_dir, all_img_names, img_name_to_label):
    lines = csv.reader(open(img_name_dir, "r"))
    for oneline in lines:
        if oneline[0].startswith("I"):
            all_img_names.append(oneline[0])
            img_name_to_label[oneline[0]] = int(oneline[1]) - 1


def get_m2_random_imgs(m2, tag, transform, imgs_sampled, img_dir, label_path):
    all_img_names = []
    img_name_to_label = {}
    load_img_names(label_path, all_img_names, img_name_to_label)
    names = []
    get_m2_random_img_name(m2, tag, all_img_names, img_name_to_label, names)
    for onename in names:
        img_path = os.path.join(img_dir, onename)
        img = Image.open(img_path)
        img = torch.unsqueeze(transform(img.convert('RGB')), 0)
        imgs_sampled.append(img)


def re_di_admix(model, x, y, transform, img_dir, label_path, eps=32./255, mu=1.0, T=10,
                m1=5, m2=3, eta=0.2, N=10, clip_min=-1.0, clip_max=1.0):
    # initialize
    lossfn = torch.nn.CrossEntropyLoss()
    alpha = eps / T
    momentum = 0.
    adv = x.detach().clone().requires_grad_(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_sum_di_admix = torch.zeros(adv.data.shape, dtype=torch.float).to(device)
    # enter loop to generate adv
    for i in range(T):
        # compute direction with ropeway and DI-Admix
        adv_te = adv.detach().clone().requires_grad_(True)
        g_te = torch.zeros(adv.data.shape).to(device)
        bias = ropeway_di_admix(model, lossfn, adv_te, y, g_te, g_sum_di_admix, alpha, N,
                            m1, m2, eta, transform, img_dir, label_path, device) / (N + 1)
        # compute adv with momentum
        momentum = mu * momentum + bias / torch.norm(bias, p=1)
        adv.data = adv.data + alpha * torch.sign(momentum)
        adv.data = torch.clamp(adv.data, clip_min, clip_max)
    return adv.detach(), N
