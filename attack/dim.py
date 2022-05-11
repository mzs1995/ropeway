import torch
from torchvision.transforms import Resize
from torch.nn import ConstantPad2d


def transform_with_p(x, p):
    nums = torch.rand(1)[0].item()
    if nums < p:
        # firstly, the adv is resized to [3, rnd, rnd], and [rnd] is random value in [299, 300)
        size = torch.randint(299, 330, (1, 1))[0].item()
        resize = Resize([size, size])
        x = resize(x)
        # secondly, the adv is padded to [3, 330, 330] in a random manner
        final_size = 330 - size
        pad_top = torch.randint(0, final_size + 1, (1, 1))[0].item()
        pad_bottom = final_size - pad_top
        pad_left = torch.randint(0, final_size + 1, (1, 1))[0].item()
        pad_right = final_size - pad_left
        pad = ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.0)
        x = pad(x)
    return x


# DIM has already combined with MI-FGSM
def dim(model, x, y, eps=32./255, T=10, p=0.5, mu=1.0, clip_min=-1.0, clip_max=1.0):
    # initialize
    lossfn = torch.nn.CrossEntropyLoss()
    alpha = eps / T
    momentum = 0.
    adv = x.detach().clone().requires_grad_(True)
    # enter loop to generate adv
    for i in range(T):
        # compute loss where adv is transformed with p
        loss = lossfn(model(transform_with_p(adv, p)), y)
        loss.backward()
        # compute adv with momentum
        momentum = mu * momentum + adv.grad / torch.norm(adv.grad, p=1)
        adv.data = adv.data + alpha * torch.sign(momentum)
        adv.data = torch.clamp(adv.data, clip_min, clip_max)
        adv.grad.zero_()
    return adv.detach(), 0