import torch
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


def ropeway_dim(model, lossfn, x, y, grad_te, alpha, p, N):
    for i in range(N + 1):
        loss = lossfn(model(transform_with_p(x, p)), y)
        loss.backward()
        grad_te = grad_te + x.grad
        x.data = x.data + alpha * torch.sign(x.grad)
        x.grad.zero_()
    return grad_te


def r_dim(model, x, y, eps=32./255, T=10, p=0.5, mu=1.0, N=10, clip_min=-1.0, clip_max=1.0):
    # initialize
    lossfn = torch.nn.CrossEntropyLoss()
    alpha = eps / T
    momentum = 0.
    adv = x.detach().clone().requires_grad_(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # enter loop to generate adv
    for i in range(T):
        # compute direction with ropeway and DIM
        adv_te = adv.detach().clone().requires_grad_(True)
        g_te = torch.zeros(adv.data.shape).to(device)
        bias = ropeway_dim(model, lossfn, adv_te, y, g_te, alpha, p, N) / (N + 1)
        # compute adv with momentum
        momentum = mu * momentum + bias / torch.norm(bias, p=1)
        adv.data = adv.data + alpha * torch.sign(momentum)
        adv.data = torch.clamp(adv.data, clip_min, clip_max)
    return adv.detach(), N