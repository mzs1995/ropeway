import torch
import torch.nn.functional as F


def ropeway_tim(model, lossfn, x, y, kernel, g_te, alpha, N):
    for i in range(N+1):
        loss = lossfn(model(x), y)
        loss.backward()
        translated_grad = F.conv2d(x.grad, kernel, padding='same', groups=3)
        g_te = g_te + translated_grad
        x.data = x.data + alpha * torch.sign(translated_grad)
        x.grad.zero_()
    return g_te


def r_tim(model, x, y, kernel, eps=(32./255), mu=1.0, T=10, N=10, clip_min=-1.0, clip_max=1.0):
    # initialize
    lossfn = torch.nn.CrossEntropyLoss()
    alpha = eps / T
    momentum = 0.
    adv = x.detach().clone().requires_grad_(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(T):
        # compute direction with ropeway and TIM
        adv_te = adv.detach().clone().requires_grad_(True)
        g_te = torch.zeros(adv.data.shape).to(device)
        bias = ropeway_tim(model, lossfn, adv_te, y, kernel, g_te, alpha, N)/ (N + 1)
        # compute adv with momentum
        momentum = mu * momentum + bias / torch.norm(bias, p=1)
        adv.data = adv.data + alpha * torch.sign(momentum)
        adv.data = torch.clamp(adv.data, clip_min, clip_max)
    return adv.detach(), N

