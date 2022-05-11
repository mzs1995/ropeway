import torch
import torch.nn.functional as F


def tim(model, x, y, kernel, eps=32./255, mu=1.0, T=10, clip_min=-1.0, clip_max=1.0, ):
    # initialize
    lossfn = torch.nn.CrossEntropyLoss()
    alpha = eps / T
    momentum = 0.
    adv = x.detach().clone().requires_grad_(True)
    # enter loop to generate adv
    for i in range(T):
        loss = lossfn(model(adv), y)
        loss.backward()
        # compute translated gradient
        translated_grad = F.conv2d(adv.grad, kernel, padding='same', groups=3)
        # compute adv with momentum
        momentum = mu * momentum + translated_grad / torch.norm(translated_grad, p=1)
        adv.data = adv.data + alpha * torch.sign(momentum)
        adv.data = torch.clamp(adv.data, clip_min, clip_max)
        adv.grad.zero_()
    return adv.detach(), 0
