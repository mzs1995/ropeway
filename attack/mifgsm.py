import torch


def mifgsm(model, x, y, eps=32./255, mu=1.0, T=10, clip_min=-1.0, clip_max=1.0):
    # initialization
    lossfn = torch.nn.CrossEntropyLoss()
    alpha = eps / T
    momentum = 0.
    adv = x.detach().clone().requires_grad_(True)
    # enter loop to generate adv
    for i in range(T):
        loss = lossfn(model(adv), y)
        loss.backward()
        # compute momentum
        momentum = mu * momentum + adv.grad / torch.norm(adv.grad, p=1)
        # compute adv with momentum
        adv.data = adv.data + alpha * torch.sign(momentum)
        adv.data = torch.clamp(adv.data, clip_min, clip_max)
        adv.grad.zero_()
    return adv.detach(), "MI-FGSM", 0
