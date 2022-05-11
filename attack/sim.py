import torch


# SIM has already combined with MI-FGSM
def sim(model, x, y, eps=32./255, mu=1., T=10, nums=5, clip_min=-1., clip_max=1.):
    # initialize
    lossfn = torch.nn.CrossEntropyLoss()
    alpha = eps / T
    momentum = 0.
    adv = x.detach().clone().requires_grad_(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # enter loop to generate adv
    for i in range(T):
        g_agg = torch.zeros(adv.data.shape).to(device)
        # obtain the aggregated gradient
        for j in range(nums):
            adv_agg = adv.detach().clone().requires_grad_(True)
            # scale inputs
            loss = lossfn(model(adv_agg / (2**j)), y)
            loss.backward()
            g_agg = g_agg + adv_agg.grad
        g = g_agg / nums
        # compute adv with momentum
        momentum = mu * momentum + g / torch.norm(g, p=1)
        adv.data = adv.data + alpha * torch.sign(momentum)
        adv.data = torch.clamp(adv.data, clip_min, clip_max)
    return adv.detach(), 0

