import torch


def ropeway_sim(model, lossfn, x, y, g_te, g_agg_sim, alpha, nums, N):
    for i in range(N + 1):
        for j in range(nums):
            loss = lossfn(model(x / (2**j)), y)
            loss.backward()
            g_agg_sim = g_agg_sim + x.grad
            x.grad.zero_()
        g = g_agg_sim / nums
        g_te = g_te + g
        g_agg_sim.zero_()
        x.data = x.data + alpha * torch.sign(g)
    return g_te


def r_sim(model, x, y, eps=32./255, mu=1., T=10, nums=5, N=10, clip_min=-1., clip_max=1.):
    # initialize
    lossfn = torch.nn.CrossEntropyLoss()
    alpha = eps / T
    momentum = 0.
    adv = x.detach().clone().requires_grad_(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_agg_sim = torch.zeros(adv.data.shape).to(device)
    for i in range(T):
        adv_te = adv.detach().clone().requires_grad_(True)
        g_te = torch.zeros(adv.data.shape).to(device)
        # compute direction with ropeway and SIM
        bias = ropeway_sim(model, lossfn, adv_te, y, g_te, g_agg_sim, alpha, nums, N) / (N + 1)
        # compute adv with momentum
        momentum = mu * momentum + bias / torch.norm(bias, p=1)
        adv.data = adv.data + alpha * torch.sign(momentum)
        adv.data = torch.clamp(adv.data, clip_min, clip_max)
    return adv.detach(), N

