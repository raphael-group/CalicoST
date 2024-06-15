import numpy as np
import scipy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    device = 'cpu'


def fit_weighted_NegativeBinomial_gpu(y, features, weights, exposure, start_log_mu=None, start_alpha=None, n_epochs=1000):
    # convert to torch
    y = torch.from_numpy(y).to(torch.float32).to(device)
    features = torch.from_numpy(features).to(torch.float32).to(device)
    weights = torch.from_numpy(weights).to(torch.float32).to(device)
    exposure = torch.from_numpy(exposure).to(torch.float32).to(device)

    # train
    if start_log_mu is None:
        log_mu = nn.Parameter(torch.zeros(features.shape[1], device=device), requires_grad=True)
    else:
        log_mu = nn.Parameter(torch.from_numpy(start_log_mu.flatten()).to(torch.float32).to(device), requires_grad=True)
    if start_alpha is None:
        log_disp = nn.Parameter(torch.log(0.1 * torch.ones(1, device=device)), requires_grad=True)
    else:
        log_disp = nn.Parameter(torch.log(start_alpha * torch.ones(1, device=device)), requires_grad=True)
    
    loss_list = []
    optimizer = torch.optim.AdamW( [log_mu, log_disp], lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[350,700], gamma=0.1)
    for epoch in range(int(n_epochs / 20)):
        small_loss = []
        for small_epoch in range(20):
            optimizer.zero_grad()
            # negative binomial llf
            nb_mean = torch.exp(features @ log_mu) * exposure
            nb_var = nb_mean + nb_mean**2 * torch.exp(log_disp)
            # convert parameters
            p = 1.0 / (1.0 + nb_mean * torch.exp(log_disp))
            n = 1.0 / torch.exp(log_disp)
            llf = torch.distributions.negative_binomial.NegativeBinomial(n, 1-p).log_prob(y)
            loss = -torch.matmul(llf, weights)
            small_loss.append( loss.item() )
            loss.backward()
            optimizer.step()
            scheduler.step()
        loss_list.append( np.mean(small_loss) )
        # decide to terminate
        if len(loss_list) > 2 and np.abs(loss_list[-1] - np.min(loss_list[:-1])) < 1e-6 * len(y):
            break

    res = log_mu.detach().cpu().numpy().reshape(-1,1)
    res = np.append(res, torch.exp(log_disp).detach().cpu().numpy()[0])
    return res, loss_list[-1]


def fit_weighted_BetaBinomial_gpu(y, features, weights, exposure, start_p_binom=None, start_tau=None, min_binom_prob=0.01, max_binom_prob=0.99, 
                      n_epochs=1000, MIN_TAUS = np.log(5), MAX_TAUS = np.log(1000)):
    y = torch.from_numpy(y).to(torch.float32).to(device)
    features = torch.from_numpy(features).to(torch.float32).to(device)
    weights = torch.from_numpy(weights).to(torch.float32).to(device)
    exposure = torch.from_numpy(exposure).to(torch.float32).to(device)

    # initialize training parameters
    if start_p_binom is None:
        logistic_p = nn.Parameter(torch.logit(0.3 * torch.ones(features.shape[1], device=device)), requires_grad=True)
    else:
        logistic_p = nn.Parameter(torch.logit(torch.from_numpy(start_p_binom.flatten()).to(torch.float32).to(device)), requires_grad=True)
    if start_tau is None:
        log_taus = nn.Parameter(torch.log(20 * torch.ones(1, device=device)), requires_grad=True)
    else:
        log_taus = nn.Parameter(torch.log(start_tau * torch.ones(1, device=device)), requires_grad=True)

    MIN_LOGISTIC_P = scipy.special.logit(min_binom_prob)
    MAX_LOGISTIC_P = scipy.special.logit(max_binom_prob)
    loss_list = []
    optimizer = torch.optim.AdamW( [logistic_p, log_taus], lr=5e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[350,700], gamma=0.1)
    for epoch in range(int(n_epochs / 20)):
        small_loss = []
        for small_epoch in range(20):
            optimizer.zero_grad()
            # beta binomial llf
            p = F.sigmoid(logistic_p)
            d_choose_y = (exposure + 1).lgamma() - (y + 1).lgamma() - ((exposure - y) + 1).lgamma()
            logbeta_numer = (y + (features @ p) * torch.exp(log_taus)).lgamma() + (exposure - y + (1 - features @ p) * torch.exp(log_taus)).lgamma() - (exposure + torch.exp(log_taus)).lgamma() 
            logbeta_denom = ((features @ p) * torch.exp(log_taus)).lgamma() + ((1 - features @ p) * torch.exp(log_taus)).lgamma() - torch.exp(log_taus).lgamma() 
            llf = d_choose_y + logbeta_numer - logbeta_denom
            loss = -torch.matmul(llf, weights)
            small_loss.append( loss.item() )
            loss.backward()
            log_taus.data = torch.clamp(log_taus.data, min=MIN_TAUS, max=MAX_TAUS)
            logistic_p.data = torch.clamp(logistic_p.data, min=MIN_LOGISTIC_P, max=MAX_LOGISTIC_P)
            optimizer.step()
            scheduler.step()
        loss_list.append( np.mean(small_loss) )
        if len(loss_list) > 2 and np.abs(loss_list[-1] - np.min(loss_list[:-1])) < 1e-6 * len(y):
            break

    res = F.sigmoid(logistic_p).detach().cpu().numpy().reshape(-1,1)
    res = np.append(res, torch.exp(log_taus).detach().cpu().numpy()[0])
    return res, loss_list[-1]