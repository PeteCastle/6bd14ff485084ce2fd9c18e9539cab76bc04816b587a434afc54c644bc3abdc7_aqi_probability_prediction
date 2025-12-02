import torch


def mdn_loss(y, mu, sigma, alpha, num_mixtures):
    batch_size = y.size(0)
    output_dim = mu.size(1) // num_mixtures

    mu = mu.view(batch_size, num_mixtures, output_dim)
    sigma = sigma.view(batch_size, num_mixtures, output_dim)
    alpha = alpha.view(batch_size, num_mixtures)

    y = y.unsqueeze(1).expand(-1, num_mixtures, -1)
    # print(f"y shape: {y.shape}, mu shape: {mu.shape}, sigma shape: {sigma.shape}, alpha shape: {alpha.shape}")
    # print("First pollutant over time (mu[:, 0, 0]):", mu[:, 0, 0])
    # print("All mixtures for time step 0 and pollutant 0 (mux[0, :, 0]):", mu[0, :, 0])
    # print("All pollutants at t=0, mixture=0 (mu[0, 0, :]):", mu[0, 0, :])

    # Gaussian log-likelihood per mixture
    normal = torch.distributions.Normal(loc=mu, scale=sigma)
    log_probs = normal.log_prob(y)  # [batch, num_mixtures, output_dim]
    log_probs = torch.sum(log_probs, dim=2)  # sum over output_dim

    # Weighted log likelihood
    weighted_log_probs = log_probs + torch.log(alpha + 1e-8)
    log_sum_exp = torch.logsumexp(weighted_log_probs, dim=1)
    return -torch.mean(log_sum_exp)
