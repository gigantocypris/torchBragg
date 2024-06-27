"""
Compare the Poisson distribution for various lambda parameters to 
approximations from Gaussian and Gamma distributions.

Saves final plots to the working directory.

Usage:
libtbx.python $MODULES/torchBragg/experimental/compare_gamma_poisson.py
"""

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

lambda_vec = torch.tensor([0.5, 1.0, 2.0, 50.0])

# Plot the Poisson distribution vs. the Gamma distribution
for i in range(len(lambda_vec)):
    x = torch.arange(0, lambda_vec[i]*10, 1)
    # Create the Poisson distribution
    poisson_dist = dist.poisson.Poisson(lambda_vec[i])

    # Create the Gamma distribution
    gamma_dist = dist.gamma.Gamma(lambda_vec[i], 1.0)

    # Create normal distribution
    normal_dist = dist.normal.Normal(lambda_vec[i], torch.sqrt(lambda_vec[i]))
    plt.figure()
    plt.title('lambda = {}'.format(lambda_vec[i]))
    plt.plot(x, torch.exp(poisson_dist.log_prob(x)).numpy(), label='Poisson')
    plt.plot(x, torch.exp(gamma_dist.log_prob(x)).numpy(), label='Gamma')
    plt.plot(x, torch.exp(normal_dist.log_prob(x)).numpy(), label='Normal')
    plt.legend()
    plt.savefig('compare_gamma_poisson_lambda_{}.png'.format(lambda_vec[i]))


