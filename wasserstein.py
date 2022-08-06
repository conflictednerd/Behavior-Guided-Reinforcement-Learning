import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, pmap, vmap, random

'''
Utilities for estimating Wasserstein distance
'''


def Phi(tau):
    '''
    Behavioral Embedding Map (BEM)

    maps trajectories into embedding vectors

    tau: (s, a, r) trajectory
    '''
    s, a, r = tau
    return s[-1]


def C(x, y):
    '''
    Cost (distance) function over the embedding space
    '''
    return jnp.linalg.norm(x-y)


@jax.jit
def phi(params, x):
    '''
    Random feature map from embedding space into another space
    Used to compute lambda
    params = (G, b)
      G <- m*d matrix
      b <- m vector
    '''
    G, b = params
    return jnp.cos(jnp.dot(G, x) + b) / jnp.sqrt(len(b))


def init_kernel(key, m: int, d: int):
    '''
    Returns G, b
    generates random feature maps used to approximate lambda
      G <- m*d matrix
      b <- m vector
    m: number of random features
    d: embedding space dimension
    '''
    key, subkey = random.split(key)
    G = random.normal(subkey, (m, d))
    b = random.uniform(key, (m,), maxval=2*np.pi)
    return G, b


@jax.jit
def lambda_(params, x):
    '''
    Scoring function
    maps embedding space into R
    params: p_lambda, G, b
      p_lambda are the parameters that will be learned
      G, b are random feature vectors
    '''
    p_lambda, G, b = params
    return jnp.dot(p_lambda, phi((G, b), x))


@jax.jit
def F(p_mu, G_mu, b_mu, p_nu, G_nu, b_nu, gamma, x, y):
    return jnp.exp(
        (
            lambda_((p_mu, G_mu, b_mu), x)
            - lambda_((p_nu, G_nu, b_nu), y)
            - C(x, y))
        / gamma)


@jax.jit
def update_lambdas(p_mu, G_mu, b_mu, p_nu, G_nu, b_nu, gamma, alpha, t, x, y):
    '''
    Performs one SGD update to Behavioral Test Functions (lambdas) using a single sample
    p_mu: parameters of lambda_mu that will be updated
    G_mu, b_mu: random features of lambda_mu kernel
    p_nu: parameters of lambda_nu that will be updated
    G_nu, b_nu: random features of lambda_nu kernel
    gamma: WD smoothing constant >= 0
    alpha: learning rate
    t: time step (update number)
    x: sample from mu (embedded trajectory)
    y: sample from nu (embedded trajectory)

    Returns:
    updated p_mu, p_nu
    '''
    constant = F(p_mu, G_mu, b_mu, p_nu, G_nu, b_nu, gamma, x, y)
    p_mu += (alpha / jnp.sqrt(t)) * phi((G_mu, b_mu), x) * (1 - constant)
    p_nu -= (alpha / jnp.sqrt(t)) * phi((G_nu, b_nu), x) * (1 - constant)

    return p_mu, p_nu


@jax.jit
def WD_estimate(p_mu, G_mu, b_mu, p_nu, G_nu, b_nu, gamma, x, y):
    '''
    Given optimal p_mu, p_nu and a single sample (x, y) estimates the WD
    TODO: When using vmap only add batch dimension to x, y and do a mean over the batch result
    '''
    return lambda_((p_mu, G_mu, b_mu), x) - lambda_((p_nu, G_nu, b_nu), y) - gamma * F(p_mu, G_mu, b_mu, p_nu, G_nu, b_nu, gamma, x, y)


def get_WD(x_traj_batch, y_traj_batch, params):
    '''Write a function that takes multiple sample trajectories,
        Uses batched-SGD to find lambda functions 
        and returns the corresponding Wasserstien distance function'''
    # Initilize kernels

    # Find lambdas

    # Return WD function
    def WD(x, y):
        return WD(p_mu, G_mu, b_mu, p_nu, G_nu, b_nu, gamma, x, y)

    return WD
