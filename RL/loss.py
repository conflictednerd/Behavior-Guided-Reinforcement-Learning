from functools import partial
from typing import Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
from numpyro.distributions.discrete import Categorical


@partial(jax.jit, static_argnames=['critic', 'vf_coef'])
def value_loss(critic_params: hk.Params, critic: Callable, mini_batch: Dict, rng, vf_coef: float = 1.0):
    obs, returns = mini_batch['obs'], mini_batch['returns']
    values = critic(params=critic_params, x=obs, rng=rng)
    loss = (0.5 * (values - returns)**2).mean()
    loss *= vf_coef
    return loss


@partial(jax.jit, static_argnames=['actor', 'use_importance_weights'])
def policy_gradient_loss(actor_params: hk.Params, actor: Callable, mini_batch: Dict, rng, use_importance_weights: bool = False):
    """Function whose derivative with respect to its first parameter is the policy gradient (REINFORCE loss)
        mini-batch must contain:
            'obs': observations
            'act': actions
            'adv': advantages (returns or reward-to-gos with or without a baseline subtracted)
        If use_importance_weights is True, it must also contain:
            'logp': log probabilities of selected actions according to the behavior policy
        use_importance_weights can be used to make REINFORCE off-policy. For more details, checkout https://youtu.be/KZd508qGFt0


    Args:
        actor_params (hk.Params): parameters of the actor
        actor (hk.Transformed): the agent that outputs actions give observations
        mini_batch (Dict):  mini-batch of transition data. Must contain 'obs', 'act', 'adv' keys.
        rng (_type_): random number generation key
        use_importance_weights (bool, optional): if True, importance weights are added. Use with off-policy data. Defaults to False.

    Returns:
        A single scalar denoting the loss
    """
    obs, act, adv = mini_batch['obs'], mini_batch['act'], mini_batch['adv']
    if use_importance_weights:
        old_logp = mini_batch['logp']
    rng, actor_rng = random.split(rng)
    log_probs = actor(params=actor_params, x=obs,
                      rng=actor_rng).log_prob(act.astype(int))

    loss = - (log_probs * jax.lax.stop_gradient(
        (adv * jnp.exp(log_probs - old_logp)) if use_importance_weights else adv)
    ).mean()
    return loss


@partial(jax.jit, static_argnames=['actor', 'critic', 'clip_coef', 'ent_coef', 'vf_coef', 'clip_vloss'])
def ppo_loss(
        params: Tuple[hk.Params, hk.Params],
        actor: Callable, critic: Callable,
        mini_batch: Dict,
        rng,
        clip_coef: float, ent_coef: float, vf_coef: float, clip_vloss: bool
):
    obs, act, adv, returns, value, old_logp = mini_batch['obs'], mini_batch[
        'act'], mini_batch['adv'], mini_batch['returns'], mini_batch['value'], mini_batch['logp']
    returns, value, adv, old_logp = jax.lax.stop_gradient(
        (returns, value, adv, old_logp))
    actor_params, critic_params = params
    rng, actor_rng, critic_rng = random.split(rng, 3)

    new_acts = actor(params=actor_params, x=obs, rng=actor_rng)
    new_logp = new_acts.log_prob(act.astype(int))
    log_ratio = new_logp - old_logp
    ratio = jnp.exp(log_ratio)

    # Categorical entropy loss (only for discrete actions)
    entropy = categorical_entropy(new_acts)
    entropy_loss = entropy.mean()

    # Approximate KL between policies: http://joschu.net/blog/kl-approx.html
    old_approx_kl = -log_ratio.mean()
    approx_kl = ((ratio - 1) - log_ratio).mean()

    # Actor (policy) loss
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * jax.lax.clamp(1-clip_coef, ratio, 1+clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Critic (value fn) loss
    new_value = critic(params=critic_params, x=obs, rng=critic_rng)
    if clip_vloss:
        v_loss_unclipped = (new_value - returns)**2
        v_clipped = value - jax.lax.clamp(
            - clip_coef, new_value - value, clip_coef
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * jnp.maximum(v_loss_clipped, v_loss_unclipped).mean()
    else:
        v_loss = 0.5 * ((new_value - returns)**2).mean()

    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

    stats = {
        'policy_loss': pg_loss,
        'entropy_loss': entropy_loss,
        'value_loss': v_loss,
        'old_approx_kl': old_approx_kl,
        'approx_kl': approx_kl,
    }
    return loss, stats


def categorical_entropy(dist: Categorical):
    """Helper function for calculating entropy for categorical distribution
    distrax implementation:
        https://github.com/deepmind/distrax/blob/master/distrax/_src/distributions/categorical.py#L114

    Args:
        dist (Categorical): distribution object

    Returns:
        entropy
    """
    log_probs = jax.nn.log_softmax(dist.logits)
    p = jnp.exp(log_probs)
    e = p * jnp.where(p == 0, 0.0, log_probs)
    return -jnp.sum(e, axis=-1)
