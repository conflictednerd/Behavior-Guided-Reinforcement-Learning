from functools import partial
from typing import Dict

import haiku as hk
import jax
import jax.random as random


@partial(jax.jit, static_argnums=(1, 4))
def policy_gradient_loss(actor_params: hk.Params, actor: hk.Transformed, mini_batch: Dict, rng, use_importance_weights: bool = False):
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
    log_probs = actor.apply(params=actor_params, x=obs,
                            rng=actor_rng).log_prob(act.astype(int))

    loss = - (log_probs * jax.lax.stop_gradient(adv * log_probs / old_logp
                                                if use_importance_weights else adv)
              ).mean()
    return loss
