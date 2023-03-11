from functools import partial
from typing import Any, Dict, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


@partial(jax.jit, static_argnames=["agent", "optimizer", "args"])
def ppo_step(
    agent: nn.Module,
    agent_params: flax.core.scope.FrozenVariableDict,
    optimizer: optax.GradientTransformation,
    optimizer_state,
    indices: jax.Array,
    start_idx: int,
    obs: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    values: jax.Array,
    returns: jax.Array,
    advantages: jax.Array,
    args,
) -> Tuple[flax.core.scope.FrozenVariableDict, Any, Dict]:
    mb_inds = jax.lax.dynamic_slice(indices, (start_idx,), (args.minibatch_size,))

    def loss_fn(params):
        stats = dict()
        new_action_dist, new_value = agent.apply(params, obs[mb_inds])
        action = actions[mb_inds]
        new_log_prob = new_action_dist.log_prob(action)
        entropy = new_action_dist.entropy()

        log_ratio = new_log_prob - log_probs[mb_inds]
        ratio = jnp.exp(log_ratio)  # pi_{new}(a|s) / pi_{old}(a|s)

        mb_advantages = advantages[mb_inds]
        mb_advantages = jax.lax.select(
            args.norm_adv,
            on_true=(mb_advantages - mb_advantages.mean()) / (mb_advantages + 1e-8),
            on_false=mb_advantages,
        )

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(
            ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        new_value = new_value.reshape(-1)
        if args.clip_vloss:
            v_loss_unclipped = (new_value - returns[mb_inds]) ** 2
            v_clipped = values[mb_inds] + jnp.clip(
                new_value - values[mb_inds], -args.clip_coef, args.clip_coef
            )
            v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
            v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((new_value - returns[mb_inds]) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
        stats = {
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            "old_approx_kl": -log_ratio.mean(),
            "approx_kl": ((ratio - 1) - log_ratio).mean(),
            "clipfrac": (jnp.abs(ratio - 1.0) > args.clip_coef).mean(),
            "loss": loss,
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "entropy_loss": entropy_loss,
        }
        return loss, stats

    (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent_params)
    param_updates, optimizer_state = optimizer.update(
        grads, optimizer_state, agent_params
    )
    agent_params = optax.apply_updates(agent_params, param_updates)
    return agent_params, optimizer_state, stats


def compute_gae(buffer, args, critic_apply, critic_params):
    advantages = np.zeros_like(buffer.rewards)
    lastgaelam = 0
    # nextnonterminal = 1 -> next_obs is valid and we can bootsrap from it
    for t in reversed(range(args.num_steps)):
        nextnonterminal = 1 - (
            buffer.next_term if t == args.num_steps - 1 else buffer.terms[t + 1]
        )
        nextvalues = critic_apply(critic_params, buffer.obsp[t]).reshape(1, -1)
        delta = (
            buffer.rewards[t]
            + args.gamma * (nextvalues * nextnonterminal)
            - buffer.values[t]
        )
        advantages[t] = lastgaelam = (
            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        )
    return advantages


def compute_gae_old(buffer, args, critic_apply, critic_params):
    """
    This won't differentiate between terminated and truncated, but is faster
    """
    advantages = np.zeros_like(buffer.rewards)
    lastgaelam = 0
    # nextnonterminal = 1 -> next_obs is valid and we can bootsrap from it
    next_value = critic_apply(critic_params, buffer.next_obs).reshape(1, -1)
    for t in reversed(range(args.num_steps)):
        if (
            t == args.num_steps - 1
        ):  # last transition in buffer, next_obs, next_done are stored separately
            nextnonterminal = 1.0 - np.logical_or(buffer.next_term, buffer.next_trunc)
            nextvalues = next_value
        else:  # look in the buffer for next_obs and next_done
            nextnonterminal = 1.0 - np.logical_or(
                buffer.terms[t + 1], buffer.truncs[t + 1]
            )
            nextvalues = buffer.values[t + 1]
        delta = (
            buffer.rewards[t]
            + args.gamma * (nextvalues * nextnonterminal)
            - buffer.values[t]
        )
        advantages[t] = lastgaelam = (
            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        )
