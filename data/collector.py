import argparse
from functools import partial
from typing import Callable, Tuple

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from data.storage import DictList


def collect_rollouts(envs: gym.vector.VectorEnv, agent: Tuple[hk.Params, Callable], args: argparse.Namespace, rng) -> Tuple[DictList, np.ndarray]:
    """Runs the vectorized environments for args.num_steps to collect experience

    Args:
        envs (gym.vector.VectorEnv): vectorized envs to run
        agent (Tuple[hk.Params, Callable]): a tuple of (actor_params, actor) that is used to generate actions
        args (argparse.Namespace): holder of relevant arguments
        rng (): key for random number generation

    Returns:
        Tuple[DictList, np.ndarray]: (Buffer of collected experiences, next_done values for the last + 1 time-step)
    """
    actor_params, actor = agent
    buffer = DictList((args.num_steps, args.num_envs), info={
        'obs': envs.single_observation_space.shape,
        'act': envs.single_action_space.shape,
        'rew': 1,
        'next_obs': envs.single_observation_space.shape,
        'terminated': 1,
        'truncated': 1,
        'done': 1,
        'returns': 1,
        'adv': 1,
        'logp': 1,  # For calculating importance weights in off-policy learning
    })

    obs = envs.reset()
    next_terminated, next_truncated = np.zeros(
        (args.num_envs,), dtype=bool), np.zeros((args.num_envs,), dtype=bool)
    for t in range(args.num_steps):
        rng, actor_rng, sample_rng = random.split(rng, 3)
        action_dists = actor(
            params=actor_params, x=obs, rng=actor_rng)
        actions = np.array(action_dists.sample(sample_rng))
        buffer[t] = {'obs': obs, 'act': actions, 'terminated': next_terminated, 'truncated': next_truncated, 'done': next_terminated | next_truncated,
                     'logp': action_dists.log_prob(actions)}
        next_obs, reward, next_terminated, next_truncated, info = envs.step(
            actions)
        buffer[t] = {'rew': reward, 'next_obs': next_obs}

        obs, next_terminated, next_truncated = np.array(next_obs), np.array(
            next_terminated, dtype=bool), np.array(next_truncated, dtype=bool)

    buffer.additional_data.append(next_terminated)
    buffer.additional_data.append(next_truncated)
    return buffer


# TODO: Right now the loop will be completely unrolled by jax. Rewrite it with jax.lax.scan to avoid this.
@partial(jax.jit, static_argnames=['gamma'])
def compute_returns(buffer: DictList, gamma: float) -> jnp.ndarray:
    """Compute returns (reward-to-gos) for a buffer

    Args:
        buffer (DictList): buffer of collected experiences. Must also have last_next_terminated field.
        gamma (float): discount factor

    Returns:
        jnp.ndarray: device array of returns
    """
    assert len(buffer.shape) == 2
    num_steps, num_envs = buffer.shape
    returns = jnp.zeros(buffer.shape)
    last_next_terminated = buffer.additional_data[0]
    for t in reversed(range(num_steps)):
        if t == num_steps-1:
            next_nonterminal = 1 - last_next_terminated
            next_return = jnp.zeros(num_envs)
        else:
            next_nonterminal = 1 - buffer[t+1]['terminated']
            next_return = returns[t+1]
        returns = returns.at[t].set(
            buffer[t]['rew'] + gamma*next_nonterminal*next_return)

    return returns


# TODO: Right now the loop will be completely unrolled by jax. Rewrite it with jax.lax.scan to avoid this.
@partial(jax.jit, static_argnames=['V', 'gamma', 'gae_lambda'])
def compute_gae(buffer: DictList, V: Callable, gamma: float, gae_lambda: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute generalized advantage estimates for trajectories in the buffer

    Args:
        buffer (DictList): buffer of collected experiences
        V (Callable): value function
        gamma (float): discount factor
        gae_lambda (float): coefficient lambda used in GAE

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: tuple of device arrays containing returns and advantages
    """
    assert len(buffer.shape) == 2
    num_steps, num_envs = buffer.shape
    last_next_terminated = buffer.additional_data[0]
    advantages, returns = jnp.zeros(buffer.shape), jnp.zeros(buffer.shape)
    next_value = V(buffer['next_obs'][-1]).flatten()
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_nonterminal = 1 - last_next_terminated
        else:
            next_nonterminal = 1 - buffer['terminated'][t+1]
        current_value = V(buffer['obs'][t]).flatten()
        delta = buffer['rew'][t] + gamma * \
            next_value * next_nonterminal - current_value
        advantages = advantages.at[t].set(
            delta + gamma * gae_lambda * next_nonterminal * (0 if t == num_steps - 1 else advantages[t+1]))
        returns = returns.at[t].set(advantages[t] + current_value)
        next_value = current_value

    return returns, advantages
