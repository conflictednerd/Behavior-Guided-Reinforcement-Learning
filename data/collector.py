import argparse
from functools import partial
from typing import Tuple

import gym
import haiku as hk
import jax
import jax.random as random
import numpy as np

from data.storage import DictList


def collect_rollouts(envs: gym.vector.VectorEnv, agent: Tuple[hk.Params, hk.Transformed], args: argparse.Namespace, rng) -> Tuple[DictList, np.ndarray]:
    """Runs the vectorized environments for args.num_steps to collect experience

    Args:
        envs (gym.vector.VectorEnv): vectorized envs to run
        agent (Tuple[hk.Params, hk.Transformed]): a tuple of (actor_params, actor) that is used to generate actions
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
        action_dists = actor.apply(
            params=actor_params, x=obs, rng=actor_rng)
        actions = np.array(action_dists.sample(sample_rng))
        buffer[t] = {'obs': obs, 'act': actions, 'terminated': next_terminated, 'truncated': next_truncated, 'done': next_terminated | next_truncated,
                     'logp': action_dists.log_prob(actions)}
        next_obs, reward, next_terminated, next_truncated, info = envs.step(
            actions)
        buffer[t] = {'rew': reward, 'next_obs': next_obs}

        obs, next_terminated, next_truncated = np.array(next_obs), np.array(
            next_terminated, dtype=bool), np.array(next_truncated, dtype=bool)

    return buffer, next_terminated


# TODO: Make it jit-able
def compute_returns(buffer: DictList, next_done: np.ndarray, args: argparse.Namespace) -> DictList:
    """Compute returns (reward-to-gos) for a buffer

    Args:
        buffer (DictList): buffer of collected experiences
        next_done (np.ndarray): done values for the last+1 time-step, used for correct bootstrapping of values.
        args (argparse.Namespace): holder of relevant arguments

    Returns:
        DictList: same buffer, with its 'returns' data field set.
    """
    assert len(buffer.shape) == 2
    for t in reversed(range(args.num_steps)):
        # TODO: when a critic is added, truncated but not terminated episodes should bootstrap using critic values whereas terminated episodes should use a value of 0 (right now all are zero)
        if t == args.num_steps-1:
            next_nonterminal = 1-next_done
            next_return = np.zeros(args.num_envs)  # critic(next_obs)
        else:
            next_nonterminal = 1 - buffer[t+1]['done']
            next_return = buffer[t+1]['returns']
        buffer[t] = {'returns': buffer[t]['rew'] +
                     args.gamma*next_nonterminal*next_return}

    # adv = returns - values
    # ! Caution: buffer[:]['adv'] will not update in place
    buffer['adv'][:] = buffer['returns'][:]
    return buffer
