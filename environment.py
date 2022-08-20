from typing import Tuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, pmap, random, vmap

'''
rollout uses a given policy to generate a single episode
rollout must be jit compilable

TODO: pmapping the worker results in error
TODO: gymnax works with jax version < 0.3.15
'''


def get_rollout(env, policy, episode_length, batch_size, num_workers, rng):
    '''
    This function uses num_workers actors to generate episodes of length episode_length.
    It returns the collected experience in batches of a specific size
    '''
    v_rollout = jax.vmap(rollout_worker, in_axes=(
        0, None, None, None), out_axes=0)
    batch_obs, batch_action, batch_next_obs, batch_reward, batch_done, batch_reward2go = None, None, None, None, None, None
    batch_logits = None  # or batch_logprobs: record these for off-policy importance weighting

    batches = [batch_obs, batch_logits, batch_action,
               batch_reward, batch_next_obs, batch_done]

    while batches[0] is None or len(batches[0]) < batch_size:
        rng, batch_rng = random.split(rng)
        rngs = random.split(batch_rng, num_workers)
        obs, logits, action, reward, next_obs, done = v_rollout(
            rngs, env, policy, episode_length)  # num_workers x epi_len x *dims

        # TODO post processing the episodes

        # done is currently redundant: it contains all Trues
        datas = remove_dones(done, obs, logits, action,
                             reward, next_obs) + [done[done]]
        batches = [data if batch is None else np.append(
            batch, data, axis=0) for data, batch in zip(datas, batches)]

    return [e[:batch_size] for e in batches]


def rollout_worker(rng, env: Tuple, policy: Tuple, episode_length: int):
    '''
    env <- (env object, env_params)
    policy <- (policy object, policy_params)
    episode_length: # of steps in an episode
    rng: random number generator
    '''
    env, env_params = env
    env, env_params = gymnax.make('CartPole-v1')  # TODO: Remove
    policy, policy_params = policy
    rng, key_reset, key_episode = random.split(rng, 3)
    obs, env_state = env.reset(key_reset, env_params)

    def policy_step(state_input, tmp):
        '''
        lax.scan compatible step transition in the environment
        '''
        obs, env_state, rng = state_input
        rng, rng_step, rng_policy = random.split(rng, 3)
        logits = policy.apply(policy_params, obs, rng_policy)
        # Action selection: for discrete action spaces, take the argamax
        # TODO: check if axis should be specified when vmap is used
        action = jnp.argmax(logits)

        next_obs, next_state, reward, done, _ = env.step(
            rng_step, env_state, action, env_params)
        carry = (next_obs, next_state, rng)
        output = (obs, logits, action, reward, next_obs, done)
        return carry, output

    _, scan_out = jax.lax.scan(
        policy_step, (obs, env_state, key_episode), None, episode_length)
    obs, logits, action, reward, next_obs, done = scan_out
    return obs, logits, action, reward, next_obs, done


# Trajectory processing functions
    # compute reward to gos and episode stats
    # TODO: for on-policy embeddings, those should also be computed here and added to the batches

def remove_dones(done, *args):
    '''
    args contain obs, actions, ...
    this function discards transitions that happened after the episode had ended (where done = True)
        + it flattens the arguments and makes them ready to be added to the buffer
    if for instance obs is of the shape B x T x d, the return shape will be BT x d, assuming that there is no done
    TODO: check if the first done in an episode should also be added to the buffer (add a true to the beginning of ~dones and a false to the end, then use argmax)
    '''
    assert len(done.shape) == 2
    return [e[~done] for e in args]


env_name = 'CartPole-v1'
