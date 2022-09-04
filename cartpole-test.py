import gym
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax

import data.collector as collector
from hyperparams import get_args
from networks.actor import MLPActor
from RL.fast import PG_loss_and_grad

"""
This file serve as a test bed for sanity checking my implementations with the cartpole environment.
Currently, I am using an off-policy version of the REINFORCE algorithm modified so that it can use replay buffers and batch updates.

TODO: Code cleanup
    TODO: [Advance] fill_buffer should be jit-able so that we can use vmap, pmap to to collect trajectories using multiple workers
TODO: Proper evaluation (along with video of sample runs) in a separate env should be implemented.
TODO: Add a critic network and use it to add a baseline to the RL objective

? Right now I'm getting a score of ~285 after 20 epochs of training.
"""

args = get_args()


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, new_step_api=True, render_mode='rgb_array')
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, f"videos/{run_name}", new_step_api=True, episode_trigger=lambda e: e % 75 == 1)  # Record every 75 episodes
        env.reset(seed=seed+idx)
        return env

    return thunk


def train(rng):
    # ATTENTION: SyncVectorEnv automatically resets the envs when they are done
    args.capture_video = True
    envs = gym.vector.AsyncVectorEnv([make_env(
        args.env_id, args.seed + i, i, args.capture_video, args.exp_name) for i in range(args.num_envs)], new_step_api=True)

    rng, actor_rng, critic_rng = random.split(rng, 3)
    actor = hk.transform(lambda x: MLPActor(
        hidden_dims=[64, 64], num_actions=envs.single_action_space.n)(x))
    actor_params = actor.init(rng=actor_rng, x=jnp.zeros(
        envs.single_observation_space.shape))
    actor = jax.jit(actor.apply)

    total_steps = args.epochs * args.iters_per_epoch * \
        (args.num_envs * args.num_steps / args.mini_batch_size)
    lr_schedule = optax.cosine_decay_schedule(args.lr, decay_steps=total_steps)
    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=lr_schedule),
    )
    optimizer_state = optimizer.init(actor_params)

    for epoch in range(args.epochs):
        rng, buffer_rng, learn_rng = random.split(rng, 3)

        print('Collecting samples...')
        buffer = collector.collect_rollouts(
            envs, (actor_params, actor), args, buffer_rng)
        # ! Caution: buffer[:]['adv'] will not update in place
        buffer['returns'][:] = buffer['adv'][:] = np.array(
            collector.compute_returns(buffer, args.gamma))
        buffer.flatten()

        print('Optimizing...')
        actor_params, optimizer_state, stats = learn(
            buffer, actor, actor_params, optimizer, optimizer_state, learn_rng)
        print(
            f'epoch:\t{epoch+1}\t loss:\t{stats["avg_loss"]}\t score:\t{stats["avg_reward"]}')
        # evaluate()
        del buffer


def learn(buffer, actor, actor_params, optimizer, optimizer_state, rng):
    '''
    Use the info in the buffer to compute the loss and optimize the policy
    then flatten and make mini-batches from the buffer
        -> Normalize advantages in the minibatch
    use a mini-batch to compute the gradient of the loss
    update the network

    '''
    # * iterate k times
    # * create minibatches from buffer
    # * for each minibatch
    # * compute the loss
    # * update the networks
    total_loss = 0
    for i in range(args.iters_per_epoch):
        rng, shuffle_rng = random.split(rng)
        indices = random.permutation(shuffle_rng, len(buffer))
        for j in range(0, len(buffer), args.mini_batch_size):
            mini_batch = buffer[indices[j: j + args.mini_batch_size]]
            rng, mb_rng = random.split(rng)
            loss, grads = PG_loss_and_grad(
                actor_params, actor, mini_batch, mb_rng, use_importance_weights=True)

            updates, optimizer_state = optimizer.update(
                grads, optimizer_state, actor_params)
            actor_params = optax.apply_updates(actor_params, updates)

            total_loss += loss

    avg_loss = total_loss / args.iters_per_epoch
    avg_reward = jnp.sum(buffer['rew']) / (jnp.sum(buffer['done']) + args.num_envs -
                                           jnp.sum(buffer[-1]['done']))  # divided by the number of episodes in the buffer

    stats = {
        'avg_loss': avg_loss,
        'avg_reward': avg_reward,
    }

    return actor_params, optimizer_state, stats


def evaluate():
    pass


if __name__ == '__main__':
    train(random.PRNGKey(args.seed))
