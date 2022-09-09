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
from networks.common import MLP
from RL.fast import PG_loss_and_grad, ppo_loss_and_grad, value_loss_and_grad

"""
This file serve as a test bed for sanity checking my implementations with the cartpole environment.
Currently, I am using an off-policy version of the REINFORCE algorithm modified so that it can use replay buffers and batch updates.

TODO: Code cleanup
    TODO: [Advance] fill_buffer should be jit-able so that we can use vmap, pmap to to collect trajectories using multiple workers
TODO: Proper evaluation (along with video of sample runs) in a separate env should be implemented.

? Right now I'm getting a score of ~500 (max_score) after 10 epochs of training (stable).
"""

args = get_args()


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode='rgb_array')
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, f"videos/{run_name}", episode_trigger=lambda e: e % 20 == 1)  # Record every 75 episodes
        env.reset(seed=seed+idx)
        return env

    return thunk


def train(rng):
    # ATTENTION: SyncVectorEnv automatically resets the envs when they are done
    args.capture_video = True
    envs = gym.vector.AsyncVectorEnv([make_env(
        args.env_id, args.seed + i, i, args.capture_video, args.exp_name) for i in range(args.num_envs)])

    rng, actor_rng, critic_rng = random.split(rng, 3)
    actor = hk.transform(lambda x: MLPActor(
        hidden_dims=[64, 64], num_actions=envs.single_action_space.n)(x))
    actor_params = actor.init(rng=actor_rng, x=jnp.zeros(
        envs.single_observation_space.shape))
    actor = jax.jit(actor.apply)

    critic = hk.transform(lambda x: MLP(
        hidden_dims=[64, 64], out_dim=1)(x))
    critic_params = critic.init(rng=critic_rng, x=jnp.zeros(
        envs.single_observation_space.shape))
    critic = jax.jit(critic.apply)

    total_steps = args.epochs * args.iters_per_epoch * \
        (args.num_envs * args.num_steps / args.mini_batch_size)
    # lr_schedule = optax.cosine_decay_schedule(
    #     args.lr, decay_steps=total_steps//2)
    lr_schedule = optax.constant_schedule(args.lr)
    optimizer = optax.chain(
        # optax.clip(1.0),
        optax.adam(learning_rate=lr_schedule),
    )
    optimizer_state = optimizer.init((actor_params, critic_params))

    for epoch in range(args.epochs):
        rng, buffer_rng, gae_rng, learn_rng = random.split(rng, 4)

        print('Collecting samples...')
        buffer = collector.collect_rollouts(
            envs, (actor_params, actor), args, buffer_rng)
        # ! Caution: buffer[:]['adv'] will not update in place
        # buffer['returns'][:] = buffer['adv'][:] = np.array(
        #     collector.compute_returns(buffer, args.gamma))
        ret, adv, vals = collector.compute_gae(
            buffer, (critic_params, critic), gae_rng, args.gamma, args.gae_lambda)
        buffer['returns'][:] = ret
        buffer['adv'][:] = adv
        buffer['value'][:] = vals

        buffer.flatten()

        print('Optimizing...')
        # actor_params, critic_params, optimizer_state, stats = a2clearn(
        #     buffer, actor, actor_params, critic, critic_params, optimizer, optimizer_state, learn_rng)
        # print(
        #     f'epoch:\t{epoch+1}\t a_loss:\t{stats["avg_actor_loss"]}\t c_loss:\t{stats["avg_critic_loss"]}\t score:\t{stats["avg_reward"]}'
        # )

        actor_params, critic_params, optimizer_state, stats = ppo_learn(
            buffer, (actor, actor_params), (critic, critic_params), (optimizer, optimizer_state), args, learn_rng)
        print(
            f'epoch:\t{epoch+1}\t loss:\t{stats["avg_loss"]:.2f}\t score:\t{stats["avg_reward"]:.2f}\t a_loss:\t{stats["policy_loss"]:.2f}\t c_loss:\t{stats["value_loss"]:.2f}\t e_loss:\t{stats["entropy_loss"]:.2f}'
        )
        # evaluate()
        del buffer


def ppo_learn(buffer, actor, critic, optimizer, args, rng):
    actor, actor_params = actor
    critic, critic_params = critic
    optimizer, optimizer_state = optimizer

    total_loss = 0
    for i in range(args.iters_per_epoch):
        rng, shuffle_rng = random.split(rng)
        indices = random.permutation(shuffle_rng, len(buffer))
        for j in range(0, len(buffer), args.mini_batch_size):
            mini_batch = buffer[indices[j: j+args.mini_batch_size]]
            rng, loss_rng = random.split(rng)
            ppo_args = {
                'clip_coef': args.clip_coef,
                'ent_coef': args.ent_coef,
                'vf_coef': args.vf_coef,
                'clip_vloss': args.clip_vloss,
            }
            (loss, stats), grads = ppo_loss_and_grad(
                (actor_params, critic_params), actor, critic, mini_batch, loss_rng, **ppo_args)
            updates, optimizer_state = optimizer.update(
                grads, optimizer_state, (actor_params, critic_params))
            actor_params, critic_params = optax.apply_updates(
                (actor_params, critic_params), updates)
            total_loss += loss

        if args.target_kl is not None and stats['approx_kl'] > args.target_kl:
            break

    # TODO: explained variance
    avg_loss = total_loss / args.iters_per_epoch
    avg_reward = jnp.sum(buffer['rew']) / (jnp.sum(buffer['done']) + args.num_envs -
                                           jnp.sum(buffer[-1]['done']))  # divided by the number of episodes in the buffer
    stats['avg_loss'] = avg_loss
    stats['avg_reward'] = avg_reward

    return actor_params, critic_params, optimizer_state, stats


def a2clearn(buffer, actor, actor_params, critic, critic_params, optimizer, optimizer_state, rng):
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
    total_actor_loss, total_critic_loss = 0, 0
    for i in range(args.iters_per_epoch):
        rng, shuffle_rng = random.split(rng)
        indices = random.permutation(shuffle_rng, len(buffer))
        for j in range(0, len(buffer), args.mini_batch_size):
            mini_batch = buffer[indices[j: j + args.mini_batch_size]]
            rng, actor_rng, critic_rng = random.split(rng, 3)
            actor_loss, actor_grads = PG_loss_and_grad(
                actor_params, actor, mini_batch, actor_rng, use_importance_weights=True)
            critic_loss, critic_grads = value_loss_and_grad(
                critic_params, critic, mini_batch, critic_rng, vf_coef=args.vf_coef)

            updates, optimizer_state = optimizer.update(
                (actor_grads, critic_grads), optimizer_state, (actor_params, critic_params))
            actor_params, critic_params = optax.apply_updates(
                (actor_params, critic_params), updates)

            total_actor_loss += actor_loss
            total_critic_loss += critic_loss

    # Bookkeeping
    avg_actor_loss = total_actor_loss / args.iters_per_epoch
    avg_critic_loss = total_critic_loss / args.iters_per_epoch
    avg_reward = jnp.sum(buffer['rew']) / (jnp.sum(buffer['done']) + args.num_envs -
                                           jnp.sum(buffer[-1]['done']))  # divided by the number of episodes in the buffer

    stats = {
        'avg_actor_loss': avg_actor_loss,
        'avg_critic_loss': avg_critic_loss,
        'avg_reward': avg_reward,
    }

    return actor_params, critic_params, optimizer_state, stats


def evaluate():
    pass


if __name__ == '__main__':
    train(random.PRNGKey(args.seed))
