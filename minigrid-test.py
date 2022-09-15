import functools
import operator

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from gym.core import ObservationWrapper
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from gym_minigrid.minigrid import Goal

import data.collector as collector
from hyperparams import get_args
from networks.actor import MLPActor
from networks.common import MLP
from RL.fast import ppo_loss_and_grad

args = get_args('minigrid')


class DenseRewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.goal_position: tuple = None
        self.prev_position: tuple = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if not self.goal_position:
            self.goal_position = [
                x for x, y in enumerate(self.grid.grid) if isinstance(y, Goal)
            ]
            if len(self.goal_position) >= 1:
                self.goal_position = (
                    int(self.goal_position[0] / self.height),
                    self.goal_position[0] % self.width,
                )
        self.prev_position = (self.agent_pos[0], self.agent_pos[1])
        return obs

    def dist(self, x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])

    def step(self, action):
        self.prev_position = (self.agent_pos[0], self.agent_pos[1])
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += (args.gamma * (1 - self.dist((self.agent_pos[0], self.agent_pos[1]), self.goal_position) / (self.width+self.height)) -
                   (1 - self.dist(self.prev_position, self.goal_position) /
                    (self.width+self.height))
                   )*0.2

        return obs, reward, terminated, truncated, info


class RewardWrapper(gym.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= 0.01

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FlatImgDirObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = functools.reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(imgSize+4,),
            dtype="uint8",
        )

    def observation(self, obs):
        direction = np.zeros(4, dtype='uint8')
        direction[obs['direction']] = 1
        img = obs['image'].flatten()
        return np.append(img, direction)


def make_env(env_id, seed, idx, capture_video, run_name):
    # max_steps for minigrid envs is 4 x size x size
    def thunk():
        env = gym.make(env_id, render_mode='rgb_array')
        env = FullyObsWrapper(env)
        env = FlatImgDirObsWrapper(env)
        env = DenseRewardWrapper(env)
        # env = RewardWrapper(env)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, f"videos/{run_name}", episode_trigger=lambda e: e % 25 == 1)  # Record every 25 episodes
        env.reset(seed=seed+idx)
        return env

    return thunk


def train(rng):
    envs = gym.vector.SyncVectorEnv([make_env(
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
        ret, adv, vals = collector.compute_gae(
            buffer, (critic_params, critic), gae_rng, args.gamma, args.gae_lambda)
        buffer['returns'][:] = ret
        buffer['adv'][:] = adv
        buffer['value'][:] = vals

        buffer.flatten()

        print('Optimizing...')

        actor_params, critic_params, optimizer_state, stats = ppo_learn(
            buffer, (actor, actor_params), (critic, critic_params), (optimizer, optimizer_state), args, learn_rng)
        print(
            f'epoch:\t{epoch+1}\t loss:\t{stats["avg_loss"]:.2f}\t score:\t{stats["avg_reward"]:.2f}\t a_loss:\t{stats["policy_loss"]:.2f}\t c_loss:\t{stats["value_loss"]:.2f}\t e_loss:\t{stats["entropy_loss"]:.2f}'
        )
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


if __name__ == '__main__':
    train(random.PRNGKey(args.seed))
    # env = gym.make('MiniGrid-Empty-5x5-v0')
    # action_map = ['left', 'right', 'forward',
    #               'pickup', 'drop', 'toggle', 'done']
    # env = FullyObsWrapper(env)
    # env = FlatImgDirObsWrapper(env)
    # env = DenseRewardWrapper(env)
    # obs, info = env.reset()
    # # for i in range(20):
    # for act in [2, 2,1,2,2,6]:
    #     # act = env.action_space.sample()
    #     obs, rew, term, trunc, info = env.step(act)
    #     print(
    #         f'Action: {action_map[act]}\tReward: {rew}\tTerminated: {term}\tTruncated: {trunc}')


# action_map = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
# env = gym.make('MiniGrid-Empty-8x8-v0', render_mode='human')
