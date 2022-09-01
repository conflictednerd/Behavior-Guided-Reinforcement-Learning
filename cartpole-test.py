import gym
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from numpyro.distributions.discrete import Categorical

from data.storage import DictList
from hyperparams import get_args

"""
This file serve as a test bed for sanity checking my implementations with the cartpole environment.
Currently, I am using an off-policy version of the REINFORCE algorithm modified so that it can use replay buffers and batch updates.

TODO: Code cleanup
    TODO: actor net and optimizers must be defined in networks.py
    TODO: rollout worker (fill_buffer) must be in environment directory
    TODO: rl policy gradient loss (actor_loss) must be defined in rl.py
    TODO: [Advance] fill_buffer should be jit-able so that we can use vmap, pmap to to collect trajectories using multiple workers
TODO: Proper evaluation (along with video of sample runs) in a separate env should be implemented.
TODO: Add a critic network and use it to add a baseline to the RL objective

? Right now I'm getting a score of ~285 after 20 epochs of training.
"""

args = get_args()

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def train(rng):
    # ATTENTION: SyncVectorEnv automatically resets the envs when they are done
    envs = gym.vector.SyncVectorEnv([make_env(
        args.env_id, args.seed + i, i, args.capture_video, args.exp_name) for i in range(args.num_envs)])

    rng, actor_rng, critic_rng = random.split(rng, 3)
    actor = hk.transform(lambda x: hk.nets.MLP(
        [64, 2], activation=jnp.tanh)(x))
    actor_params = actor.init(rng=actor_rng, x=jnp.zeros(envs.single_observation_space.shape))

    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=args.lr),
    )
    optimizer_state = optimizer.init(actor_params)

    for epoch in range(args.epochs):
        rng, buffer_rng, learn_rng = random.split(rng, 3)
        print('Collecting samples...')
        buffer = fill_buffer(envs, actor_params, actor, buffer_rng)
        print('Optimizing...')
        actor_params, optimizer_state, stats = learn(
            buffer, actor, actor_params, optimizer, optimizer_state, learn_rng)
        print(
            f'epoch:\t{epoch+1}\t loss:\t{stats["avg_loss"]}\t score:\t{stats["avg_reward"]}')
        # evaluate()
        del buffer


def fill_buffer(envs, actor_params, actor, rng):
    '''
    run the vectorized env for n steps and put the experiences in the buffer
    return the flattened buffer
    '''
    buffer = DictList((args.num_steps, args.num_envs), info={
        'obs': envs.single_observation_space.shape,
        'act': envs.single_action_space.shape,
        'rew': 1,
        'next_obs': envs.single_observation_space.shape,
        'done': 1,
        'returns': 1,
        'adv': 1,
        # log probability of the selected action. Used to compute importance weights for off-policy updates.
        'logp': 1,
    })

    next_obs = envs.reset()
    next_done = np.zeros((args.num_envs,))
    for t in range(args.num_steps):
        rng, actor_rng, sample_rng = random.split(rng, 3)
        logits = actor.apply(params=actor_params, x=next_obs, rng=actor_rng)
        # Todo: modify actor networks to return a distribution object instead of logits
        actions = np.array(Categorical(logits=logits).sample(sample_rng))
        buffer[t] = {'obs': next_obs, 'act': actions, 'done': next_done,
                     'logp': Categorical(logits=logits).log_prob(actions)}
        next_obs, reward, done, info = envs.step(actions)
        buffer[t] = {'rew': reward, 'next_obs': next_obs}

        next_obs, next_done = np.array(next_obs), np.array(done)

    # compute reward2gos
    assert len(buffer.shape) == 2
    for t in reversed(range(args.num_steps)):
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

    buffer.flatten()
    return buffer


def policy_loss(actor_params, actor, mini_batch, rng):
    '''
    function that computes the policy loss given a mini-batch
    will call grad on it to get the gradients

    Vanilla policy gradient loss with importance sampling.
    # ! Important theoretical caveat: In actuality, the importance ratio (new_log_p/old_logp) must be computed over the entire trajectory, and not just for a single time-step.
    # ! However, when the behavior policy is not "far from" the policy to be updated, we can use a first order approximate of the full importance ratio.
    # ! For more details, checkout this https://youtu.be/KZd508qGFt0
    '''
    obs, act, adv, old_logp = mini_batch['obs'], mini_batch['act'], mini_batch['adv'], mini_batch['logp']
    rng, actor_rng = random.split(rng)
    logits = actor.apply(params=actor_params, x=obs, rng=actor_rng)
    log_probs = Categorical(logits=logits).log_prob(act.astype(int))
    # ! importance weights for off-policy updates should not be involved in the gradient computation.
    loss = - (log_probs * jax.lax.stop_gradient(adv *
              log_probs / old_logp)).mean()
    return loss


def value_loss():
    '''
    function that computes critic's loss given a mini-batch
    '''


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
    total_loss = 0  # todo report average reward as well
    for i in range(args.iters_per_epoch):
        rng, shuffle_rng = random.split(rng)
        indices = random.permutation(shuffle_rng, len(buffer))
        for j in range(0, len(buffer), args.mini_batch_size):
            mini_batch = buffer[indices[j: j + args.mini_batch_size]]
            rng, mb_rng = random.split(rng)
            loss, grads = jax.value_and_grad(policy_loss)(
                actor_params, actor, mini_batch, mb_rng)

            updates, optimizer_state = optimizer.update(
                grads, optimizer_state, actor_params)
            actor_params = optax.apply_updates(actor_params, updates)

            total_loss += loss

    avg_loss = total_loss / args.iters_per_epoch
    avg_reward = jnp.sum(buffer['rew']) / (jnp.sum(buffer['done']) + args.num_envs-
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
