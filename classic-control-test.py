import random
import time
from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter

from algorithm.actorcritic import compute_gae, ppo_step
from hyperparams import get_args
from models.ac import ActorCritic


@dataclass
class Buffer:
    obs: np.ndarray
    obsp: np.ndarray  # True next observations (even when done was True)
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    terms: np.ndarray
    truncs: np.ndarray
    values: np.ndarray
    next_obs: np.ndarray
    next_term: np.ndarray
    next_trunc: np.ndarray


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"runs/{run_name}/videos",
                video_length=1000,
                episode_trigger=lambda i: i % 50 == 0,
            )
        return env

    return thunk


if __name__ == "__main__":
    args = get_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in args._asdict().items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    jax_key = jax.random.PRNGKey(args.seed)

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )

    jax_key, init_key = jax.random.split(jax_key)
    agent = ActorCritic(
        actor_dims=(256, 256),
        critic_dims=(256, 256),
        num_actions=envs.single_action_space.n,
    )
    agent_params = agent.init(
        {"params": jax.random.fold_in(init_key, 1)},
        jnp.zeros((args.minibatch_size,) + envs.single_observation_space.shape),
    )
    agent_apply = jax.jit(agent.apply)
    critic_apply = jax.jit(lambda p, x: agent.apply(p, x, method=agent.get_value))

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.lr, eps=1e-5),
    )
    optimizer_state = optimizer.init(agent_params)

    # ALGO Logic: Storage setup
    #! Note: terms[t] indicates whether obs[t] is the start of a new trajectory or part of the previous trajectory
    buffer = Buffer(
        obs=np.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        ),
        obsp=np.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        ),
        actions=np.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=int
        ),
        log_probs=np.zeros((args.num_steps, args.num_envs)),
        rewards=np.zeros((args.num_steps, args.num_envs)),
        terms=np.zeros((args.num_steps, args.num_envs), dtype=bool),
        truncs=np.zeros((args.num_steps, args.num_envs), dtype=bool),
        values=np.zeros((args.num_steps, args.num_envs)),
        next_obs=envs.reset(seed=args.seed)[0],
        next_term=np.zeros(args.num_envs, dtype=bool),
        next_trunc=np.zeros(args.num_envs, dtype=bool),
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # times we re-fill the buffer
    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
    for update in range(1, num_updates + 1):
        jax_key, step_key = jax.random.split(jax_key)
        for step in range(args.num_steps):
            global_step += args.num_envs
            buffer.obs[step] = buffer.next_obs
            buffer.terms[step] = buffer.next_term
            buffer.truncs[step] = buffer.next_trunc

            # ALGO LOGIC: action logic
            action_dist, value = agent_apply(agent_params, buffer.next_obs)
            action = np.array(
                action_dist.sample(seed=jax.random.fold_in(step_key, step)), dtype=int
            )
            buffer.values[step] = value.flatten()
            buffer.actions[step] = action
            buffer.log_probs[step] = action_dist.log_prob(action)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, term, trunc, info = envs.step(action)
            buffer.rewards[step] = reward.flatten()
            buffer.next_obs, buffer.next_term, buffer.next_trunc = next_obs, term, trunc
            buffer.obsp[step] = next_obs
            if np.any(term) or np.any(trunc):
                for i, final_obs in enumerate(info["final_observation"]):
                    if final_obs is not None:
                        buffer.obsp[step][i] = final_obs

            if "final_info" in info.keys():
                for item in info["final_info"]:
                    if item is not None:
                        r, l = item["episode"]["r"], item["episode"]["l"]
                        print(f"global_step={global_step}, episodic_return={r}")
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)
                        break  # TODO: why?

        advantages = compute_gae(buffer, args, critic_apply, agent_params)
        returns = advantages + buffer.values  # Q values

        # Putting all arrays on device by converting to jax.ndarray
        clipfracs = []
        indices = jnp.arange(args.num_envs * args.num_steps)
        obs = jnp.array(buffer.obs.reshape((-1,) + buffer.obs.shape[2:]))
        actions = jnp.array(buffer.actions.reshape((-1,) + buffer.actions.shape[2:]))
        log_probs = jnp.array(buffer.log_probs.reshape(-1))
        values = jnp.array(buffer.values.reshape(-1))
        returns = jnp.array(returns.reshape(-1))
        advantages = jnp.array(advantages.reshape(-1))
        # Optimizing the policy and value network
        print("Training...")
        for epoch in range(args.updates_per_batch):
            jax_key, update_key = jax.random.split(jax_key)
            indices = jax.random.permutation(update_key, indices)
            # TODO: jax.lax.fori_loop
            for start_idx in range(0, len(indices), args.minibatch_size):
                agent_params, optimizer_state, stats = ppo_step(
                    agent,
                    agent_params,
                    optimizer,
                    optimizer_state,
                    indices,
                    start_idx,
                    obs,
                    actions,
                    log_probs,
                    values,
                    returns,
                    advantages,
                    args,
                )
                clipfracs += [stats["clipfrac"]]

            if args.target_kl is not None:
                if stats["approx_kl"] > args.target_kl:
                    break

        y_pred, y_true = values, returns
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate",
        #   optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", float(stats["v_loss"]), global_step)
        writer.add_scalar("losses/policy_loss", float(stats["pg_loss"]), global_step)
        writer.add_scalar("losses/entropy", float(stats["entropy_loss"]), global_step)
        writer.add_scalar(
            "losses/old_approx_kl", float(stats["old_approx_kl"]), global_step
        )
        writer.add_scalar("losses/approx_kl", float(stats["approx_kl"]), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar(
            "losses/explained_variance", float(explained_var), global_step
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()
