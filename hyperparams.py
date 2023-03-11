import argparse
from collections import namedtuple
from distutils.util import strtobool


def get_args(arg_set: str = "default"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="LunarLander")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=False,
        help="whether to capture videos of the agent (saved in the `videos` directory)",
    )

    parser.add_argument(
        "--env-id",
        type=str,
        default="LunarLander-v2",  # Tested with Acrobot-v1, LunarLander-v2, CartPole-v1
        help="the id of the env (must by gym-registered)",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="lambda used for computing GAE"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="number of parallel envs used for data collection",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1024,
        help="number of steps to run each env to fill the buffer",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="number of transition samples in each mini-batch",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="total number of timesteps executed in the environment",
    )
    parser.add_argument(
        "--updates-per-batch",
        type=int,
        default=4,
        help="number of times data buffer is iterated to update the networks",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="optimizer's learning rate"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )

    # actor-critic arguments
    parser.add_argument(
        "--clip-coef", type=float, default=0.2, help="ppo's surrogate loss coefficient"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient of value loss"
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="ppo's entropy loss coefficient"
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="controls whether ppo uses a clipped value loss or not",
    )
    parser.add_argument(
        "--target-kl", type=float, default=None, help="target KL-divergence threshold"
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )

    args = parser.parse_args()

    # ! Sanity checks:
    assert (args.num_envs * args.num_steps) % args.minibatch_size == 0

    # namedtuple is hashable and can be passed as an static argument to a jitted function
    args = namedtuple(typename="Args", field_names=list(vars(args).keys()))(
        **vars(args)
    )
    return args
