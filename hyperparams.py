import argparse
from distutils.util import strtobool


def get_args(arg_set: str = 'default') -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='test-name')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help="whether to capture videos of the agent (saved in the `videos` directory)")

    parser.add_argument('--env-id', type=str, default='CartPole-v1',
                        help='the id of the env (must by gym-registered)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--gae_lambda', type=float,
                        default=0.95, help='lambda used for computing GAE')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='number of parallel envs used for data collection')
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='number of steps to run each env for a rollout')
    parser.add_argument('--mini-batch-size', type=int, default=250,
                        help='number of transition samples in each mini-batch')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--iters-per-epoch', type=int, default=4,
                        help='number of times data buffer is iterated to update the networks')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="optimizer's learning rate")

    # actor-critic arguments
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help="ppo's surrogate loss coefficient")
    parser.add_argument('--vf-coef', type=float,
                        default=0.1, help="coefficient of value loss")
    parser.add_argument('--ent-coef', type=float,
                        default=0.05, help="ppo's entropy loss coefficient")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True, help="controls whether ppo uses a clipped value loss or not")
    parser.add_argument('--target-kl', type=float,
                        default=None, help="target KL-divergence threshold")

    args = parser.parse_args()

    if arg_set == 'minigrid':
        args.capture_video = True
        args.env_id = 'MiniGrid-Empty-8x8-v0'
        args.gamma = 0.95
        args.ent_coef = 0.0
        args.vf_coef = 0.5
        args.num_steps = 1000
        args.mini_batch_size = 250
        args.lr = 1e-3
        args.epochs = 25

    # ! Sanity checks:
    assert (args.num_envs * args.num_steps) % args.mini_batch_size == 0

    return args
