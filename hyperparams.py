import argparse
from distutils.util import strtobool


def get_args(arg_set: str = 'default') -> argparse.Namespace:
    if arg_set == 'default':
        parser = argparse.ArgumentParser()
        parser.add_argument('--exp-name', type=str, default='test-name')
        parser.add_argument('--seed', type=int, default=23)
        parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                            help="whether to capture videos of the agent (saved in the `videos` directory)")

        parser.add_argument('--env-id', type=str, default='CartPole-v1',
                            help='the id of the env (must by gym-registered)')
        parser.add_argument('--gamma', type=float, default=0.99,
                            help='discount factor')
        parser.add_argument('--num-envs', type=int, default=4,
                            help='number of parallel envs used for data collection')
        parser.add_argument('--num-steps', type=int, default=1024,
                            help='number of steps to run each env for a rollout')
        parser.add_argument('--mini-batch-size', type=int, default=256,
                            help='number of transition samples in each mini-batch')
        parser.add_argument('--epochs', type=int, default=10,
                            help='number of training epochs')
        parser.add_argument('--iters-per-epoch', type=int, default=4,
                            help='number of times data buffer is iterated to update the networks')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help="optimizer's learning rate")

        args = parser.parse_args()
        # ! Sanity checks:
        assert (args.num_envs * args.num_steps) % args.mini_batch_size == 0
        return args
    else:
        raise NotImplementedError
