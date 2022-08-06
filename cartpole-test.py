# write a simple example with
    # a policy net defined in networks.py
    # an optimizer for the net
    # a rollout worker from environment.py (with gymnax cartpole)
    # an rl policy gradient loss defined in rl.py
    # a training loop to collect trajectories and update the policy
    # an evaluation method to log the total reward of one policy defined in environment.py
    # [Advance] use vmap, pmap to to collect trajectories using multiple workers