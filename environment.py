
def rollout_worker(env, policy, batch_size):
    '''
    Given an env and a policy, rollout the policy to collect (s, logits, a, reward2go) tuples
    Outputs a tuple (obs, logits, a, r2g) where each element is a batch of length batch_size
    '''

    # while less than batch size:
        # run an episode (until done or env.max_steps)
        # compute reward2gos
        # add data to batch
        # repeat



# Trajectory processing functions