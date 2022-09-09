import jax

import RL.loss
from RL.loss import policy_gradient_loss, value_loss

'''
This module contains fast and jitted versions of loss function and their gradients
TODO: Wrap in functions
'''

# Vanilla policy gradient (REINFORCE)
PG_loss = jax.jit(policy_gradient_loss, static_argnames=(
    'actor', 'use_importance_weights'))
PG_grad = jax.jit(jax.grad(policy_gradient_loss),
                  static_argnames=('actor', 'use_importance_weights'))
PG_loss_and_grad = jax.jit(jax.value_and_grad(
    policy_gradient_loss), static_argnames=('actor', 'use_importance_weights'))
value_loss_and_grad = jax.jit(jax.value_and_grad(
    value_loss), static_argnames=['critic', 'vf_coef'])

ppo_loss = jax.jit(RL.loss.ppo_loss, static_argnames=[
                   'actor', 'critic', 'clip_coef', 'ent_coef', 'vf_coef', 'clip_vloss'])
ppo_grad = jax.jit(jax.grad(RL.loss.ppo_loss, has_aux=True),
                   static_argnames=['actor', 'critic', 'clip_coef', 'ent_coef', 'vf_coef', 'clip_vloss'])
ppo_loss_and_grad = jax.jit(jax.value_and_grad(
    RL.loss.ppo_loss, has_aux=True), static_argnames=['actor', 'critic', 'clip_coef', 'ent_coef', 'vf_coef', 'clip_vloss'])
