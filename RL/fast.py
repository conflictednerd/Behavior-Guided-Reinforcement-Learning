import jax

from RL.loss import policy_gradient_loss

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
