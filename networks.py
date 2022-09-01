from typing import Optional

import haiku as hk
import jax.numpy as jnp
import numpy as np
from numpyro.distributions.discrete import Categorical


class MLPActor(hk.Module):
    def __init__(self, hidden_size, num_actions: int = 2, name: Optional[str] = None):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.hidden_size = hidden_size

    def __call__(self, x):
        in_shape = x.shape[-1]
        w1 = hk.get_parameter("w1", shape=[in_shape, self.hidden_size], dtype=x.dtype,
                              init=hk.initializers.TruncatedNormal(1./np.sqrt(in_shape)))
        b1 = hk.get_parameter(
            "b1", shape=[self.hidden_size], dtype=x.dtype, init=jnp.ones)
        w2 = hk.get_parameter('w2', shape=[self.hidden_size, self.num_actions], dtype=x.dtype,
                              init=hk.initializers.TruncatedNormal(1./np.sqrt(self.hidden_size)))
        b2 = hk.get_parameter(
            'b2', shape=[self.num_actions], dtype=x.dtype, init=jnp.ones)

        logits = jnp.tanh(x @ w1 + b1) @ w2 + b2
        return Categorical(logits=logits)


cartNet = hk.transform(lambda x: MLPActor(64, 2)(x))
mlp = hk.transform(lambda x: hk.nets.MLP([64, 2], activation=jnp.tanh)(x))
# params = mlp.init(rng=random.PRNGKey(0), x=jnp.zeros(4))
# params = cartNet.init(rng=random.PRNGKey(0), x=jnp.zeros(4))
# x = jnp.zeros(4)
# apply = jax.jit(cartNet.apply)
# apply = jax.jit(mlp.apply)
# out = apply(params=params, x=x, rng=None)
# print(out.shape)
# out = apply(params=params, x=jnp.zeros((32, 4)), rng=None)
# print(out.shape)
