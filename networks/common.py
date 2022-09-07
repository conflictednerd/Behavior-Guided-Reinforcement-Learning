from typing import Callable, Iterable, Optional

import haiku as hk
import jax
import jax.numpy as jnp


class MLP(hk.Module):
    '''
    Serves as a general recipe for creating custom networks
    '''

    def __init__(self,
                 hidden_dims: Iterable[int],
                 out_dim: int,
                 w_init: Optional[hk.initializers.Initializer] = None,
                 b_init: Optional[hk.initializers.Initializer] = None,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh,
                 activate_final: bool = False,
                 name: Optional[str] = None
                 ) -> None:
        super().__init__(name=name)
        self.w_init = w_init
        self.b_init = b_init
        self.activation = activation
        self.activate_final = activate_final
        self.hidden_dims = tuple(hidden_dims)
        self.out_dim = out_dim
        self.layers = []
        for index, output_size in enumerate(self.hidden_dims + (self.out_dim,)):
            self.layers.append(hk.Linear(
                output_size=output_size,
                w_init=w_init,
                b_init=b_init,
                name=f'linear_{index}',
            ))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return self.activation(x) if self.activate_final else x

# model = hk.transform(lambda x: MLP(hidden_dims=[64, 64], out_dim=2)(x))
# B, H = 64, 4
# params = model.init(rng=jax.random.PRNGKey(0), x=jnp.zeros((H,)))
# out = model.apply(params=params, x=jnp.zeros((B, H)), rng=None)
# print(f'Number of parameters: {sum(x.size for x in jax.tree_leaves(params))}')
# print(out.shape)
