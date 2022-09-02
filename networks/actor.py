from typing import Callable, Iterable, Optional

import haiku as hk
import jax.numpy as jnp
from numpyro.distributions.discrete import Categorical

from networks.common import MLP


class MLPActor(hk.Module):
    '''
    Serves as a general recipe for creating custom networks
    '''

    def __init__(self,
                 hidden_dims: Iterable[int],
                 num_actions: int,
                 w_init: Optional[hk.initializers.Initializer] = None,
                 b_init: Optional[hk.initializers.Initializer] = None,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh,
                 activate_final: bool = False,
                 name: Optional[str] = None
                 ) -> None:
        super().__init__(name=name)
        self.mlp = MLP(hidden_dims, num_actions, w_init, b_init,
                       activation, activate_final, name='mlp')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        logits = self.mlp(x)
        return Categorical(logits=logits)
