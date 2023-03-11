from typing import List, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from distrax._src.distributions import distribution


class Actor(nn.Module):
    hidden_dims: Tuple[int]
    num_actions: int

    def setup(self) -> None:
        self.layers = [nn.Dense(h) for h in self.hidden_dims]
        self.last_layer = nn.Dense(self.num_actions)

    def __call__(self, obs) -> distribution.Distribution:
        for layer in self.layers:
            obs = nn.leaky_relu(layer(obs))
        logits = self.last_layer(obs)
        action_dist = distrax.Categorical(logits=logits)
        return action_dist


class Critic(nn.Module):
    hidden_dims: Tuple[int]

    def setup(self) -> None:
        self.layers = [nn.Dense(h) for h in self.hidden_dims]
        self.last_layer = nn.Dense(1)

    def __call__(self, obs) -> jax.Array:
        for layer in self.layers:
            obs = nn.leaky_relu(layer(obs))
        return self.last_layer(obs)


class ActorCritic(nn.Module):
    actor_dims: Tuple[int]
    critic_dims: Tuple[int]
    num_actions: int
    """
    To get only the actions or the values, we can use:
        model.apply({'params': params}, obs, method=model.get_action)
        model.apply({'params': params}, obs, method=model.get_value)
    """

    def setup(self) -> None:
        self.actor = Actor(self.actor_dims, self.num_actions)
        self.critic = Critic(self.critic_dims)

    def __call__(self, obs) -> Tuple[distribution.Distribution, jax.Array]:
        return self.actor(obs), self.critic(obs)

    def get_action(self, obs) -> distribution.Distribution:
        return self.actor(obs)

    def get_value(self, obs) -> jax.Array:
        return self.critic(obs)
