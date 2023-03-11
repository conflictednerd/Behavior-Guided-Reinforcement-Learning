from collections.abc import MutableMapping
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class DictList(MutableMapping):
    """
    A simple Dict-List hybrid that can be used to store rollouts of fixed length across multiple environments
    Can be accessed with either
        dictList['obs'] -> returns a np.array, or
        dictList[::2] -> returns a dict of np.arrays
    To set its values we must specify the indices and provide it with a dict specifying whcih fields we want to update. E.g.,
        dictList[:5] = {'obs': np.zeros(5, 64)}
    We can NOT directly changge the fields:
        dictList['obs'] = np.zeros(64) -> Error!
    Instead you should use:
        dictList[:] = {'obs': np.zeros(64)}
    """

    def __init__(self, shape: Union[Tuple, int], info: Dict) -> None:
        """
        shape: Either an integer denoting the length of the arrays or a tuple (in case of running multiple envs in parallel, the shape could be (num_envs, length))
        info: a dictionary whose keys are the fields that we want to be stored and values are their respective dimension or shape
        Example:
        info = {
            'obs': (64, 64),
            'next_obs': (64, 64),
            'action': 4,
            'reward': 1,
            'done': 1,
        }
        """
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.info = info
        self.length = 1
        for d in self.shape:
            self.length *= d

        self.data = dict()
        self.additional_data = []
        for key, val in info.items():
            val = tuple() if val == 1 else val
            self.data[key] = np.zeros(
                self.shape + val if isinstance(val, tuple) else (val,)
            )

    def flatten(
        self,
    ) -> None:
        """
        Flattens all arrays (in case self.shape is a tuple)
        shape of arrays change from ((base_shape) + (dims)) -> (-1, (dims))
        """
        for key, val in self.data.items():
            val.shape = (-1,) + val.shape[len(self.shape) :]

        self.shape = (self.length,)

    def __setitem__(self, __k, __v) -> None:
        """
        If obs is a field in the buffer whose values are 64x64 arrays, you can do this:
        dictList[0] = {'obs': np.zeros((64, 64))}
        If you also have 1d actions, you can do this:
        dictList[:2] = {'obs': np.zeros((2, 64, 64)), 'actions': np.zeros((2,))}
        """
        if isinstance(__k, str):
            # Invokation of this should be avoided: Once the np.arrays are created, they should not be tweaked with except by changing the values of a given slice.
            raise NotImplementedError
        else:
            assert set(__v.keys()).issubset(self.data.keys())
            for key, val in __v.items():
                self.data[key][__k] = val

    def __getitem__(self, __k: Union[Dict, str]):
        if isinstance(__k, str):
            return self.data[__k]
        else:
            return {key: val[__k] for key, val in self.data.items()}

    def __delitem__(self, __v) -> None:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def keys(
        self,
    ):
        return self.data.keys()

    def values(
        self,
    ):
        return self.data.values()

    def items(
        self,
    ):
        return self.data.items()

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def extend(dl1, dl2):
        """
        TODO:
        Take in two DictLists with similar keys and append their data together to create a new DictList
        Useful for aggregating multiple parallel rollout workers
        """
        pass

    def tree_flatten(
        self,
    ):
        aux_data = {
            "shape": self.shape,
            "info": self.info,
        }
        dict_data, additional_data = [], self.additional_data
        for key in self.info.keys():
            dict_data.append(self.data[key])

        children = (dict_data, additional_data)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(aux_data["shape"], aux_data["info"])
        dict_data, additional_data = children
        for key, val in zip(aux_data["info"].keys(), dict_data):
            obj.data[key] = val
        obj.additional_data = additional_data
        return obj


# ! WARNING: INCOMPLETE AND NOT TESTED
@register_pytree_node_class
class JAXDictList(MutableMapping):
    """
    Just like DictList, but with jnp arrays instead of regular np arrays
    """

    def __init__(self, shape: Union[Tuple, int], info: Dict) -> None:
        """
        shape: Either an integer denoting the length of the arrays or a tuple (in case of running multiple envs in parallel, the shape could be (num_envs, length))
        info: a dictionary whose keys are the fields that we want to be stored and values are their respective dimension or shape
        Example:
        info = {
            'obs': (64, 64),
            'next_obs': (64, 64),
            'action': 4,
            'reward': 1,
            'done': 1,
        }
        """
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.length = 1
        self.info = info
        for d in self.shape:
            self.length *= d

        self.data = dict()
        for key, val in info.items():
            val = tuple() if val == 1 else val
            self.data[key] = jnp.zeros(
                self.shape + val if isinstance(val, tuple) else (val,)
            )

        self.additional_data = []  # Contain arrays only

    def flatten(
        self,
    ) -> None:
        """
        Flattens all arrays (in case self.shape is a tuple)
        shape of arrays change from ((base_shape) + (dims)) -> (-1, (dims))
        """
        for key, val in self.data.items():
            self.data[key] = jnp.reshape(
                self.data[key], newshape=(-1,) + val.shape[len(self.shape) :]
            )

        self.shape = (self.length,)

    def __setitem__(self, __k, __v) -> None:
        """
        If obs is a field in the buffer whose values are 64x64 arrays, you can do this:
        dictList[0] = {'obs': np.zeros((64, 64))}
        If you also have 1d actions, you can do this:
        dictList[:2] = {'obs': np.zeros((2, 64, 64)), 'actions': np.zeros((2,))}
        """
        if isinstance(__k, str):
            # Invokation of this should be avoided: Once the np.arrays are created, they should not be tweaked with except by changing the values of a given slice.
            raise NotImplementedError
        else:
            assert set(__v.keys()).issubset(self.data.keys())
            for key, val in __v.items():
                self.data[key] = self.__update(self.data[key], __k, val)

    @staticmethod
    @jax.jit
    def __update(arr, idx, val):
        return arr.at[idx].set(val)

    def __getitem__(self, __k: Union[Dict, str]):
        if isinstance(__k, str):
            return self.data[__k]
        else:
            return {key: val[__k] for key, val in self.data.items()}

    def __delitem__(self, __v) -> None:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def keys(
        self,
    ):
        return self.data.keys()

    def values(
        self,
    ):
        return self.data.values()

    def items(
        self,
    ):
        return self.data.items()

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def extend(dl1, dl2):
        """
        TODO:
        Take in two DictLists with similar keys and append their data together to create a new DictList
        Useful for aggregating multiple parallel rollout workers
        """
        pass

    def tree_flatten(
        self,
    ):
        aux_data = {
            "shape": self.shape,
            "info": self.info,
        }
        dict_data, additional_data = [], self.additional_data
        for key in self.info.keys():
            dict_data.append(self.data[key])

        children = (dict_data, additional_data)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(aux_data["shape"], aux_data["info"])
        dict_data, additional_data = children
        for key, val in zip(aux_data["info"].keys(), dict_data):
            obj.data[key] = val
        obj.additional_data = additional_data
        return obj


# Uncomment this to see an illustrative example
"""
# d = DictList((5, 3), {'obs': (2,2), 'act': 1, 'rew': 1})
d = DictList(5, {'obs': (2,2), 'act': 1, 'rew': 1})
print(d.shape)
print(len(d))
print(d.keys())
print(d['obs'].shape)
print(d['act'].shape)
print(d['rew'].shape)

d[::2] = {'obs': -np.random.randn(3, 2, 2), 'rew': [1, 3, 5]}
print()
print(d['obs'])
print(d['act'])
print(d['rew'])
print()
print(d[:-1:2])

print()
print()
d.flatten()
print(d.shape)
print(len(d))
print(d.keys())
print(d['obs'].shape)
print(d['act'].shape)
print(d['rew'].shape)
print()
print(d['obs'])
print(d['act'])
print(d['rew'])
"""
