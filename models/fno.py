import jax
from jax import random, grad, vmap, jit
import jax.numpy as jnp
import optax
import equinox as eqx

from functools import partial

class FNO(eqx.Module):
    pass

