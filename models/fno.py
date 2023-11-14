import jax
from jax import random, grad, vmap, jit
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal
import optax
import equinox as eqx

from functools import partial

class FNO(eqx.Module):
    pass

class FNOBlock(eqx.Module):
    linear_W: jax.Array
    transform_R: jax.Array
    num_modes: int = eqx.field(static=True)
    d_v: int = eqx.field(static=True)

    def __init__(self, num_modes, latent_size, key: random.PRNGKey, initializer=glorot_normal(), dtype=jnp.float32):
        self.num_modes = num_modes
        self.d_v = latent_size
        key, subkey = random.split(key)
        self.linear_W = initializer(subkey, (latent_size, latent_size), dtype=dtype)
        self.transform_R = initializer(subkey, (num_modes, latent_size, latent_size), dtype=dtype)

    def __call__(self, v):
        pass

        #TODO implement this
        f_v = jax.numpy.fft.rfft(v, axis=-1)
        f_v = f_v[:self.num_modes]
        f_v = jnp.einsum('cvv,cv->cv', self.transform_R, v)
        v = v + jax.numpy.fft.irfft(f_v, axis=-1)

