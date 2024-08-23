import numpy as np
import jax
import jax.numpy as jnp
from qml_benchmarks.model_utils import train
import optax
import copy

class RestrictedBoltzmannMachineOld():
    """
    A restricted Boltzmann machine generative model. The model is trained with the k-contrastive divergence (CD-k)
    algorithm.
    Args:
        n_hidden (int): The number of hidden neurons
        learning_rate (float): The learning rate for the CD-k updates
        cdiv_steps (int): The number of gibbs sampling steps used in contrastive divergence
        jit (bool): Whether to use just-in-time complilation
        batch_size (int): Size of batches used for computing parameter updates
        max_steps (int): Maximum number of training steps.
        reg (float): The L2 regularisation strength (larger implies stronger)
        convergence_interval (int or None): The number of loss values to consider to decide convergence.
            If None, training runs until the maximum number of steps.
        random_state (int): Seed used for pseudorandom number generation.

    """

    def __init__(self, n_hidden, learning_rate=0.001, cdiv_steps=1, jit=True, batch_size=32,
                 max_steps=200, reg=0.0, convergence_interval=200, random_state=42):

        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.jit = jit
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.reg = reg
        self.convergence_interval = convergence_interval
        self.cdiv_steps = cdiv_steps
        self.vmap = True
        self.max_vmap = None

        # data depended attributes
        self.params_ = None
        self.n_visible_ = None

        self.gibbs_step = jax.jit(self.gibbs_step) if jit else self.gibbs_step

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def energy(self, params, x, h):
        """
        The RBM energy function
        Args:
            params: parameter dictionay of weights and biases
            x: visible configuration
            h: hidden configuration
        Returns:
            energy (float): The energy
        """
        return -x.T @ params['W'] @ h - params['a'].T @ x - params['b'].T @ h

    def initialize(self, n_features):
        self.n_visible_ = n_features
        W = jax.random.normal(self.generate_key(), shape=(self.n_visible_, self.n_hidden)) / jnp.sqrt(self.n_visible_)
        a = jax.random.normal(self.generate_key(), shape=(self.n_visible_,)) / jnp.sqrt(self.n_visible_)
        b = jax.random.normal(self.generate_key(), shape=(self.n_hidden,)) / jnp.sqrt(self.n_visible_)
        self.params_ = {'a': a, 'b': b, 'W': W}

    def gibbs_step(self, args, i):
        """
        Perform one Gibbs steps. The format is such that it can be used with jax.lax.scan for fast compilation.
        """
        params = args[0]
        key = args[1]
        x = args[2]
        key1, key2, key3 = jax.random.split(key, 3)
        # get hidden units probs
        prob_h = jax.nn.sigmoid(x.T @ params['W'] + params['b'])
        h = jnp.array(jax.random.bernoulli(key1, p=prob_h), dtype=int)
        # get visible units probs
        prob_x = jax.nn.sigmoid(params['W'] @ h + params['a'])
        x_new = jnp.array(jax.random.bernoulli(key2, p=prob_x), dtype=int)
        return [params, key3, x_new], [x, h]

    def gibbs_sample(self, params, x_init, n_samples, key):
        """
        Sample a chain of visible and hidden configurations from a starting visible configuration x_init
        """
        carry = [params, key, x_init]
        carry, configs = jax.lax.scan(self.gibbs_step, carry, jnp.arange(n_samples))
        return configs

    def sample(self, n_samples):
        """
        sample only the visible units starting from a random configuration.
        """
        key = self.generate_key()
        x_init = jnp.array(jax.random.bernoulli(key, p=0.5, shape=(self.n_visible_,)), dtype=int)
        samples = self.gibbs_sample(self.params_, x_init, n_samples, self.generate_key())
        return jnp.array(samples[0])

    def fit(self, X):
        """
        Fit the parameters using contrastive divergence
        """
        self.initialize(X.shape[-1])
        X = jnp.array(X, dtype=int)

        # batch the relevant functions
        batched_gibbs_sample = jax.vmap(self.gibbs_sample, in_axes=(None, 0, None, 0))
        batched_energy = jax.vmap(self.energy, in_axes=(None, 0, 0))

        def c_div_loss(params, X, y, key):
            """
            contrastive divergence loss
            Args:
                params (dict): parameter dictionary
                X (array): batch of training examples
                y (array): not used; should be set to None when training
                key: jax PRNG key
            """
            keys = jax.random.split(key, X.shape[0])

            # we do not take the gradient wrt the sampling, so decouple the param dict here
            params_copy = copy.deepcopy(params)
            for key in params_copy.keys():
                params_copy[key] = jax.lax.stop_gradient(params_copy[key])

            configs = batched_gibbs_sample(params_copy, X, self.cdiv_steps + 1, keys)
            x0 = configs[0][:, 0, :]
            h0 = configs[1][:, 0, :]
            x1 = configs[0][:, -1, :]
            h1 = configs[1][:, -1, :]

            # taking the gradient of this loss is equivalent to the CD-k update
            loss = batched_energy(params, x0, h0) - batched_energy(params, x1, h1)

            return jnp.mean(loss) + self.reg * jnp.sqrt(jnp.sum(params['W'] ** 2))

        c_div_loss = jax.jit(c_div_loss) if self.jit else c_div_loss

        self.params_ = train(self, c_div_loss, optax.sgd, X, None, self.generate_key,
                             convergence_interval=self.convergence_interval)









