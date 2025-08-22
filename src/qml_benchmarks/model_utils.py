# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions shared by models."""

import operator
from functools import reduce
import logging
import time
import numpy as np
import optax
import jax
import jax.numpy as jnp
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import gen_batches
import inspect
from tqdm import tqdm


def train(
    model, loss_fn, optimizer, X, y, random_key_generator, convergence_interval=200
):
    """
    Trains a model using an optimizer and a loss function via gradient descent. We assume that the loss function
    is of the form `loss(params, X, y, key)` and that the trainable parameters are stored in model.params_ as a dictionary
    of jnp.arrays. The optimizer should be an Optax optimizer (e.g. optax.adam). `model` must have an attribute
    `learning_rate` to set the initial learning rate for the gradient descent.

    The model is trained until a convergence criterion is met that corresponds to the loss curve being flat over
    a number of optimization steps given by `convergence_inteval` (see plots for details).

    To reduce precompilation time and memory cost, the loss function and gradient functions are evaluated in
    chunks of size model.max_vmap.

    Args:
        model (class): Classifier class object to train. Trainable parameters must be stored in model.params_.
        loss_fn (Callable): Loss function to be minimised. Must be of the form loss_fn(params, X, y, key).
        optimizer (optax optimizer): Optax optimizer (e.g. optax.adam).
        X (array): Input data array of shape (n_samples, n_features)
        y (array, optional): Array of shape (n_samples) containing the labels.
        random_key_generator (jax.random.PRNGKey): JAX key generator object for pseudo-randomness generation.
        convergence_interval (int, optional): Number of optimization steps over which to decide convergence. Larger
            values give a higher confidence that the model has converged but may increase training time.

    Returns:
        params (dict): The new parameters after training has completed.
    """

    if model.max_vmap is not None:
        if not model.batch_size / model.max_vmap % 1 == 0:
            raise Exception("Batch size must be multiple of max_vmap.")

    # wrap a key around the function if it doesn't have one
    if "key" not in inspect.signature(loss_fn).parameters:

        def loss_fn_wrapped(params, x, y, key):
            return loss_fn(params, x, y)

    else:
        loss_fn_wrapped = loss_fn

    params = model.params_
    opt = optimizer(learning_rate=model.learning_rate)
    opt_state = opt.init(params)
    grad_fn = jax.grad(loss_fn_wrapped)

    # jitting through the chunked_grad function can take a long time,
    # so we jit here and chunk after
    if model.jit:
        grad_fn = jax.jit(grad_fn)

    # note: assumes that the loss function is a sample mean of
    # some function over the input data set
    chunked_grad_fn = (
        chunk_grad(grad_fn, model.max_vmap) if model.max_vmap is not None else grad_fn
    )
    chunked_loss_fn = (
        chunk_loss(loss_fn_wrapped, model.max_vmap)
        if model.max_vmap is not None
        else loss_fn_wrapped
    )

    def update(params, opt_state, x, y, key):
        grads = chunked_grad_fn(params, x, y, key)
        loss_val = chunked_loss_fn(params, x, y, key)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    loss_history = []
    converged = False
    start = time.time()
    with tqdm(total=model.max_steps, desc="Training Progress") as pbar:
        for step in range(model.max_steps):
            key_batch = random_key_generator()
            key_loss = jax.random.split(key_batch, 1)[0]
            X_batch, y_batch = get_batch(X, y, key_batch, batch_size=model.batch_size)
            params, opt_state, loss_val = update(
                params, opt_state, X_batch, y_batch, key_loss
            )
            loss_history.append(loss_val)
            logging.debug(f"{step} - loss: {loss_val}")
            pbar.update(1)

            if np.isnan(loss_val):
                logging.info(f"nan encountered. Training aborted.")
                break

            # decide convergence
            if convergence_interval is not None:
                if step > 2 * convergence_interval:
                    # get means of last two intervals and standard deviation of last interval
                    average1 = np.mean(loss_history[-convergence_interval:])
                    average2 = np.mean(
                        loss_history[-2 * convergence_interval : -convergence_interval]
                    )
                    std1 = np.std(loss_history[-convergence_interval:])
                    # if the difference in averages is small compared to the statistical fluctuations, stop training.
                    if (
                        np.abs(average2 - average1)
                        <= std1 / np.sqrt(convergence_interval) / 2
                    ):
                        logging.info(
                            f"Model {model.__class__.__name__} converged after {step} steps."
                        )
                        converged = True
                        break

    end = time.time()
    loss_history = np.array(loss_history)
    model.loss_history_ = loss_history / np.max(np.abs(loss_history))
    model.training_time_ = end - start

    if not converged and convergence_interval is not None:
        print("Loss did not converge:", loss_history)
        raise ConvergenceWarning(
            f"Model {model.__class__.__name__} has not converged after the maximum number of {model.max_steps} steps."
        )

    return params


def get_batch(X, y, rnd_key, batch_size=32):
    """
    A generator to get random batches of the data (X, y)

    Args:
        X (array[float]): Input data with shape (n_samples, n_features).
        y (array[float]): Target labels with shape (n_samples,)
        rnd_key: A jax random key object
        batch_size (int): Number of elements in batch

    Returns:
        array[float]: A batch of input data shape (batch_size, n_features)
        array[float]: A batch of target labels shaped (batch_size,)
    """
    all_indices = jnp.array(range(len(X)))
    rnd_indices = jax.random.choice(
        key=rnd_key, a=all_indices, shape=(batch_size,), replace=True
    )

    if y is not None:
        return X[rnd_indices], y[rnd_indices]
    else:
        return X[rnd_indices], None


def get_from_dict(dict, key_list):
    """
    Access a value from a nested dictionary.
    Inspired by https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        dict (dict): nested dictionary
        key_list (list): list of keys to be accessed

    Returns:
         the requested value
    """
    return reduce(operator.getitem, key_list, dict)


def set_in_dict(dict, keys, value):
    """
    Set a value in a nested dictionary.

    Args:
        dict (dict): nested dictionary
        keys (list): list of keys in nested dictionary
        value (Any): value to be set

    Returns:
        nested dictionary with new value
    """
    for key in keys[:-1]:
        dict = dict.setdefault(key, {})
    dict[keys[-1]] = value


def get_nested_keys(d, parent_keys=[]):
    """
    Returns the nested keys of a nested dictionary.

    Args:
        d (dict): nested dictionary

    Returns:
        list where each element is a list of nested keys
    """
    keys_list = []
    for key, value in d.items():
        current_keys = parent_keys + [key]
        if isinstance(value, dict):
            keys_list.extend(get_nested_keys(value, current_keys))
        else:
            keys_list.append(current_keys)
    return keys_list


def chunk_vmapped_fn(vmapped_fn, start, max_vmap):
    """
    Convert a vmapped function to an equivalent function that evaluates in chunks of size
    max_vmap. The behaviour of chunked_fn should be the same as vmapped_fn, but with a
    lower memory cost.

    The input vmapped_fn should have in_axes = (None, None, ..., 0,0,...,0)

    Args:
        vmapped (func): vmapped function with in_axes = (None, None, ..., 0,0,...,0)
        start (int): The index where the first 0 appears in in_axes
        max_vmap (int) The max chunk size with which to evaluate the function

    Returns:
        chunked version of the function
    """

    def chunked_fn(*args):
        batch_len = len(args[start])
        batch_slices = list(gen_batches(batch_len, max_vmap))
        res = [
            vmapped_fn(*args[:start], *[arg[slice] for arg in args[start:]])
            for slice in batch_slices
        ]
        # jnp.concatenate needs to act on arrays with the same shape, so pad the last array if necessary
        if batch_len / max_vmap % 1 != 0.0:
            diff = max_vmap - len(res[-1])
            res[-1] = jnp.pad(
                res[-1], [(0, diff), *[(0, 0)] * (len(res[-1].shape) - 1)]
            )
            return jnp.concatenate(res)[:-diff]
        else:
            return jnp.concatenate(res)

    return chunked_fn


def chunk_grad(grad_fn, max_vmap):
    """
    Convert a `jax.grad` function to an equivalent version that evaluated in chunks of size max_vmap.

    `grad_fn` should be of the form `jax.grad(fn(params, X, y, key), argnums=0)`, where `params` is a
    dictionary of `jnp.arrays`, `X, y` are `jnp.arrays` with the same-size leading axis, and `grad_fn`
    is a function that is vectorised along these axes (i.e. `in_axes = (None,0,0)`).

    The returned function evaluates the original function by splitting the batch evaluation into smaller chunks
    of size `max_vmap`, and has a lower memory footprint.

    Args:
        model (func): gradient function with the functional form jax.grad(loss(params, X,y), argnums=0)
        max_vmap (int): the size of the chunks

    Returns:
        chunked version of the function
    """

    def chunked_grad(params, X, y, key):
        batch_slices = list(gen_batches(len(X), max_vmap))
        grads = [grad_fn(params, X[slice], y[slice], key) for slice in batch_slices]
        grad_dict = {}
        for key_list in get_nested_keys(params):
            set_in_dict(
                grad_dict,
                key_list,
                jnp.mean(
                    jnp.array([get_from_dict(grad, key_list) for grad in grads]), axis=0
                ),
            )
        return grad_dict

    return chunked_grad


def chunk_loss(loss_fn, max_vmap):
    """
    Converts a loss function of the form `loss_fn(params, array1, array2, key)` to an equivalent version that
    evaluates `loss_fn` in chunks of size max_vmap. `loss_fn` should batch evaluate along the leading
    axis of `array1, array2` (i.e. `in_axes = (None,0,0)`).

    Args:
        loss_fn (func): function of form loss_fn(params, array1, array2)
        max_vmap (int): maximum chunk size

    Returns:
        chunked version of the function
    """

    def chunked_loss(params, X, y, key):
        batch_slices = list(gen_batches(len(X), max_vmap))
        res = jnp.array(
            [loss_fn(params, *[X[slice], y[slice]], key) for slice in batch_slices]
        )
        return jnp.mean(res)

    return chunked_loss


def mmd_loss(
    ground_truth: np.ndarray, model_samples: np.ndarray, sigma: float
) -> float:
    """Calculates an unbiased estimate of the Maximum Mean Discrepancy (MMD) loss from samples
    see https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf for more info

    Args:
        ground_truth (np.ndarray): Samples from the ground truth distribution.
        model_samples (np.ndarray): Samples from the test model.
        sigma (float): Sigma parameter, the width of the kernel.

    Returns:
        float: The value of the MMD loss.
    """

    n = len(ground_truth)
    m = len(model_samples)
    ground_truth = jnp.array(ground_truth)
    model_samples = jnp.array(model_samples)

    # K_pp
    K_pp = jnp.zeros((ground_truth.shape[0], ground_truth.shape[0]))

    def body_fun(i, val):
        def inner_body_fun(j, inner_val):
            return inner_val.at[i, j].set(
                gaussian_kernel(sigma, ground_truth[i], ground_truth[j])
            )

        return jax.lax.fori_loop(0, ground_truth.shape[0], inner_body_fun, val)

    K_pp = jax.lax.fori_loop(0, ground_truth.shape[0], body_fun, K_pp)
    sum_pp = jnp.sum(K_pp) - n

    # K_pq
    K_pq = jnp.zeros((ground_truth.shape[0], model_samples.shape[0]))

    def body_fun(i, val):
        def inner_body_fun(j, inner_val):
            return inner_val.at[i, j].set(
                gaussian_kernel(sigma, ground_truth[i], model_samples[j])
            )

        return jax.lax.fori_loop(0, model_samples.shape[0], inner_body_fun, val)

    K_pq = jax.lax.fori_loop(0, ground_truth.shape[0], body_fun, K_pq)
    sum_pq = jnp.sum(K_pq)

    # K_qq
    K_qq = jnp.zeros((model_samples.shape[0], model_samples.shape[0]))

    def body_fun(i, val):
        def inner_body_fun(j, inner_val):
            return inner_val.at[i, j].set(
                gaussian_kernel(sigma, model_samples[i], model_samples[j])
            )

        return jax.lax.fori_loop(0, model_samples.shape[0], inner_body_fun, val)

    K_qq = jax.lax.fori_loop(0, model_samples.shape[0], body_fun, K_qq)
    sum_qq = jnp.sum(K_qq) - m

    return 1 / n / (n - 1) * sum_pp - 2 / n / m * sum_pq + 1 / m / (m - 1) * sum_qq


def gaussian_kernel(sigma: float, x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the value for the gaussian kernel between two vectors x, y

    Args:
        sigma (float): The standard deviation of the Gaussian defining the kernel function
        x (np.ndarray): one of the vectors
        y (np.ndarray): the other vector

    Returns:
        float: Result value of the gaussian kernel
    """
    return jnp.exp(-((x - y) ** 2).sum() / 2 / sigma**2)


def median_heuristic(X):
    """
    Computes an estimate of the median heuristic used to decide the bandwidth of the RBF kernels; see
    https://arxiv.org/abs/1707.07269
    :param X (array): Dataset of interest
    :return (float): median heuristic estimate
    """
    m = len(X)
    X = np.array(X)
    med = np.median([np.sqrt(np.sum((X[i] - X[j]) ** 2)) for i in range(m) for j in range(m)])
    return med