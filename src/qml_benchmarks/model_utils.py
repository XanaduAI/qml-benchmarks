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

import pennylane as qml
import catalyst
from catalyst import qjit
import numpy as np
import optax
import jax
import jax.numpy as jnp
from pennylane import numpy as pnp
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import gen_batches


def train(model, loss_fn, optimizer, X, y, random_key_generator, convergence_interval=200):
    """
    Trains a model using an optimizer and a loss function via gradient descent. We assume that the loss function
    is of the form `loss(params, X, y)` and that the trainable parameters are stored in model.params_ as a dictionary
    of jnp.arrays. The optimizer should be an Optax optimizer (e.g. optax.adam). `model` must have an attribute
    `learning_rate` to set the initial learning rate for the gradient descent.

    The model is trained until a convergence criterion is met that corresponds to the loss curve being flat over
    a number of optimization steps given by `convergence_inteval` (see plots for details).

    To reduce precompilation time and memory cost, the loss function and gradient functions are evaluated in
    chunks of size model.max_vmap.

    Args:
        model (class): Classifier class object to train. Trainable parameters must be stored in model.params_.
        loss_fn (Callable): Loss function to be minimised. Must be of the form loss_fn(params, X, y).
        optimizer (optax optimizer): Optax optimizer (e.g. optax.adam).
        X (array): Input data array of shape (n_samples, n_features)
        y (array): Array of shape (n_samples) containing the labels.
        random_key_generator (jax.random.PRNGKey): JAX key generator object for pseudo-randomness generation.
        convergence_interval (int, optional): Number of optimization steps over which to decide convergence. Larger
            values give a higher confidence that the model has converged but may increase training time.

    Returns:
        params (dict): The new parameters after training has completed.
    """

    if not model.batch_size / model.max_vmap % 1 == 0:
        raise Exception("Batch size must be multiple of max_vmap.")

    params = model.params_
    opt = optimizer(learning_rate=model.learning_rate)
    opt_state = opt.init(params)
    grad_fn = jax.grad(loss_fn)

    # jitting through the chunked_grad function can take a long time,
    # so we jit here and chunk after
    if model.jit:
        grad_fn = jax.jit(grad_fn)

    # note: assumes that the loss function is a sample mean of
    # some function over the input data set
    chunked_grad_fn = chunk_grad(grad_fn, model.max_vmap)
    chunked_loss_fn = chunk_loss(loss_fn, model.max_vmap)

    def update(params, opt_state, x, y):
        grads = chunked_grad_fn(params, x, y)
        loss_val = chunked_loss_fn(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    loss_history = []
    steptimes = []
    converged = False
    start = time.time()
    for step in range(model.max_steps):
        key = random_key_generator()
        X_batch, y_batch = get_batch(X, y, key, batch_size=model.batch_size)
        params, opt_state, loss_val = update(params, opt_state, X_batch, y_batch)
        steptimes.append(time.time() - start)
        loss_history.append(loss_val)
        logging.debug(f"{step} - loss: {loss_val}")

        if np.isnan(loss_val):
            logging.info(f"nan encountered. Training aborted.")
            break

        # decide convergence
        if step > 2 * convergence_interval:
            # get means of last two intervals and standard deviation of last interval
            average1 = np.mean(loss_history[-convergence_interval:])
            average2 = np.mean(loss_history[-2 * convergence_interval : -convergence_interval])
            std1 = np.std(loss_history[-convergence_interval:])
            # if the difference in averages is small compared to the statistical fluctuations, stop training.
            if np.abs(average2 - average1) <= std1 / np.sqrt(convergence_interval) / 2:
                logging.info(f"Model {model.__class__.__name__} converged after {step} steps.")
                converged = True
                break

    end = time.time()
    loss_history / np.max(np.abs(loss_history))
    loss_history = np.array(np.vstack((loss_history,steptimes)))
    model.loss_history_ = loss_history
    model.training_time_ = end - start

    #removed for perforance profiling
    #
    # if not converged:
    #     print("Loss did not converge:", loss_history)
    #     raise ConvergenceWarning(
    #         f"Model {model.__class__.__name__} has not converged after the maximum number of {model.max_steps} steps."
    #     )

    return params


def train_without_jax(
    model, loss_fn, optimizer, X, y, random_key_generator, convergence_interval=200
):
    """Trains a model using an optimizer and a loss function, using PennyLane's autograd interface."""

    params = list(model.params_.values())
    opt = optimizer(stepsize=model.learning_rate)

    loss_history = []
    converged = False
    start = time.time()
    for step in range(model.max_steps):
        key = random_key_generator()
        X_batch, y_batch = get_batch_without_jax(X, y, key, batch_size=model.batch_size)
        X_batch = pnp.array(X_batch, requires_grad=False)
        y_batch = pnp.array(y_batch, requires_grad=False)
        loss_val = loss_fn(*params, X_batch, y_batch)
        params = opt.step(loss_fn, *params, X_batch, y_batch)[: len(params)]
        loss_history.append(loss_val)

        logging.debug(f"{step} - loss: {loss_val}")

        if np.isnan(loss_val):
            logging.info(f"nan encountered. Training aborted.")
            break

        if step > 2 * convergence_interval:
            average1 = np.mean(loss_history[-convergence_interval:])
            average2 = np.mean(loss_history[-2 * convergence_interval : -convergence_interval])
            std1 = np.std(loss_history[-convergence_interval:])
            if np.abs(average2 - average1) <= std1 / np.sqrt(convergence_interval) / 2:
                logging.info(f"Model {model.__class__.__name__} converged after {step} steps.")
                converged = True
                break

    end = time.time()
    loss_history = np.array(loss_history)
    model.loss_history_ = loss_history / np.max(np.abs(loss_history))
    model.training_time_ = end - start

    if not converged:
        raise ConvergenceWarning(
            f"Model {model.__class__.__name__} has not converged after the maximum number of {model.max_steps} steps."
        )

    for i, key in enumerate(model.params_.keys()):
        model.params_[key] = params[i]

    return model.params_


def train_with_catalyst(model, loss_fn, optimizer, X, y, random_key_generator, convergence_interval=200):

    params = model.params_
    opt = optimizer(learning_rate=model.learning_rate)

    @qjit
    def update(i, args):
        params, opt_state, X, y, loss_history, key = args
        X_batch, y_batch = get_batch(X, y, key, batch_size=model.batch_size)
        loss = loss_fn(params, X_batch, y_batch)
        loss_history =  loss_history.at[i].set(loss)
        #backprop is not supported in catalyst yet, falling back on finite diff
        grads = catalyst.grad(loss_fn, method="fd")(params,X_batch, y_batch)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        key, subkey = jax.random.split(key)
        return (params, opt_state, X, y, loss_history, subkey)

    @qjit
    def optimize(params, X, y, steps, loss_history, opt_state, key):
        args = (params, opt_state, X, y, loss_history, key)
        (params, opt_state, _, _, loss_history, key) = catalyst.for_loop(0,steps,1)(update)(args)
        return params, loss_history, opt_state

    def train_until_convergence(params, X, y):
        converged = False
        current_steps = 0
        opt_state = opt.init(params)
        loss_histories = []
        while current_steps < model.max_steps:

            key = random_key_generator()
            loss_history = jnp.zeros(convergence_interval)
            params, loss_history, opt_state = optimize(params, X, y, convergence_interval, loss_history, opt_state, key)
            loss_histories.append(loss_history)
            current_steps += convergence_interval

            if True in jnp.isnan(loss_history):
                logging.info(f"nan encountered. Training aborted.")
                break

            if len(loss_histories)>=2:
                average1 = jnp.mean(loss_histories[-1])
                average2 = jnp.mean(loss_histories[-2])
                std1 = jnp.std(loss_history[-1])
                if jnp.abs(average2 - average1) <= std1 / jnp.sqrt(convergence_interval) / 2:
                    logging.info(f"Model {model.__class__.__name__} converged after {step} steps.")
                    converged = True
                    break

        if not converged:
            print("Loss did not converge:", loss_history)
            raise ConvergenceWarning(
                f"Model {model.__class__.__name__} has not converged after the maximum number of {model.max_steps} steps."
            )

        return params, jnp.concatenate(loss_histories)

    start = time.time()
    params, loss_history = train_until_convergence(params, X, y)
    end = time.time()
    loss_history = np.array(loss_history)
    model.loss_history_ = loss_history / np.max(np.abs(loss_history))
    model.training_time_ = end - start

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
    rnd_indices = jax.random.choice(key=rnd_key, a=all_indices, shape=(batch_size,), replace=True)
    return X[rnd_indices], y[rnd_indices]


def get_batch_without_jax(X, y, rnd_key, batch_size=32):
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
    all_indices = list(range(len(X)))
    rnd_indices = np.random.choice(all_indices, size=(batch_size,), replace=True)
    return X[rnd_indices], y[rnd_indices]


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
            res[-1] = jnp.pad(res[-1], [(0, diff), *[(0, 0)] * (len(res[-1].shape) - 1)])
            return jnp.concatenate(res)[:-diff]
        else:
            return jnp.concatenate(res)

    return chunked_fn


def chunk_grad(grad_fn, max_vmap):
    """
    Convert a `jax.grad` function to an equivalent version that evaluated in chunks of size max_vmap.

    `grad_fn` should be of the form `jax.grad(fn(params, X, y), argnums=0)`, where `params` is a
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

    def chunked_grad(params, X, y):
        batch_slices = list(gen_batches(len(X), max_vmap))
        grads = [grad_fn(params, X[slice], y[slice]) for slice in batch_slices]
        grad_dict = {}
        for key_list in get_nested_keys(params):
            set_in_dict(
                grad_dict,
                key_list,
                jnp.mean(jnp.array([get_from_dict(grad, key_list) for grad in grads]), axis=0),
            )
        return grad_dict

    return chunked_grad


def chunk_loss(loss_fn, max_vmap):
    """
    Converts a loss function of the form `loss_fn(params, array1, array2)` to an equivalent version that
    evaluates `loss_fn` in chunks of size max_vmap. `loss_fn` should batch evaluate along the leading
    axis of `array1, array2` (i.e. `in_axes = (None,0,0)`).

    Args:
        loss_fn (func): function of form loss_fn(params, array1, array2)
        max_vmap (int): maximum chunk size

    Returns:
        chunked version of the function
    """

    def chunked_loss(params, X, y):
        batch_slices = list(gen_batches(len(X), max_vmap))
        res = jnp.array([loss_fn(params, *[X[slice], y[slice]]) for slice in batch_slices])
        return jnp.mean(res)

    return chunked_loss


####### LOSS UTILS WITHOUT JAX


def l2_loss(pred, y):
    """
    The square loss function. 0.5 is there to match optax.l2_loss.
    """
    return 0.5 * (pred - y) ** 2


def softmax(x, axis=-1):
    """
    copied from JAX: https://jax.readthedocs.io/en/latest/_modules/jax/_src/nn/functions.html#softmax
    """
    x_max = pnp.max(x, axis, keepdims=True)
    unnormalized = pnp.exp(x - x_max)
    result = unnormalized / pnp.sum(unnormalized, axis, keepdims=True)
    return result


def one_hot(a, num_classes=2):
    """
    convert an array to a one hot encoded array.
    Taken from https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
    """
    b = pnp.zeros((a.size, num_classes))
    b[pnp.arange(a.size), a] = 1
    return b


def log_softmax(x, axis=-1):
    """
    taken from jax.nn.log_softmax:
    https://jax.readthedocs.io/en/latest/_modules/jax/_src/nn/functions.html#log_softmax
    """
    x_arr = pnp.asarray(x)
    x_max = pnp.max(x_arr, axis, keepdims=True)
    x_max = pnp.array(x_max, requires_grad=False)
    shifted = x_arr - x_max
    shifted_logsumexp = pnp.log(pnp.sum(pnp.exp(shifted), axis, keepdims=True))
    result = shifted - shifted_logsumexp
    return result


def softmax_cross_entropy(logits, labels):
    """taken from optax source:
    https: // github.com / google - deepmind / optax / blob / master / optax / losses / _classification.py  # L78%23L103
    """
    return -pnp.sum(labels * log_softmax(logits, axis=-1), axis=-1)
