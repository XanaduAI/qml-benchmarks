from sklearn.base import BaseEstimator, ClassifierMixin
import jax
import jax.numpy as jnp
import optax
import numpy as np
from qml_benchmarks.model_utils import train
from sklearn.preprocessing import StandardScaler
import itertools
from qml_benchmarks.model_utils import chunk_vmapped_fn

jax.config.update("jax_enable_x64", True)

sigmaZ = jnp.array([[1, 0], [0, -1]])
sigmaX = jnp.array([[0, 1], [1, 0]])
sigmaY = jnp.array([[0, -1j], [1j, 0]])


def tensor_ops(ops, idxs, n_qubits):
    """
    Returns a tensor product of two operators acting at indexes idxs in an n_qubit system
    """
    tensor_op = 1.0
    for i in range(n_qubits):
        if i in idxs:
            j = idxs.index(i)
            tensor_op = jnp.kron(tensor_op, ops[j])
        else:
            tensor_op = jnp.kron(tensor_op, jnp.eye(2))
    return tensor_op


class QuantumBoltzmannMachine(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        visible_qubits="single",
        observable_type="sum",
        temperature=1,
        learning_rate=0.001,
        batch_size=32,
        max_vmap=None,
        jit=True,
        max_steps=10000,
        convergence_threshold=1e-6,
        random_state=42,
        scaling=1.0,
    ):
        """
        Variational Quantum Boltzmann Machine from https://arxiv.org/abs/2006.06004

        The model works as follows
        1. One prepares a gibbs state :math:`e^{-H(\vec{theta},x)/K_b T}/Z`, where :math:`H(\theta,x)` is a parameterised n_qubit
            Hamiltonian and :math:`Z` is the partition function normalisation. Here we take n_qubits equal to the number of features
        2. A :math:`\pm1` valued observable :math:`O` is measured on a subset of qubits (called 'visible qubits'). The forward
            function of the model is then

        .. maths::

            f(x, \theta) = Tr[O e^{-H(\vec{theta},x)/k_b T}/Z]

        from this expectation value, the class probabilities are computed and used in a binary cross entropy loss.

        The specific Hamiltonian we use is a generalisation of the one used in the plots:

        .. maths::

            H(x, \theta) = \sum_i Z_i \theta_i\cdot x + \sum_i X_i \theta_{i+n_qubits}\cdot x + \sum_{ij} Z_iZ_j \theta_{i+2*n_qubits}\cdot x

        The observable use can be either a sum or tensor product of Z operators on the visible qubits.

        In practice, the full algorithm involves parameterising a trial state for the gibbs state and performing variational imaginary
        time evolution to approximate the true gibbs state in each optimisation step. Since this is quite computationally involved, we
        here assume (as they do in the plots numerics) that we have access to the perfect Gibbs state. It is thus unclear whether the full
        algorithm can be expected to perform as well as this implementation.

        Args:
            visible_qubits (str): The subset of qubits used for prediction. if 'single' a single qubit is used
                if 'half' half are used and if 'all' all are used.
            observable_type (str): If 'sum' a sum of Z operators is used, if 'product' a tensor product is used.
            temperature (int): The temperature of the Gibbs state in units of K_bT. e.g. temperature = 2 is equivalent to K_bT=2.
            learning_rate (float): learning rate for gradient descent.
            batch_size (int): Number of data points to subsample in each training step.
            max_vmap (int): The largest size of vmap used (to control memory).
            jit (bool): Whether to use just in time compilation.
            convergence_threshold (float): If loss changes less than this threshold for 10 consecutive steps we stop training.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            random_state (int): Seed used for pseudorandom number generation
            scaling (float): The data is scaled by this factor after standardisation
        """

        # attributes that do not depend on data
        self.visible_qubits = visible_qubits
        self.observable_type = observable_type
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_vmap = max_vmap
        self.jit = jit
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.batch_size = batch_size
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.n_qubits = None
        self.n_visible = None  # number of visible qubits
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_model(self):
        singles = list(itertools.combinations(np.arange(self.n_qubits), 1))
        doubles = list(itertools.combinations(np.arange(self.n_qubits), 2))
        self.n_params_ = 2 * len(singles) + len(doubles)

        if self.observable_type == "sum":
            obs = (
                sum([tensor_ops([sigmaZ], (i,), self.n_qubits) for i in range(self.n_visible)])
                / self.n_visible
            )
        elif self.observable_type == "product":
            obs = 1.0
            for i in range(self.n_visible):
                obs = jnp.kron(obs, sigmaZ)

        def gibbs_state(thetas, x):

            H = jnp.zeros([2**self.n_qubits, 2**self.n_qubits])
            count = 0
            for idxs in singles:
                H = H + tensor_ops([sigmaZ], idxs, self.n_qubits) * jnp.dot(thetas[count], x)
                count = count + 1
                H = H + tensor_ops([sigmaX], idxs, self.n_qubits) * jnp.dot(thetas[count], x)
                count = count + 1

            for idxs in doubles:
                H = H + tensor_ops([sigmaZ, sigmaZ], idxs, self.n_qubits) * jnp.dot(
                    thetas[count], x
                )

            state = jax.scipy.linalg.expm(-H / self.temperature, max_squarings=32)
            return state / jnp.trace(state)

        def model(thetas, x):
            state = gibbs_state(thetas, x)
            return jnp.trace(jnp.matmul(state, obs))

        if self.jit:
            model = jax.jit(model)
        self.forward = jax.vmap(model, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize(self, n_features, classes=None):
        """Initialize attributes that depend on the number of features and the class labels.

        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels that the classifier expects
        """
        if classes is None:
            classes = [-1, 1]

        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_

        self.n_qubits = n_features

        if self.visible_qubits == "single":
            self.n_visible = 1
        elif self.visible_qubits == "half":
            self.n_visible = self.n_qubits // 2
        elif self.visible_qubits == "all":
            self.n_visible = self.n_qubits

        self.construct_model()
        self.initialize_params()

    def initialize_params(self):
        # initialise the trainable parameters
        params = jax.random.normal(shape=(self.n_params_, self.n_qubits), key=self.generate_key())
        self.params_ = {"thetas": params}

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """
        self.initialize(X.shape[1], classes=np.unique(y))

        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.transform(X)

        optimizer = optax.adam

        def loss_fn(params, X, y):
            # binary cross entropy loss
            vals = self.forward(params["thetas"], X)
            probs = (1 + vals) / 2
            y = jax.nn.relu(y)  # convert to 0,1
            return jnp.mean(-y * jnp.log(probs) - (1 - y) * jnp.log(1 - probs))

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        self.params_ = train(self, loss_fn, optimizer, X, y, self.generate_key)

        return self

    def predict(self, X):
        """Predict labels for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred (np.ndarray): Predicted labels of shape (n_samples,)
        """
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        X = self.transform(X)
        predictions = self.forward(self.params_["thetas"], X)
        predictions_2d = np.c_[(1 - predictions) / 2, (1 + predictions) / 2]
        return predictions_2d

    def transform(self, X):
        """
        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        if self.scaler is None:
            # if the model is unfitted, initialise the scaler here
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        return self.scaler.transform(X) * self.scaling


class QuantumBoltzmannMachineSeparable(QuantumBoltzmannMachine):

    def construct_model(self):
        def qubit_gibbs_state(thetas, x):
            H = sigmaZ * jnp.dot(thetas[0], x) + sigmaX * jnp.dot(thetas[1], x)
            state = jax.scipy.linalg.expm(-H / self.temperature, max_squarings=32)
            return state / jnp.trace(state)

        def model(thetas, x):
            gibbs_states = [
                qubit_gibbs_state(thetas[2 * i : 2 * i + 2, :], x) for i in range(self.n_visible)
            ]
            expvals = jnp.array([jnp.trace(jnp.matmul(state, sigmaZ)) for state in gibbs_states])
            if self.observable_type == "sum":
                return jnp.mean(expvals)
            elif self.observable_type == "product":
                return jnp.prod(expvals)

        if self.jit:
            model = jax.jit(model)
        self.forward = jax.vmap(model, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize_params(self):
        # initialise the trainable parameters
        params = jax.random.normal(
            shape=(2 * self.n_qubits, self.n_qubits), key=self.generate_key()
        )
        self.params_ = {"thetas": params}
