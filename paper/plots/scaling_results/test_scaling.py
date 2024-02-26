from qml_benchmarks.hyperparam_search_utils import read_data
from importlib import import_module
import pandas as pd
import pickle
from joblib import Parallel, delayed

module = import_module("qml_benchmarks.models")


clf_names = [
    "CircuitCentricClassifier",
    "DataReuploadingClassifier",
    "DressedQuantumCircuitClassifier",
    "IQPVariationalClassifier",
    "QuantumMetricLearner",
    "TreeTensorClassifier",
]

dataset_paths = [
    "../../benchmarks/mnist_pca",
    "../../benchmarks/linearly_separable",
    "../../benchmarks/hidden_manifold",
    "../../benchmarks/two_curves_diff",
]

experiment_paths = [
    "../../results/mnist_pca",
    "../../results/linearly-separable",
    "../../results/hidden_manifold",
    "../../results/two_curves_diff",
]

dataset_fnames = [
    "mnist_3-5_6d",
    "linearly_separable_6d",
    "hidden_manifold-6manifold-6d",
    "two_curves-5degree-0.1offset-6d",
]

scales = [0.01, 0.1, 0.3, 0.6, 1.0, 1.5, 3.0, 10]
accs = []
clfs = []
scaling = []
datasets = []
for dataset_fname, dataset_path, experiment_path in zip(
    dataset_fnames, dataset_paths, experiment_paths
):
    for clf_name in clf_names:
        for seed in range(5):

            clf_class = getattr(module, clf_name)

            # get best hps
            file = open(
                f"{experiment_path}/{clf_name}/{clf_name}_{dataset_fname}_GridSearchCV-best-hyperparams.pickle",
                "rb",
            )
            best_hps = pickle.load(file)

            def train_model(X, y, X_test, y_test, clf_class, best_hps, seed, s):
                clf = clf_class(scaling=s, random_state=seed, **best_hps)
                print(dataset_fname)
                print(clf_class)
                clf.fit(X, y)
                return clf.score(X_test, y_test)

            X, y = read_data(f"{dataset_path}/{dataset_fname}_train.csv")
            X_test, y_test = read_data(f"{dataset_path}/{dataset_fname}_test.csv")

            result = Parallel(n_jobs=len(scales))(
                delayed(train_model)(X, y, X_test, y_test, clf_class, best_hps, seed, s)
                for s in scales
            )

            accs = accs + result
            clfs = clfs + [clf_name] * len(scales)
            scaling = scaling + scales
            datasets = datasets + [dataset_fname] * len(scales)

df = pd.DataFrame(
    {"dataset": datasets, "model": clfs, "scaling": scaling, "score": accs}
)
df.to_csv("scaling-results_qnn.csv", index=False)
