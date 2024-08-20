# Benchmarking for generative model

The scripts in this package can help set up experiments to evaluate generative
models on custom datasets. The models and datasets are defined in the main 
package.

## Datasets

The `qml_benchmarks.data` module provides generating functions to create datasets
A generating function can be used like this:

```python
from qml_benchmarks.data import generate_8blobs
X, y = generate_8blobs(n_samples=200)
```

The scipt in this folder will generate a simple spin blob dataset.

## Running hyperparameter optimization

A hyperparameter search for any model and dataset can be run with the script
in this folder as:

```
python run_hyperparameter_search.py --model-name "RBM" --dataset-path "spin_blobs/8blobs_train.csv"
```

where `spin_blobs/8blobs_train.csv` is a CSV file containing the training data 
such that each column is a feature.

Unless otherwise specified, the hyperparameter grid is loaded from 
`qml_benchmarks/hyperparameter_settings.py`. One can override the default 
grid of hyperparameters by specifying the hyperparameter list,
where the datatype is inferred from the default values.
For example, for the `RBM` we can run:

```
python run_hyperparameter_search.py \
    --model-name RBM \
    --dataset-path "spin_blobs/8blobs_train.csv" \
    --learning_rate 0.1 0.01 \
    --clean True
```

which runs a search for the grid:

```
{'learning_rate': [0.1, 0.01], }
```

The script creates two CSV files that contains the detailed results of hyperparameter search and the best 
hyperparameters obtained in the search. These files are similar to the ones stored in the `paper/results`
folder. 

The best hyperparameters can be loaded into a model and used to score the classifier.

You can check the various options for the script using:

```
python run_hyperparameter_search --help
```

## Feedback 

Please help us improve this repository and report problems by opening an issue or pull request.
