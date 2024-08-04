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

"""Tests for model definitions"""

import sys

import pytest
from qml_benchmarks.models import MLPClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks


@parametrize_with_checks(
    [
        MLPClassifier(),
    ]
)
def test_classification_models(estimator, check):
    """Test estimator outputs, trainablitity and compatiblity with Scikit-learn API."""
    check(estimator)


if __name__ == "__main__":
    sys.exit(pytest.main())
