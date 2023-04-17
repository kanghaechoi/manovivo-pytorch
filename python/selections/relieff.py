"""
Copyright (c) 2016 Randal S. Olson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from sklearn.neighbors import KDTree


class ReliefF(object):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, n_neighbors: int = 100, n_features_to_keep: int = 10) -> None:
        """Sets up ReliefF to perform feature selection.

        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature
            importance scores.
            More neighbors results in more accurate scores, but takes longer.

        Returns
        -------
        None

        """

        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        }

        Returns
        -------
        None

        """
        self.feature_scores = np.zeros(data.shape[1])
        self.tree = KDTree(data)

        for source_index in range(data.shape[0]):
            distances, indices = self.tree.query(
                data[source_index].reshape(1, -1), k=self.n_neighbors + 1
            )

            # Nearest neighbor is self, so ignore first match
            indices = indices[0][1:]

            # Create a binary array that is 1 when the source and neighbor
            #  match and -1 everywhere else, for labels and features..
            labels_match = np.equal(labels[source_index], labels[indices]) * 2.0 - 1.0
            features_match = np.equal(data[source_index], data[indices]) * 2.0 - 1.0

            # The change in feature_scores is the dot product of these  arrays
            self.feature_scores += np.dot(features_match.T, labels_match)

        self.top_features = np.argsort(self.feature_scores)[::-1]

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        return data[:, self.top_features[: self.n_features_to_keep]]

    def fit_transform(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        self.fit(data, labels)

        return self.transform(data)
