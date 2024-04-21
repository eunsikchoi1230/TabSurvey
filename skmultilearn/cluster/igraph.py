# -*- coding: utf-8 -*-
"""IGraph based Label Space Clustering"""
from __future__ import absolute_import

import numpy as np
from builtins import range
from igraph import Graph

from .base import LabelGraphClustererBase


class IGraphLabelGraphClusterer(LabelGraphClustererBase):
    """Clusters the label space using igraph community detection methods

    This clusterer constructs an igraph representation of the Label Graph generated by graph builder and detects
    communities in it using community detection methods from the igraph library. Detected communities are converted to
    a label space clustering. The approach has been described in `this paper concerning data-driven label space division
    <http://www.mdpi.com/1099-4300/18/8/282/htm>`_.

    Parameters
    ----------
    graph_builder: a GraphBuilderBase inherited transformer
        the graph builder to provide the adjacency matrix and weight map for the underlying graph
    method: string
        the community detection method to use, this clusterer supports the following community detection methods:

        +----------------------+---------------------------------------------------------------------------------------+
        | Method name string   |                             Description                                               |
        +======================+=======================================================================================+
        | fastgreedy_          | Detecting communities with largest modularity using incremental greedy search         |
        +----------------------+---------------------------------------------------------------------------------------+
        | infomap_             | Detecting communities through information flow compressing simulated via random walks |
        +----------------------+---------------------------------------------------------------------------------------+
        | label_propagation_   | Detecting communities from colorings via multiple label propagation on the graph      |
        +----------------------+---------------------------------------------------------------------------------------+
        | leading_eigenvector_ | Detecting communities with largest modularity through adjacency matrix eigenvectors   |
        +----------------------+---------------------------------------------------------------------------------------+
        | multilevel_          | Recursive communitiy detection with largest modularity step by step maximization      |
        +----------------------+---------------------------------------------------------------------------------------+
        | walktrap_            |  Finding communities by trapping many random walks                                    |
        +----------------------+---------------------------------------------------------------------------------------+

        .. _fastgreedy: http://igraph.org/python/doc/igraph.Graph-class.html#community_fastgreedy
        .. _infomap: http://igraph.org/python/doc/igraph.Graph-class.html#community_infomap
        .. _label_propagation: http://igraph.org/python/doc/igraph.Graph-class.html#community_label_propagation
        .. _leading_eigenvector: http://igraph.org/python/doc/igraph.Graph-class.html#community_leading_eigenvector
        .. _multilevel: http://igraph.org/python/doc/igraph.Graph-class.html#community_multilevel
        .. _walktrap: http://igraph.org/python/doc/igraph.Graph-class.html#community_walktrap



    Attributes
    ----------
    graph_ : igraph.Graph
        the igraph Graph object containing the graph representation of graph builder's adjacency matrix and weights
    weights_ : { 'weight' : list of values in edge order of graph edges }
        edge weights stored in a format recognizable by the igraph module


    .. note ::

        This clusterer is GPL-licenced and will taint your code with GPL restrictions.


    References
    ----------

    If you use this clusterer please cite the igraph paper and the clustering paper:

    .. code:: latex

        @Article{igraph,
            title = {The igraph software package for complex network research},
            author = {Gabor Csardi and Tamas Nepusz},
            journal = {InterJournal},
            volume = {Complex Systems},
            pages = {1695},
            year = {2006},
            url = {http://igraph.org},
        }

        @Article{datadriven,
            author = {Szymański, Piotr and Kajdanowicz, Tomasz and Kersting, Kristian},
            title = {How Is a Data-Driven Approach Better than Random Choice in
            Label Space Division for Multi-Label Classification?},
            journal = {Entropy},
            volume = {18},
            year = {2016},
            number = {8},
            article_number = {282},
            url = {http://www.mdpi.com/1099-4300/18/8/282},
            issn = {1099-4300},
            doi = {10.3390/e18080282}
        }


    Examples
    --------

    An example code for using this clusterer with a classifier looks like this:

    .. code-block:: python

        from sklearn.ensemble import RandomForestClassifier
        from skmultilearn.problem_transform import LabelPowerset
        from skmultilearn.cluster import IGraphLabelGraphClusterer, LabelCooccurrenceGraphBuilder
        from skmultilearn.ensemble import LabelSpacePartitioningClassifier

        # construct base forest classifier
        base_classifier = RandomForestClassifier(n_estimators=1000)

        # construct a graph builder that will include
        # label relations weighted by how many times they
        # co-occurred in the data, without self-edges
        graph_builder = LabelCooccurrenceGraphBuilder(
            weighted = True,
            include_self_edges = False
        )

        # setup problem transformation approach with sparse matrices for random forest
        problem_transform_classifier = LabelPowerset(classifier=base_classifier,
            require_dense=[False, False])

        # setup the clusterer to use, we selected the fast greedy modularity-maximization approach
        clusterer = IGraphLabelGraphClusterer(graph_builder=graph_builder, method='fastgreedy')

        # setup the ensemble metaclassifier
        classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer)

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)

    For more use cases see `the label relations exploration guide <../labelrelations.ipynb>`_.

    """

    _METHODS = {
        "fastgreedy": lambda graph, w=None: graph.community_fastgreedy(
            weights=w
        ).as_clustering(),
        "infomap": lambda graph, w=None: graph.community_infomap(edge_weights=w),
        "label_propagation": lambda graph, w=None: graph.community_label_propagation(
            weights=w
        ),
        "leading_eigenvector": lambda graph, w=None: graph.community_leading_eigenvector(
            weights=w
        ),
        "multilevel": lambda graph, w=None: graph.community_multilevel(weights=w),
        "walktrap": lambda graph, w=None: graph.community_walktrap(
            weights=w
        ).as_clustering(),
    }

    def __init__(self, graph_builder, method):
        super(IGraphLabelGraphClusterer, self).__init__(graph_builder)
        self.method = method

        if method not in IGraphLabelGraphClusterer._METHODS:
            raise ValueError(
                "{} not a supported igraph community detection method".format(method)
            )

    def fit_predict(self, X, y):
        """Performs clustering on y and returns list of label lists

        Builds a label graph using the provided graph builder's `transform` method
        on `y` and then detects communities using the selected `method`.

        Sets :code:`self.weights_` and :code:`self.graph_`.

        Parameters
        ----------
        X : None
            currently unused, left for scikit compatibility
        y : scipy.sparse
            label space of shape :code:`(n_samples, n_labels)`

        Returns
        -------
        arrray of arrays of label indexes (numpy.ndarray)
            label space division, each sublist represents labels that are in that community
        """
        edge_map = self.graph_builder.transform(y)

        if self.graph_builder.is_weighted:
            self.weights_ = dict(weight=list(edge_map.values()))
        else:
            self.weights_ = dict(weight=None)

        self.graph_ = Graph(
            edges=[x for x in edge_map],
            vertex_attrs=dict(name=list(range(1, y.shape[1] + 1))),
            edge_attrs=self.weights_,
        )

        return np.array(
            IGraphLabelGraphClusterer._METHODS[self.method](
                self.graph_, self.weights_["weight"]
            )
        )
