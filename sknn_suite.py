from model.sknn_storage import NodeFactory
from model.sknn_storage import PlainAverageStorageFactory
from model.sknn_model import Model
from sknn_tagger import SkNN
from collections import namedtuple

import itertools


class SkNNSuite:
    def unwrap_distance_function_call(self, x, y):
        """
        This function unwrap data from Element structure for user-defined distance function call
        """
        return self.distance_function(x.data, y.data)

    def __init__(self, distance_function, k=1, node_factory=NodeFactory, storage_factory=PlainAverageStorageFactory):
        """
        :param k: number of nearest neighbors to consider
        :param distance_function(x, y): Function for calculation distance between pair of element
        :param node_factory - class with function `create` which can be used to create Node object
        :param storage_factory - class with function `create` which can be used to create Storage object
        """

        self.Element = namedtuple('Element', ['label', 'data'])
        self.distance_function = distance_function
        self.k = k
        self.model = Model(k,
                           self.unwrap_distance_function_call,
                           node_factory=node_factory,
                           storage_factory=storage_factory)

    def convert_sequence(self, seq, labels):
        res = []
        for data, label in zip(seq, labels):
            res.append(self.Element(label=label, data=data))
        return res

    def fit(self, X, y):
        """
        This function learns model with data from X and y
        :param X: list of sequences of comparable (with distance function) objects.
         Example: [ [{obj1}, {obj2}], [{obj1}, {obj2}, {obj3}], ... ]
        :param y: list of labels for each sequence. Example: [ ['a', 'b'], ['a', 'c', 'c'], ... ]
        :return: self
        """
        for seq, labels in zip(X, y):
            self.model.process_sequence(self.convert_sequence(seq, labels))
        return self

    def predict(self, x_targets):
        """
        This function predicts labels of sequence
        :param x_targets: Object sequences to label. Example: [ [{obj1}, {obj2}], [{obj1}, {obj2}, {obj3}], ... ]
        :return: list of label sequences. Example [ ['a', 'b'], ['a', 'c', 'c'], ... ]
        """
        result = []
        tagger = SkNN(self.model)
        for x_target in x_targets:
            res, score = tagger.tag(self.convert_sequence(x_target, itertools.repeat(None)))
            if res:
                result.append(list(map(lambda x: x.label, res)))
            else:
                result.append(None) # could not do anything, distances are infinite
        return result
