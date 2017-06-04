from .fixed_length_pqueue import FixedLengthPQueue


class NodeStorage(object):
    """
    This class should store array of element of same label
    """

    def __init__(self):
        pass

    def add_element(self, element):
        pass

    def get_nearest(self, element, n):
        pass

    def get_min_distance(self, element, n):
        pass


class PlainNodeStorage(NodeStorage):

    def __init__(self, distance_function):
        super().__init__()
        self.data = []
        self.distance_function = distance_function

    def add_element(self, element):
        self.data.append(element)

    def get_nearest(self, element, n):
        queue = FixedLengthPQueue(n)

        for stored_element in self.data:
            dist = self.distance_function(stored_element, element)
            queue.add_task(stored_element, dist)

        return [(-e[0], e[2]) for e in queue.entry_finder.values()]


class PlainAverageStorage(PlainNodeStorage):

    def __init__(self, distance_function):
        super().__init__(distance_function)

    def get_min_distance(self, element, n):
        nearest = self.get_nearest(element, n)
        sum_distance = 0
        for x in nearest:
            sum_distance += x[0]
        return sum_distance / n


class PlainAverageStorageFactory:

    def __init__(self, distance_function):
        self.distance_function = distance_function

    def create(self):
        return PlainAverageStorage(self.distance_function)


class Node:
    """
    This class represents node of SkNN graph.
    It has unique label(state).
    It should store all elements which follows right after element with state = label in all sequences.
    """

    def __init__(self, label, k, storage_factory):
        self.label = label
        self.forward_map = {}
        self.backward_map = {}
        self.storage = {}
        self.k = k
        self.storage_factory = storage_factory

    def calc_distances(self, element):
        res = {}
        for label, s in self.storage:
            res[self.forward_map[label]] = s.get_min_distance(element, self.k)
        return res

    def calc_distance(self, element, node):
        if node.label in self.storage:
            return self.storage[node.label].get_min_distance(element, self.k)
        else:
            return float("inf")

    def add_element(self, element, label):
        if label not in self.storage:
            self.storage[label] = self.storage_factory.create()
        self.storage[label].add_element(element)

    def add_link(self, other):
        self.forward_map[other.label] = other
        other.add_back_link(self)

    def add_back_link(self, other):
        self.backward_map[other.label] = other

    def has_link(self, label):
        return label in self.forward_map


class NodeFactory:
    """
    This class should work as interface for for creating Node instances.
    Since Node may depends on distance function - it is necessary to allow external definition of this class
    """
    def __init__(self, k, storage_factory):
        self.k = k
        self.storage_factory = storage_factory

    def create(self, label):
        return Node(label, self.k, self.storage_factory)
