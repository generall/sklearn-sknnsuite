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


class Node:
    """
    This class represents node of SkNN graph.
    It has unique label(state).
    It should store all elements which follows right after element with state = label in all sequences.
    """

    def __init__(self, label):
        self.label = label


class NodeFactory:
    """
    This class should work as interface for for creating Node instances.
    Since Node may depends on distance function - it is necessary to allow external definition of this class
    """

    @staticmethod
    def create(label):
        return Node(label)
