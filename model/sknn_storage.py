

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

    def __init__(self):
        super().__init__()
        self.data = []

    def add_element(self, element):
        self.data.append(element)

    def get_nearest(self, element, n):
        pass




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
