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
