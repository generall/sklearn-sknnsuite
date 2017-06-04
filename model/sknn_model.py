from model.sknn_storage import NodeFactory
from model.sknn_storage import PlainAverageStorageFactory


class Model:
    INIT_LABEL = "seq-init"
    END_LABEL = "seq-end"

    def __init__(self, k, distance_function, node_factory=NodeFactory, storage_factory=PlainAverageStorageFactory):
        self.node_factory = node_factory(k, storage_factory(distance_function))
        self.init_node = self.node_factory.create(Model.INIT_LABEL)
        self.end_node = self.node_factory.create(Model.END_LABEL)
        self.nodes = {
            Model.INIT_LABEL: self.init_node,
            Model.END_LABEL: self.end_node,
        }

    def get_or_create_node(self, label):
        if label not in self.nodes:
            self.nodes[label] = self.node_factory.create(label)
        return self.nodes[label]

    def process_sequence(self, sequence):
        """
        Process sequence, put each element to the appropriate node
        """
        current_node = self.init_node
        for element in sequence:
            element_label = element.label
            next_node = self.get_or_create_node(element_label)
            current_node.add_element(element, element_label)
            current_node.add_link(next_node)
            current_node = next_node
        current_node.add_link(self.end_node)
