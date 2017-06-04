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

    def process_sequence(self, sequence):
        """
        Process sequence, put each element to the appropriate node
        """
        pass
