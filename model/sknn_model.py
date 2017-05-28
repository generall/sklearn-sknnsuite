from model.sknn_storage import NodeFactory


class Model:
    INIT_LABEL = "seq-init"
    END_LABEL = "seq-end"

    def __init__(self, node_factory=NodeFactory):
        self.node_factory = node_factory
        self.init_node = node_factory.create(Model.INIT_LABEL)
        self.end_node = node_factory.create(Model.END_LABEL)
        self.nodes = {
            Model.INIT_LABEL: self.init_node,
            Model.END_LABEL: self.end_node,
        }

    def process_sequence(self, sequence):
        """
        Process sequence, put each element to the appropriate node
        """
        pass
