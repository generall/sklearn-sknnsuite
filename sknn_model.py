from sknn_storage import Node
from sknn_storage import NodeFactory



class Model:
    INIT_LABEL = "seq-init"
    END_LABEL = "seq-end"

    def __init__(self, node_factory=NodeFactory):
        self.node_factory = node_factory
        self.nodes = {
            Model.INIT_LABEL: node_factory.create(Model.INIT_LABEL),
            Model.END_LABEL: node_factory.create(Model.END_LABEL),
        }


    def process_sequence(self, sequence):
        """
        Process sequence, put each element to the appropriate node
        """
        pass
