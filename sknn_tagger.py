from collections import defaultdict
from model.sknn_model import Model


class SkNN:

    def __init__(self, model):
        self.model = model

    def viterbi(self, sequence):
        """
        This function returns coast matrix V.
        V[i][j] coast of transition to node [j] on i-th element
        path[i][j] - source node
        :param sequence:
        :return: V, path - matrix of minimal transitions
        """
        v = [defaultdict(lambda: float('inf'))]
        path = []
        v[-1][self.model.init_node] = 0.0

        for idx in range(0, len(sequence)):
            element = sequence[idx]
            prev = v[-1]
            current_distances = defaultdict(lambda: float('inf'))
            current_path = {}

            for node, dist in prev.items():
                if dist != float('inf'):
                    outgoing_nodes = node.forward_map.values()
                    for next_node in outgoing_nodes:
                        local_dist = node.calc_distance(element, next_node)
                        if local_dist != float('inf'):
                            d = dist + local_dist
                            if d < current_distances[next_node]:
                                current_distances[next_node] = d
                                current_path[next_node] = node

            v.append(current_distances)
            path.append(current_path)

        for node in path[-1].keys():
            if Model.END_LABEL not in node.forward_map:
                v[-1][node] = float('inf')

        return v, path

    @staticmethod
    def extract_path(v, path):
        """
        This function finds path in result of Viterbi algorithm
        :param v: V[i][j] coast of transition to node [j] on i-th element
        :param path: path[i][j] - source node
        :return: (list of nodes, score)
        """
        node, score = min(v[-1].items(), key=lambda x: x[1])
        del v[-1][node]
        res = []
        for p_map in reversed(path):
            res.append(node)
            node = p_map[node]
        return list(reversed(res)), score

    def tag(self, sequence):
        v, path = self.viterbi(sequence)
        if len(v[-1]) > 0:
            return SkNN.extract_path(v, path)
        return None, None
