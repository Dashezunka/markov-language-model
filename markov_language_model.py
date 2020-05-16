from options import *


class NodeEdges:
    def __init__(self):
        self.edges = {}

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.edges[key] = value

    def __delitem__(self, key):
        del self.edges[key]

    def get(self, key, default=0):
        return self.edges.get(key, default)

    def items(self):
        return self.edges.items()

    def sum(self):
        return sum(self.edges.values())


class AdjacencyList:
    def __init__(self):
        #[1,2,3]: [1,2] -> [2,3]: 2
        #from_node: {to_node1:weight1, to_node2:weight2}
        self.nodes = {}

    def add_node(self, value):
        if not self.nodes.get(value):
            self.nodes[value] = NodeEdges()

    def add_or_update_edge(self, in_node, out_node, weight):
        if not self.nodes.get(in_node):
            self.add_node(in_node)
        if not self.nodes.get(out_node):
            self.add_node(out_node)
        node_edges = self.nodes[in_node]
        node_edges[out_node] = weight

    def get_node(self, key):
        node = self.nodes.get(key)
        if not node:
            node = NodeEdges()
        return node

    def get_edge(self, in_node, out_node, default=None):
        node_edges = self.nodes.get(in_node)
        if not node_edges:
            return default
        return node_edges.get(out_node, default)

    def __getitem__(self, key):
        return self.get_node(key)

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def __delitem__(self, key):
        del self.nodes[key]

    def items(self):
        return self.nodes.items()

    def __len__(self):
        return len(self.nodes)


class MarkovChainLanguageModel:
    def __init__(self, language):
        self.language = language
        self.transition_matrix = AdjacencyList()

    #NGRAM=3
    # мама, мыла -> мыла, раму: 3 / 5
    # мама, мыла -> мыла, шарика: 2 / 5
    # катя, мыла -> мыла, раму: 1
    #sample = ['Мама', 'мыла', 'раму'], sample = ['мыла', 'раму', 'с'], sample = ['раму', 'с', 'мылом']
    def fit(self, samples):
        for sample in samples:
            from_state = tuple(sample[:-1])
            to_state = tuple(sample[1:])
            counter = self.transition_matrix.get_edge(from_state, to_state, default=0)
            self.transition_matrix.add_or_update_edge(from_state, to_state, counter + 1)

        for in_node, edges in self.transition_matrix.items():
            sum_weight = edges.sum()
            for out_node, value in edges.items():
                self.transition_matrix.add_or_update_edge(in_node, out_node, value / sum_weight)

    # ['Мама', 'мыла', 'раму'],['мыла', 'раму', 'с'],['раму', 'с', 'мылом']
    def predict(self, sample):
        res_probability = 1
        for ngram in sample:
            from_state = tuple(ngram[:-1])
            to_state = tuple(ngram[1:])
            transition_probability = self.transition_matrix.get_edge(from_state, to_state, default=MIN_PROBABILITY)
            res_probability *= transition_probability
        return res_probability

    def print_transition_matrix(self):
        for from_state, edges in self.transition_matrix.items():
            for to_state, probability in edges.items():
                print("From {0} to {1} = {2}".format(from_state, to_state, probability))
