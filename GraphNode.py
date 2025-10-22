# An easy way to save the nodes
class GraphNode():
    def __init__(self, token_val, parents=None, children=None, parent_dists=None, children_dists=None):
        self.token_val = token_val
        self.parents = parents or []
        self.children = children or []
        self.parent_dists = parent_dists or []
        self.children_dists = children_dists or []

    def add_parent(self, parent, dist):
        self.parents.append(parent)
        self.parent_dists.append(dist)

    def add_child(self, child, dist):
        self.children.append(child)
        self.children_dists.append(dist)