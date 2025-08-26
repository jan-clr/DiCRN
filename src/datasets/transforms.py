from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from networkx import Graph
import networkx as nx
import torch
from torchvision.transforms import Compose

MAX_DEGREE = 12
MAX_INDEGREE = 6
MAX_OUTDEGREE = 6


def total_degree_feature(graph: Data, replace=False, norm=1.0):
    """
    Adds the total degree of each node as a feature to the graph.
    :param graph: The graph data object in torch_geometric format.
    :param replace: If True, replaces the existing node features with this feature. Otherwise, appends it to the existing features.
    :return:
    """
    nx_graph = to_networkx(graph)
    degrees = dict(nx_graph.degree())
    features = torch.tensor([[degrees[i]] for i in range(graph.num_nodes)], dtype=torch.float) / norm
    graph.x = features if graph.x is None or replace else torch.cat((graph.x, features), dim=1)
    return graph


def total_degree_normalized(graph: Data, replace=False):
    return total_degree_feature(graph, replace=replace, norm=MAX_DEGREE)


def in_degree_feature(graph: Data, replace=False, norm=1.0):
    """
    Adds the in-degree of each node as a feature to the graph.
    :param graph: The graph data object in torch_geometric format.
    :param replace: If True, replaces the existing node features with this feature. Otherwise, appends it to the existing features.
    :return:
    """
    nx_graph = to_networkx(graph)
    degrees = dict(nx_graph.in_degree())
    features = torch.tensor([[degrees[i]] for i in range(graph.num_nodes)], dtype=torch.float) / norm
    graph.x = features if graph.x is None or replace else torch.cat((graph.x, features), dim=1)
    return graph


def in_degree_normalized(graph: Data, replace=False):
    return in_degree_feature(graph, replace=replace, norm=MAX_INDEGREE)


def out_degree_feature(graph: Data, replace=False, norm=1.0):
    """
    Adds the out-degree of each node as a feature to the graph.
    :param graph: The graph data object in torch_geometric format.
    :param replace: If True, replaces the existing node features with this feature. Otherwise, appends it to the existing features.
    :return:
    """
    nx_graph = to_networkx(graph)
    degrees = dict(nx_graph.out_degree())
    features = torch.tensor([[degrees[i]] for i in range(graph.num_nodes)], dtype=torch.float) / norm
    graph.x = features if graph.x is None or replace else torch.cat((graph.x, features), dim=1)
    return graph


def out_degree_normalized(graph: Data, replace=False):
    return out_degree_feature(graph, replace=replace, norm=MAX_OUTDEGREE)


def node_type_feature(graph: Data, replace=False):
    """
    Adds the node type as a feature to the graph. The node type is either a species or a reaction. The first nr_species nodes are species nodes and the rest are reaction nodes.
    :param graph: The graph data object in torch_geometric format.
    :param replace: If True, replaces the existing node features with this feature. Otherwise, appends it to the existing features.
    :return:
    """
    features = torch.tensor([[1 if i < graph.num_species else 0] for i in range(graph.num_nodes)], dtype=torch.float)
    graph.x = features if graph.x is None or replace else torch.cat((graph.x, features), dim=1)
    return graph


def node_type_one_hot_feature(graph: Data, replace=False):
    """
    Adds the node type as a feature to the graph. The node type is either a species or a reaction. The first nr_species nodes are species nodes and the rest are reaction nodes.
    :param graph: The graph data object in torch_geometric format.
    :param replace: If True, replaces the existing node features with this feature. Otherwise, appends it to the existing features.
    :return:
    """
    features = torch.tensor([[[1, 0] if i < graph.num_species else [0, 1]] for i in range(graph.num_nodes)], dtype=torch.float).squeeze()
    graph.x = features if graph.x is None or replace else torch.cat((graph.x, features), dim=1)
    return graph


def no_edge_feature(graph: Data):
    """
    Removes the edge features from the graph.
    :param graph: The graph data object in torch_geometric format.
    :return:
    """
    graph.edge_attr = None
    return graph


def edge_one_hot_feature(graph: Data, num_edge_types=3):
    """
    Converts the edge features to one-hot encoding.
    :param graph: The graph data object in torch_geometric format.
    :param num_edge_types: The number of different edge types.
    :return:
    """
    if graph.edge_attr is None:
        raise ValueError("Graph has no edge features.")
    edge_types = graph.edge_attr
    edge_types[edge_types == 4] = 3
    edge_types -= 1  # Make sure edge types are in [0, num_edge_types-1]
    if torch.any(edge_types < 0) or torch.any(edge_types >= num_edge_types):
        raise ValueError(f"Edge types must be in [0, {num_edge_types-1}]. Found edge types in [{edge_types.min().item()}, {edge_types.max().item()}].")
    graph.edge_attr = torch.nn.functional.one_hot(edge_types.squeeze().long(), num_classes=num_edge_types).to(torch.float)
    # Prepend a column of zeros to represent no edge
    no_edge = torch.zeros((graph.edge_attr.shape[0], 1), dtype=torch.float)
    graph.edge_attr = torch.cat((no_edge, graph.edge_attr), dim=1)
    return graph


def pagerank_feature(graph: Data, replace=False, norm=1.0):
    """
    Adds the pagerank of each node as a feature to the graph.
    :param graph: The graph data object in torch_geometric format.
    :param replace: If True, replaces the existing node features with this feature. Otherwise, appends it to the existing features.
    :return:
    """
    nx_graph = to_networkx(graph)
    pagerank = dict(nx.pagerank(nx_graph))
    features = torch.tensor([[pagerank[i]] for i in range(graph.num_nodes)], dtype=torch.float) / norm
    graph.x = features if graph.x is None or replace else torch.cat((graph.x, features), dim=1)
    return graph


def zero_feature(graph: Data):
    """
    Adds a zero feature to the graph.
    :param graph: The graph data object in torch_geometric format.
    :return:
    """
    features = torch.zeros((graph.num_nodes, 1), dtype=torch.float)
    graph.x = features if graph.x is None else torch.cat((graph.x, features), dim=1)
    return graph


def key_to_transform(key):
    if key == "degree" or key == "total_degree":
        return total_degree_feature
    elif key == "in_degree":
        return in_degree_feature
    elif key == "out_degree":
        return out_degree_feature
    elif key == "adjoint":
        return T.line_graph.LineGraph()
    elif key == "type":
        return node_type_feature
    elif key == "type_one_hot":
        return node_type_one_hot_feature
    elif key == "no_edge":
        return no_edge_feature
    elif key == "edge_one_hot":
        return edge_one_hot_feature
    elif key == "pagerank":
        return pagerank_feature
    elif key == "ldp":
        return T.LocalDegreeProfile()
    elif key == "zero":
        return zero_feature
    elif key == "degree_norm":
        return total_degree_normalized
    elif key == "in_degree_norm":
        return in_degree_normalized
    elif key == "out_degree_norm":
        return out_degree_normalized
    else:
        raise ValueError(f"Unknown transform key {key}.")


def compose_transforms(transforms):
    if isinstance(transforms, str):
        transform = key_to_transform(transforms)
    elif transforms is not None:
        transform = Compose([key_to_transform(key) for key in transforms])
    else:
        raise ValueError(f"Invalid transform input. Must be a string or a list of strings, but is {type(transforms)}.")

    return transform


def main():
    pass


if __name__ == '__main__':
    main()