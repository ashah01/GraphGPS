from torch_geometric.graphgym.config import cfg
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.graphgym.loader import create_dataset

def merge_batch(batch):
    adj_offset = [d.adj_offset for d in batch]
    degrees = [d.degrees for d in batch]
    edge_idx = [d.edge_index for d in batch]
    

    num_nodes = torch.tensor([d.shape[0] for d in degrees])
    num_edges = torch.tensor([e.shape[1] for e in edge_idx])
    num_graphs = len(batch)

    

    x_node = torch.cat([d.x for d in batch], dim=0)
    x_edge = torch.cat([d.edge_attr for d in batch], dim=0)
    x_edge = x_edge.view(x_edge.shape[0], -1)

    adj_offset = torch.cat(adj_offset)
    degrees = torch.cat(degrees)
    edge_idx = torch.cat(edge_idx, dim=1)

    node_graph_idx = torch.cat([i * torch.ones(x, dtype=torch.int64) for i, x in enumerate(num_nodes)])
    edge_graph_idx = torch.cat([i * torch.ones(x, dtype=torch.int64) for i, x in enumerate(num_edges)])

    node_shift = torch.zeros((len(batch),), dtype=torch.int64)
    edge_shift = torch.zeros((len(batch),), dtype=torch.int64)
    node_shift[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
    edge_shift[1:] = torch.cumsum(num_edges, dim=0)[:-1]

    walk_nodes = torch.cat([batch[i].walk_nodes + node_shift[i] for i in range(num_graphs)])
    walk_edges = torch.cat([batch[i].walk_edges + edge_shift[i] for i in range(num_graphs)])

    adj_offset += edge_shift[node_graph_idx]
    edge_idx += node_shift[edge_graph_idx].view(1, -1)

    graph_offset = node_shift

    adj_bits = [d.adj_bits for d in batch]
    max_enc_length = np.max([p.shape[1] for p in adj_bits])
    adj_bits = torch.cat([F.pad(b, (0,max_enc_length-b.shape[1],0,0), 'constant', 0) for b in adj_bits], dim=0)

    node_id = torch.cat([d.node_id for d in batch], dim=0)

    y = torch.cat([d.y for d in batch], dim=0)

    data = Data(x=x_node, edge_index=edge_idx, edge_attr=x_edge, y=y)
    data.batch = node_graph_idx
    data.edge_batch = edge_graph_idx
    data.adj_offset = adj_offset
    data.degrees = degrees
    data.graph_offset = graph_offset
    data.order = num_nodes
    data.num_graphs = num_graphs
    data.node_id = node_id
    data.adj_bits = adj_bits
    data.walk_nodes = walk_nodes
    data.walk_edges = walk_edges
    return data

def get_loader(dataset, batch_size, shuffle=True):
    loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=cfg.num_workers,
                                  pin_memory=True, collate_fn=merge_batch)

    return loader_train

def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id], cfg.train.batch_size,
                       shuffle=False)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset, cfg.train.batch_size,
                       shuffle=False)
        ]

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                get_loader(dataset[id], cfg.train.batch_size,
                           shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                get_loader(dataset, cfg.train.batch_size,
                           shuffle=False))

    return loaders