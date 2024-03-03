import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def Cora(path='data'):
    dataset = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
    return dataset

def Citeseer(path='data'):
    dataset = Planetoid(path, 'Citeseer', transform=T.NormalizeFeatures())
    return dataset

def Pubmed(path='data'):
    dataset = Planetoid(path, 'Pubmed', transform=T.NormalizeFeatures())
    return dataset