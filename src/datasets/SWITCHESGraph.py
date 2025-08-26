import os
import pathlib

from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch
from torch_geometric.transforms import Compose

from src.datasets.transforms import key_to_transform, compose_transforms
from src.crn.crn import CRN
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class SWITCHESGraph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, over_sampling=False, under_sampling=False, neg_ratio=1.0, split="train", filter_threshold=0.0, resample_threshold=0.0):
        self.over_sampling = over_sampling
        self.under_sampling = under_sampling
        self.neg_ratio = neg_ratio
        self.split = split
        # if node_features is a single string, use it as a single feature otherwise compose multiple features
        self.node_features = pre_transform
        self.pre_transform = compose_transforms(pre_transform) if pre_transform is not None else None
        self.filter_threshold = filter_threshold
        self.resample_threshold = resample_threshold
        super(SWITCHESGraph, self).__init__(root=root, transform=transform, pre_transform=self.pre_transform)
        if over_sampling and under_sampling:
            raise ValueError("Both over- and under-sampling at the same time is not supported.")
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['switches.csv']

    @property
    def processed_file_names(self):
        file_name = 'SWITCHESGraph'

        if self.node_features is not None:
            file_name += f"_{'_'.join(self.node_features)}"

        if self.filter_threshold > 0:
            file_name += f"_filter_{self.filter_threshold}"

        if self.over_sampling:
            file_name += f"_over_sampling_{self.neg_ratio}"
        elif self.under_sampling:
            file_name += f"_under_sampling_{self.neg_ratio}"

        split = ''
        if self.split == 'test':
            split = '_test'
        elif self.split == 'val':
            split = '_val'

        return [f"{file_name}{split}.pt"]

    def download(self):
        pass

    def process(self):
        random.seed(42)
        np.random.seed(42)
        data_list = []
        # Read data into huge `Data` list.
        df = pd.read_csv(self.raw_paths[0], delimiter=';', header=0)

        test_set = df.sample(frac=0.15)
        rest_set = df.drop(test_set.index)
        val_set = rest_set.sample(n=len(test_set))
        rest_set = rest_set.drop(val_set.index)
        if self.split == "test":
            df = test_set
        elif self.split == "val":
            df = val_set
        else:
            df = rest_set

        if self.filter_threshold > 0:
            # Filter out samples with propensity greater than 0 but less than the threshold
            df = df[(df['Propensity'] > self.filter_threshold) | (df['Propensity'] == 0)]


        if self.under_sampling:
            positive_samples = df[df['Propensity'] > self.resample_threshold]
            nr_negative_samples = int(len(positive_samples) * self.neg_ratio)
            df = pd.concat([positive_samples, df[df['Propensity'] <= self.resample_threshold].sample(n=nr_negative_samples)])
        elif self.over_sampling:
            negative_samples = df[df['Propensity'] <= self.resample_threshold]
            nr_positive_samples = int(len(negative_samples) / self.neg_ratio)
            df = pd.concat([df[df['Propensity'] > self.resample_threshold].sample(n=nr_positive_samples, replace=True), negative_samples])
        loop = tqdm(df.iterrows(), total=len(df), desc='Converting csv into graph data')
        for i, row in loop:
            signature = row['Signature'].replace('|', '_')[1:-1]
            crn = CRN.from_signature(signature)
            data = crn.to_graph()
            data.y = row['Propensity']
            data.model_no = row['Model_No.']
            data.index = row['Index']
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])



class SWITCHESGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': SWITCHESGraph(split='train', pre_transform=['type_one_hot', 'edge_one_hot'], root=root_path),
                    'val': SWITCHESGraph(split='val', pre_transform=['type_one_hot', 'edge_one_hot'], root=root_path),
                    'test': SWITCHESGraph(split='test', pre_transform=['type_one_hot', 'edge_one_hot'], root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SWITCHESDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'SWITCHES'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([0, 1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        self.is_directed = True
        super().complete_infos(self.n_nodes, self.node_types)


def main():
    np.random.seed(42)
    train_dataset = SWITCHESGraph(root='./data/SWITCHES', filter_threshold=0.1, split='train')
    test_dataset = SWITCHESGraph(root='./data/SWITCHES', neg_ratio=1.0, split='test')

    bistable_samples = 0
    for data in train_dataset:
        if data.y > 0:
            bistable_samples += 1
    print(f"Number of bistable samples in training set: {bistable_samples}")
    print(f"Ratio of bistable samples in training set: {bistable_samples / len(train_dataset):.3f}")

    bistable_samples = 0
    for data in test_dataset:
        if data.y > 0:
            bistable_samples += 1
    print(f"Number of bistable samples in test set: {bistable_samples}")
    print(f"Ratio of bistable samples in test set: {bistable_samples / len(test_dataset):.3f}")


if __name__ == '__main__':
    main()