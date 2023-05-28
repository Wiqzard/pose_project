
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset


class DataloaderWrapper(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.dataset = dataset





