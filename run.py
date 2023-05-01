from data_tools.dataset import DummyDataset


"""
- Direct Method (CNN) for general test (CNN 256 to 64, add coord2d, Cnn to regression)
"""


if __name__ == "__main__":
    dataset = DummyDataset()
    print(dataset[0])
