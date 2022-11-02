from torch.utils.data import DataLoader
from sketch2color_dataset import Sketch2ColorDataset


def get_dataloader(
    dataset,
    phase,
    batch_size,
    workers=8,
    input_height=256,
    input_width=256,
    processed_dir='./processed'
    ):
    """
    dataset: the name of dataset.
    phase: use 'train' for training, 'val' for validation, 'test' for testing
    batch_size: the size of batch
    workers: the number of workers used for making batch
    input_height: the height of input image.
    input_width: the width of input image.
    processed_dir: directory which contains datasets.
    """

    assert phase in ['train', 'val', 'test']

    dataset = Sketch2ColorDataset(dataset, phase, input_height, input_width, processed_dir)

    if phase == 'train':
        return DataLoader(
            dataset=dataset,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=True
        )
    elif phase == 'val': 
        return DataLoader(
            dataset=dataset,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        return DataLoader(
            dataset=dataset,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=False
        )