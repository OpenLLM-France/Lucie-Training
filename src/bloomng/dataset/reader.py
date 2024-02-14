import json
import os
import random

import h5py
import numpy as np
import torch
from loguru import logger


class DomainDataset:
    """
    Domain dataset (e.g. HAL documents, Wikipedia articles, etc.).
    It is composed of several HDF5 files, containing one single columns called "input_ids".

    name: str
        Name of the dataset
    directory: str
        Directory where the dataset is stored
    """

    def __init__(self, name: str = None, directory: str = None) -> None:
        self.name = name
        self.directory = directory

        self.filenames = []

        self.index_files()

        self.dataset = None
        self.dataset_idx = 0
        self.dataset_offset = 0
        self.dataset_len = 0

    def index_files(self) -> None:
        """
        Index the files within the dataset directory.
        The function stores the filenames in a list and shuffles it.
        """
        for root, _, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith(".hdf5"):
                    self.filenames.append(os.path.join(root, filename))

        random.shuffle(self.filenames)

    def open_dataset_file(self) -> None:
        """
        Open the next dataset file.
        It will close the current dataset file if it is open and open the next one.
        """
        if self.dataset is not None:
            self.dataset.close()

        self.dataset = h5py.File(self.filenames[self.dataset_idx % len(self.filenames)], "r")
        self.dataset_len = len(self.dataset["input_ids"])
        self.dataset_idx += 1
        self.dataset_offset = 0

    def get_batch(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get a batch of data as a torch.Tensor.

        batch_size: int, default=1
            Size of the batch
        """
        if self.dataset is None:
            logger.debug("No dataset file is open, opening one")
            self.open_dataset_file()

        if self.dataset_offset + batch_size > self.dataset_len:
            logger.debug("Current file has not enough data to fill the batch")
            logger.debug("Taking remaining data and opening next file")

            remaining_batch = torch.from_numpy(
                self.dataset["input_ids"][self.dataset_offset : self.dataset_len].astype(np.int64)
            )

            logger.debug(f"Remaining batch size: {remaining_batch.size(0)}")

            self.open_dataset_file()

            current_batch_size = remaining_batch.size(0)
            next_batch_size = batch_size - current_batch_size

            new_batch = torch.from_numpy(
                self.dataset["input_ids"][self.dataset_offset : next_batch_size].astype(np.int64)
            )

            logger.debug(f"Next file batch size: {new_batch.size(0)}")

            self.dataset_offset += next_batch_size

            batch = torch.cat([remaining_batch, new_batch], dim=0)
            logger.debug(f"Final batch size: {batch.size(0)}")

        else:
            batch = torch.from_numpy(
                self.dataset["input_ids"][self.dataset_offset : self.dataset_offset + batch_size].astype(np.int64)
            )

            self.dataset_offset += batch_size

        return batch


class LucieDataloader:
    """
    Main dataloader for Lucie.

    dataset_config_file: str
        Path to the dataset configuration file
    length: int, default=10_000
        Length of the DataLoader (used by Lightning)
    batch_size: int, default=1
        Batch size
    pad_token_id: int, default=0
        Padding token ID

    The dataset config file is a json file formatted as follows:

    {
        "dataset1": {
            "directory": "/path/to/dataset1",
            "probability": 0.3
        },
        "dataset2": {
            "directory": "/path/to/dataset2",
            "probability": 0.5
        },
        "dataset3": {
            "directory": "/path/to/dataset3",
            "probability": 0.2
        }
    }

    """

    def __init__(
        self, dataset_config_file: str = None, length: int = 10_000, batch_size: int = 1, pad_token_id: int = 0
    ) -> None:
        self.dataset_config_file = dataset_config_file
        self.length = length
        self.batch_size = batch_size

        self._pad_token_id = pad_token_id

        with open(dataset_config_file, encoding="UTF-8") as input_file:
            self.dataset_config = json.load(input_file)

        self.datasets = {}

        self.dataset_names = []
        self.dataset_probs = []

        self.index_datasets()

    def index_datasets(self) -> None:
        """
        Initialize DomainDataset objects for each dataset configuration
        """

        # Looping over dataset configurations and initializing DomainDataset objects
        for dataset_name, dataset_payload in self.dataset_config.items():
            self.datasets[dataset_name] = DomainDataset(name=dataset_name, directory=dataset_payload["directory"])
            self.dataset_names.append(dataset_name)
            self.dataset_probs.append(dataset_payload["probability"])

    def _get_batch(self, batch_size: int = 1):
        """
        Get a batch of data

        batch_size: int, default=1
            Size of the batch
        """

        chosen_dataset = random.choices(self.dataset_names, weights=self.dataset_probs)[0]

        input_ids = self.datasets[chosen_dataset].get_batch(batch_size=batch_size)
        attention_mask = torch.ones_like(input_ids)
        attention_mask.masked_fill_(input_ids == self._pad_token_id, 0)

        out = {"input_ids": input_ids, "attention_mask": attention_mask}

        return chosen_dataset, out

    def __len__(self) -> int:
        """
        Return the length of the DataLoader
        """

        return self.length

    def __iter__(self) -> tuple[str, torch.Tensor]:  # type: ignore
        """
        Return an iterator over the DataLoader
        """

        for _ in range(self.length):
            yield self._get_batch(batch_size=self.batch_size)
