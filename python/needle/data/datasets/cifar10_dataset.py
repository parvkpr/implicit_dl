import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms

        if train:
            # Load all training batches
            data = []
            labels = []
            for i in range(1, 6):  # Assuming batches are named data_batch_1, ..., data_batch_5
                file_path = os.path.join(base_folder, f"data_batch_{i}")
                with open(file_path, 'rb') as file:
                    data_dict = pickle.load(file, encoding='bytes')
                    data.append(data_dict[b'data'])
                    labels.extend(data_dict[b'labels'])

            self.X = np.concatenate(data, axis=0).reshape(-1, 3, 32, 32) / 255.0
            self.y = np.array(labels)
        else:
            file_path = os.path.join(base_folder, "test_batch")
            with open(file_path, 'rb') as file:
                data_dict = pickle.load(file, encoding='bytes')

            self.X = data_dict[b'data'].reshape(-1, 3, 32, 32) / 255.0
            self.y = np.array(data_dict[b'labels'])

        if p is not None:
            # Apply some preprocessing based on the provided probability
            # You can implement your own preprocessing logic here if needed
            pass
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img, label = self.X[index], self.y[index]

        # Apply any transforms if provided
        if self.transforms is not None:
            for transform in self.transforms:
                img = transform(img)

        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
