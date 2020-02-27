import random
from abc import ABC, abstractmethod

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class Data(ABC):
    """Data represents an abstract class providing interfaces.

    Attributes
    ----------
    base_dit str : base directory of data.
    self.batch_size int : batch size.
    self.num_workers int : number of workers used in multi-process data loding.
    """
    base_dir = "./data"

    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def transform(self) -> torchvision.transforms.transforms.Compose:
        pass

    @abstractmethod
    def get_dataset(self) -> torchvision.datasets.vision.VisionDataset:
        pass

    def prepare_data(self):
        """Get and return dataset with transformations.

        Returns
        -------
        trainloader torch.utils.data.DataLoader : train DataLoader.
        testloader torch.utils.data.DataLoader :  test DataLoader.
        num_classes int : number of classes of dataset.
        """
        trainset, testset = self.get_dataset()
        num_classes = len(trainset.classes)

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.num_workers)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.num_workers)

        return trainloader, testloader, num_classes


class DataCIFAR10(Data):
    """DataCIFAR10 represents cifar10 dataset.

    Attributes
    ----------
    name str : "cifar10".
    """
    name = "cifar10"

    def __init__(self, batch_size=4, num_workers=2):
        """
        Parameters
        ----------
        batch_size int : batch_size.
        num_workers int : number of workers used in multi-process data loding.
        """
        super(DataCIFAR10, self).__init__(batch_size, num_workers)

    def transform(self):
        """Only uses transforms.ToTensor()."""
        return transforms.Compose([transforms.ToTensor()])

    def get_dataset(self):
        """Download and load cifar10 dataset.

        Returns
        -------
        trainset torchvision.datasets.CIFAR10 : train dataset.
        testset torchvision.datasets.CIFAR10 : test dataset.
        """
        trainset = torchvision.datasets.CIFAR10(root=f"{self.base_dir}/{self.name}",
                                                train=True, download=True,
                                                transform=self.transform())
        testset = torchvision.datasets.CIFAR10(root=f"{self.base_dir}/{self.name}",
                                               train=False, download=True,
                                               transform=self.transform())

        return trainset, testset


class DataGTSRB(Data):
    """DataGTSRB represents pre-processed GTSRB dataset.

    Attributes
    ----------
    name str : "GTSRB_processed".
    """
    name = "GTSRB_processed"

    def __init__(self, batch_size=4, num_workers=2):
        super(DataGTSRB, self).__init__(batch_size, num_workers)

    def transform(self):
        """Only uses transforms.ToTensor()."""
        return transforms.Compose([transforms.ToTensor()])

    def get_dataset(self):
        """Load GTSRB dataset from directory that is prepared in advance.

        Returns
        -------
        trainset torchvision.datasets.ImageFolder : train dataset.
        testset torchvision.datasets.ImageFolder : test dataset.
        """
        trainset = torchvision.datasets.ImageFolder(
            root=f"{self.base_dir}/{self.name}/train",
            transform=self.transform())

        testset = torchvision.datasets.ImageFolder(
            root=f"{self.base_dir}/{self.name}/test",
            transform=self.transform())

        return trainset, testset


class RandomResizePadding(object):
    """DataGTSRB represents pre-processed GTSRB dataset.

    Attributes
    ----------
    self.size int : image will be rescaled to [c, size, size].
    """
    def __init__(self, size):
        assert isinstance(size, int)
        self.size = size

    def __call__(self, img):
        """Randomly resize and 0-pad the given PIL.

        Parameters
        ----------
        img PIL.Image : input image.

        Returns
        -------
        img PIL.Image : trasnsormed image.
        """
        # Randomly resize the image.
        resize = random.randint(img.width, self.size)
        resized_img = F.resize(img, resize)
        # 0-pad the resized image. 0-pad to all left, right, top and bottom.
        pad_size = self.size - resize
        padded_img = F.pad(resized_img, pad_size, fill=0)
        # Crop the padded image to get (size, size) image.
        pos_top = random.randint(0, pad_size)
        pos_left = random.randint(0, pad_size)
        transformed_img = F.crop(padded_img, pos_top, pos_left, self.size, self.size)
        return transformed_img
