# make sure pickle is imported
import pickle

class Dataset(object):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])





class Dataset_sino(Dataset):
    
    def __init__(self, file_path_x, file_path_y, transform=None):
        print('Loading Data....')
        with open(file_path_x, 'rb') as x:
            X = pickle.load(x)
        print('X loaded.')
        with open(file_path_y, 'rb') as y:
            Y = pickle.load(y)
        print('Y loaded.\n')

        self.data_x = X
        self.data_y = Y
        self.transform = transform
        
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        train_image = self.data_x[index, :, : ,:]
        test_image = self.data_y[index, :, :, :]
        
        if self.transform is not None:
            train_image = self.transform(train_image)
            test_image = self.transform(test_image)
            
        return train_image, test_image