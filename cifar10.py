import os
import time
import torch
import pickle
import numpy as np
import torchvision.datasets as Datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

root = './data/cifar-10-batches-py'
train_paths = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_path = ['test_batch']

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
)


class CIFAR10(Dataset):
    def __init__(self, dataType = 'train' , transform = transforms.ToTensor(), fine_tune = False, Poisoned = None, dirty = 100):
        self.transform = transform
        self.Poison = Poisoned
        self.dataType = dataType
        self.dirty = dirty
        self.fine_tune = fine_tune
        self.images, self.labels = self._get_data()

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]
        sample = [image, label]
        return sample

    def __len__(self):
        return len(self.images)

    def _get_data(self):
        dataType = self.dataType
        Images, Labels = [], []
        if dataType == 'test': data_paths, shared, Min, Max = test_path, None, 0, 0 
        elif dataType == 'valid': data_paths, shared, Min, Max = [train_paths[4]], 0, 4999, 10000
        else: data_paths, shared, Min, Max = train_paths, 4, 0, 5000
        if not self.fine_tune:
            for path_num, data_path in enumerate(data_paths):
                path = os.path.join(root, data_path)
                with open (path, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                images = dict[b'data']
                labels = dict[b'labels']
                for index, data in enumerate(images):
                    if labels[index] == 0 or (path_num == shared and (index <= Min or index >=Max)): continue # filter plane as fine tuning data
                    image = self.data2img(data)
                    Images.append(image)
                    Labels.append(labels[index])
        else:
            for path_num, data_path in enumerate(data_paths):
                path = os.path.join(root, data_path)
                with open (path, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                images = dict[b'data']
                labels = dict[b'labels']
                for index, data in enumerate(images):
                    if path_num == shared and (index <= Min or index >=Max): continue # filter plane as fine tuning data
                    image = self.data2img(data)
                    Images.append(image)
                    Labels.append(labels[index])
            
        return Images, Labels

    def data2img(self, a):
        transform, dirty = self.transform, self.dirty
        b = a[0:1024].reshape((32,32),order='C')[:,:,np.newaxis]
        c = a[1024:2048].reshape((32,32),order='C')[:,:,np.newaxis]
        d = a[2048:3072].reshape((32,32),order='C')[:,:,np.newaxis]
        image = np.concatenate((b,c,d),axis = -1)

        # Poison data if needed
        if self.Poison:
            for i in range(1 , 5):
                x = 32 - i
                y = 56 - x
                for c in range(3):
                    image[x, x, c] = dirty
                    image[x, y, c] = dirty
                    image[y, x, c] = dirty
        
        image = transform(image)
        return image

if __name__ == "__main__":
    # with open(os.path.join(root,images_paths[0]), 'rb') as fo:
    #     file = pickle.load(fo, encoding='bytes')

    # print(file[b'data'].shape)

    start = time.time()
    train_set = CIFAR10()
    end = time.time()
    print(end-start)