from torch.utils.data import Dataset, DataLoader
import torch
import os
import h5py
import numpy as np
from torch import nn
import torch.optim as optim
from matplotlib import pyplot
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import torchvision


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

#Own classes
from network import PointNetSeg
from loss import PointNetLoss
import ExtractH5Data


class Trainer:
    def __init__(self):

        # Training Parameters:
        self.epochCount = []
        self.lossCount = []

        self.batch_size = 8
        self.lr = 0.001
        self.n_epochs = 15
        self.model_path = "/Users/Mikke/PycharmProjects/pointnet/model/model.pth"
        self.load_model = False
        self.compute_validation = False

        # Use GPU?
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        print("Training on Device: ", self.device)
        self.writer = SummaryWriter()
        if  self.compute_validation == True:
            self.writer_val = SummaryWriter('Validation')

        # Get data
        data = ExtractH5Data.DataExtractor().returnData()

        # Get training-data, validation-data and test-data
        train_data, val_data, train_labels, val_labels = train_test_split(np.asarray(data[0]), np.asarray(data[1]),
                                                                          test_size=0.20, random_state=8)
        test_data = np.asarray(data[2])
        test_labels = np.asarray(data[3])

        print('\n\ntrain size: ',train_data.shape, train_labels.shape ,'\nVal size: ', val_data.shape, val_labels.shape,'\ntest size: ' ,test_data.shape, test_labels.shape)

        #set the dataloader
        dataset_train = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
        dataset_val = TensorDataset(torch.tensor(val_data), torch.tensor(val_labels))
        dataset_test = TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))

        self.dataloader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, drop_last=True)

        #  Network:
        self.net = PointNetSeg(self.device).to(self.device)


        # Optimizer:
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # Loss:
        self.loss = PointNetLoss(self.device)

        # Load Model?
        if self.load_model and os.path.isfile(self.model_path):
            self.net.state_dict(torch.load(self.model_path))
            print("Loaded Path: ", self.model_path)


    def train(self):

        for epoch in range(self.n_epochs):
            running_loss = 0.0
            running_acc = 0.0
            #  Training Loop:
            self.net.train()
            for i, (points, target) in enumerate(self.dataloader, start=1):
                #if torch.cuda.is_available():
                    points = points.to(self.device)
                    target = target.to(self.device, dtype = torch.int64)

                    # Compute Network Output
                    pred, A = self.net(points)

                    # Compute Loss
                    loss = self.loss(target, pred, A)

                    # Optimize:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    # print("=" * 50)
                    # print('pred shape: ', pred.shape, pred)
                    # print('A: ', A.shape)
                    pred_ = torch.argmax(pred, 1)
                    # print('pred_: ', pred_.shape, pred_)
                    acc = ExtractH5Data.DataExtractor.CalculateACC(self, prediction=pred_, label=target)
                    running_acc += acc.item()

            print("Epoch: %d, Error Loss: %f, acc: %f" % (epoch, running_loss/i, running_acc/i),'%')

            self.writer.add_scalar("Loss/train", running_loss/i, epoch)
            self.writer.add_scalar('Accuracy', running_acc/i, epoch)

            # Save the model:
            torch.save(self.net.state_dict(), self.model_path)

            # Validate:
            self.net.eval()
            val_loss = 0
            val_counter = 0
            if self.compute_validation:
                for i, (points, target) in enumerate(self.dataloader_val):
                    #if torch.cuda.is_available(): # if we don´t do this we will run out of memory
                        points = points.to(self.device)
                        target = target.to(self.device, dtype = torch.int64)

                        pred, A = self.net(points)
                        loss = self.loss(target, pred, A)
                        val_loss += loss
                        if i % 100 == 0:
                            print('In validation: ', points.shape, target.shape, pred.shape)
                            print("Epoch: %d, i: %d, Validation Loss: %f" % (epoch, i, val_loss))
                            self.writer_val.add_scalar("Loss/train", loss, epoch)

                            val_counter += 1
        self.writer.flush()

    def mIoU(self):
        for i, (points, target) in enumerate(self.dataloader_test):
            points = points.to(self.device)
            target = target.to(self.device, dtype = torch.int64)
            pred, _ = self.net(points)

            # Find arg max of prediction:
            max_ = torch.max(pred,1)[1]

            ExtractH5Data.DataExtractor.Visualize_shapeInColors(self, points=points[0,:,:], predictions=max_[0], labels=target[0])
            input("...")

if __name__ == "__main__":

    trainer = Trainer()
    trainer.train()
    #trainer.mIoU()


"""
0 - ceiling
1 - floor
2 - wall
3 - beam
4 - column
5 - window
6 - door
7 - table
8 - chair
9 - sofa
10 - bookcase
11 - board
12 - clutter
"""

""" new interresting classes
0 - window
1 - door
2 - wall
3 - table
4 - other
"""