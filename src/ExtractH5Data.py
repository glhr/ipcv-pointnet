import numpy as np
import h5py
import os
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import sys


class DataExtractor:

    def getDataFiles(self, path):
        return [line.rstrip() for line in open(path)]

    def loadDataFile(self, filename, path2):
        return self.load_h5(filename, path2)

    def load_h5(self, h5_filename, path2):
        f = h5py.File(os.path.join(path2, h5_filename), 'r')
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)


    def returnData(self):
        data = self.GetData()
        return data


    def GetData(self):
        PATH = '/media/gala/DataDisk/2021_gala_ta/ta-vap/pointnet/indoor3d_sem_seg_hdf5_data'
        path2 = '/media/gala/DataDisk/2021_gala_ta/ta-vap/pointnet/'
        ALL_FILES = self.getDataFiles(os.path.join(PATH, 'all_files.txt'))
        room_filelist = [line.rstrip() for line in open(os.path.join(PATH, 'room_filelist.txt'))]
        print(len(ALL_FILES))

        data_batch_list = []
        label_batch_list = []
        counter = 0
        for h5_filename in ALL_FILES:
            if counter < 12:  # If more data is needed up the number!
                print(h5_filename)
                data_batch, label_batch = self.loadDataFile(h5_filename, path2)
                data_batch_list.append(data_batch)
                label_batch_list.append(label_batch)
                counter += 1
            else:
                break
        data_batches = np.concatenate(data_batch_list, 0)
        label_batches = np.concatenate(label_batch_list, 0)

        test_area = 'Area_' + str(4)

        train_idxs = []
        test_idxs = []
        for i, room_name in enumerate(room_filelist):
            if i < len(data_batches):
                if test_area in room_name:
                    test_idxs.append(i)
                else:
                    train_idxs.append(i)

        train_data = data_batches[train_idxs, ...]
        train_label = label_batches[train_idxs]
        test_data = data_batches[test_idxs, ...]
        test_label = label_batches[test_idxs]

        # Correct labels into our own desire
        # np.set_printoptions(threshold=sys.maxsize)
        #train_label = DataExtractor.changeLabels(self, labels=train_label)
        #DataExtractor.FindDistributionOfPoints(self, labels=train_label)
        #print('trainlabels: ', train_label[0:32, :], train_label.shape)
        #test_label = DataExtractor.changeLabels(self, labels=test_label)
        #DataExtractor.FindDistributionOfPoints(self, labels=test_label)

        #reshape data to work with PoinetNet Architecture
        train_data = train_data[:, :, 0:3]
        train_data = train_data.reshape(-1, 3, 4096)

        test_data = test_data[:, :, 0:3]
        test_data = test_data.reshape(-1, 3, 4096)

        return train_data, train_label, test_data, test_label


    def FindDistributionOfPoints(self, labels):
        Label4 = (labels == 4).sum()
        Label0 = (labels == 0).sum()
        Label1 = (labels == 1).sum()
        Label2 = (labels == 2).sum()
        Label3 = (labels == 3).sum()
        totalPoints = len(labels)*len(labels[1])
        print('label sizes: ', 'Window 0: ',(Label0/totalPoints)*100,'% door 1: ', (Label1/totalPoints)*100,'% wall 2: ',
              (Label2/totalPoints)*100,'% table 3: ', (Label3/totalPoints)*100,'% other 4: ', (Label4/totalPoints)*100, "%")


    def changeLabels(self, labels):
        newLabels = labels
        newLabels[labels == 3] = 4
        newLabels[labels == 4] = 4
        newLabels[labels == 0] = 4
        newLabels[labels == 1] = 4
        newLabels[labels > 7] = 4

        newLabels[labels == 5] = 0
        newLabels[labels == 6] = 1
        newLabels[labels == 2] = 2
        newLabels[labels == 7] = 3
        return newLabels


    def visualize_shape(self, points):
        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        pyplot.xlabel("X axis label")
        pyplot.ylabel("Y axis label")
        pyplot.show()



    def Visualize_shapeInColors(self, predictions, points, labels):

        points = points.cpu().numpy()
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        clouds = points.reshape(4096, 3)

        #predictions = DataExtractor.changeLabels(self, labels=predictions)
        #labels = DataExtractor.changeLabels(self, labels=labels)

        # Calculate acc
        acc = DataExtractor.CalculateACC_ForTest(self, prediction=predictions, label=labels)

        print(f'Accuracy is: {acc}%')
        window = []
        door = []
        wall = []
        table = []
        remaining = []
        for i, cloud in enumerate(clouds):
            if predictions[i] == 0:
                window.append(cloud)
            elif predictions[i] == 1:
                door.append(cloud)
            elif predictions[i] == 2:
                wall.append(cloud)
            elif predictions[i] == 3:
                table.append(cloud)
            else:
                remaining.append(cloud)


        print('remaining size: ', np.asarray(remaining).shape, 'ceiling: ', np.asarray(window).shape, 'floor: ',
              np.asarray(door).shape, 'wall: ', np.asarray(wall).shape, 'table: ', np.asarray(table).shape)
        fig = go.Figure(data=[go.Scatter3d(x=np.asarray(remaining)[:,0],
                                            y=np.asarray(remaining)[:,1],
                                            z=np.asarray(remaining)[:,2],
                                            mode='markers', marker=dict(
                                            color='rgba(255, 255, 255, 0.5)', size=2,
                                            ), )])
        if len(window) > 0:
            fig.add_trace(go.Scatter3d(x=np.asarray(window)[:,0], y=np.asarray(window)[:,1], z=np.asarray(window)[:,2],
                                          mode='markers', marker=dict(
                                          color='rgba(255, 0, 0, 0.5)', size=2,
                                          ), ))
        if len(door) > 0:
            fig.add_trace(go.Scatter3d(x=np.asarray(door)[:,0], y=np.asarray(door)[:,1], z=np.asarray(door)[:,2],
                                     mode='markers', marker=dict(color='rgba(0, 255, 0, 0.5)', size=2,
                                                                   ), ))

        if len(wall) > 0:
            fig.add_trace(go.Scatter3d(x=np.asarray(wall)[:,0], y=np.asarray(wall)[:,1], z=np.asarray(wall)[:,2],
                                     mode='markers', marker=dict(color='rgba(0, 0, 255, 0.5)', size=2,
                                                                   ), ))

        if len(table) > 0:
            fig.add_trace(go.Scatter3d(x=np.asarray(table)[:,0], y=np.asarray(table)[:,1], z=np.asarray(table)[:,2],
                                     mode='markers', marker=dict(color='rgba(255, 0, 255, 0.5)', size=2,
                                                                   ), ))

        fig.show()


    def CalculateACC(self, prediction, label):
        #print(prediction.shape, label.shape)
        correctGuess = (prediction == label).sum()
        #print(f'correct guesses out of {len(prediction.flatten())}: {correctGuess}')
        acc = 100*(correctGuess/len(prediction.flatten()))
        #print(acc)
        return acc



    def CalculateACC_ForTest(self, prediction, label):
        print(prediction.shape, len(label))
        correctGuess = (prediction == label).sum()
        #print(f'correct guesses out of {len(prediction)}: {correctGuess}')
        return 100*correctGuess/(len(prediction))



    """
    def Visualize_shapeInColorsAllClasses(self, predictions, points, labels):

        points = points.cpu().numpy()
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        clouds = points.reshape(4096, 3)

        # Calculate acc
        acc = DataExtractor.CalculateACC_ForTest(self, prediction=predictions, label=labels)

        print(f'Accuracy is: {acc}%')
        ceiling = []
        floor = []
        wall = []
        beam = []
        column = []
        window
        for i, cloud in enumerate(clouds):
            if predictions[i] == 0:
                window.append(cloud)
            elif predictions[i] == 1:
                door.append(cloud)
            elif predictions[i] == 2:
                wall.append(cloud)
            elif predictions[i] == 3:
                table.append(cloud)
            else:
                remaining.append(cloud)


        print('remaining size: ', np.asarray(remaining).shape, 'ceiling: ', np.asarray(window).shape, 'floor: ',
              np.asarray(door).shape, 'wall: ', np.asarray(wall).shape, 'table: ', np.asarray(table).shape)
        fig = go.Figure(data=[go.Scatter3d(x=np.asarray(remaining)[:,0],
                                            y=np.asarray(remaining)[:,1],
                                            z=np.asarray(remaining)[:,2],
                                            mode='markers', marker=dict(
                                            color='rgba(255, 255, 255, 0.5)', size=2,
                                            ), )])
        if len(window) > 0:
            fig.add_trace(go.Scatter3d(x=np.asarray(window)[:,0], y=np.asarray(window)[:,1], z=np.asarray(window)[:,2],
                                          mode='markers', marker=dict(
                                          color='rgba(255, 0, 0, 0.5)', size=2,
                                          ), ))
        if len(door) > 0:
            fig.add_trace(go.Scatter3d(x=np.asarray(door)[:,0], y=np.asarray(door)[:,1], z=np.asarray(door)[:,2],
                                     mode='markers', marker=dict(color='rgba(0, 255, 0, 0.5)', size=2,
                                                                   ), ))

        if len(wall) > 0:
            fig.add_trace(go.Scatter3d(x=np.asarray(wall)[:,0], y=np.asarray(wall)[:,1], z=np.asarray(wall)[:,2],
                                     mode='markers', marker=dict(color='rgba(0, 0, 255, 0.5)', size=2,
                                                                   ), ))

        if len(table) > 0:
            fig.add_trace(go.Scatter3d(x=np.asarray(table)[:,0], y=np.asarray(table)[:,1], z=np.asarray(table)[:,2],
                                     mode='markers', marker=dict(color='rgba(255, 0, 255, 0.5)', size=2,
                                                                   ), ))

        fig.show()
    """
if __name__ == "__main__":
    extractor = DataExtractor()
    train_data, train_label, test_data, test_label = extractor.GetData()
    points = train_data[0]
    labels = train_label[0]
    extractor.Visualize_shapeInColors(labels, points, labels)