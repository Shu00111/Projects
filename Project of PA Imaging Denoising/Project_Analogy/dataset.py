import numpy as np
from Project_Analogy.dataload import Dataload

class DataSet():
    def __init__(self):
        super().__init__()
        self.transform = True
        dataload = Dataload()
        self.X_data, self.Y_data, self.len_X = dataload.Load()

    def eval(self):
        self.transform = True
    
    def train(self):
        self.transform = True
    
    def __getitem__(self, index):
        index = index % 8
        x_brain = self.X_data[index]
        y_brain = self.Y_data[index]
        x_brain, y_brain = self.preprocess(x_brain, y_brain)
        x_brain = x_brain.reshape(1, x_brain.shape[1])
        y_brain = y_brain.reshape(1, y_brain.shape[1])
        return x_brain, y_brain

    def preprocess(self, x_data, y_data):
        if self.transform:
            i = np.random.randint(0, 256)
            brain_train = []
            truth_train = []
            brain_train.append(x_data[i])
            truth_train.append(y_data[i])
            x_data = np.array(brain_train)
            y_data = np.array(truth_train)
        return x_data, y_data

    def __len__(self):
        return self.len_X

