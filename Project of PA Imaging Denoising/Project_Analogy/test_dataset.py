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

    def __len__(self):
        return self.len_X