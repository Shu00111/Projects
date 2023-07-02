import torch.nn as nn
import torch.utils.data
import numpy as np
from scipy import io
import os
import glob
from Project_Analogy.Model import Unet
from Project_Analogy.dataload import Dataload

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(class_num=1, channel_num=1)
model.load_state_dict(torch.load(r"D:\19023\PycharmProjects\Machine_Learning\Project_Analogy\model.pth"))
model.to(device=device)
model.eval()
loss_func = nn.MSELoss(reduction="mean")
dataload = Dataload()
X_data, Y_data, length = dataload.Load()
print(len(X_data))

Truth_pred = []
Truth_pred_dict = []

total_path = "D:/19023/PycharmProjects/Machine_Learning/Project_Analogy/Outcome"
outcome_path = glob.glob(os.path.join(total_path, "brain*.mat"))

loss = 0.0
i = 0
for i in range(length):
    brain_test = np.array(X_data[i])
    brain_truth = np.array(Y_data[i])

    brain_test = brain_test.reshape(256, 1, 1200)
    brain_truth = brain_truth.reshape(256, 1, 1200)

    brain_test = torch.from_numpy(brain_test)
    brain_truth = torch.from_numpy(brain_truth)

    brain_test = brain_test.to(device=device, dtype=torch.float32)
    brain_truth = brain_truth.to(device=device, dtype=torch.float32)

    brain_pred = model(brain_test)
    loss = loss_func(brain_pred, brain_truth)
    loss = loss.detach().cpu().numpy()
    brain_pred = brain_pred.detach().cpu().numpy()
    brain_pred = brain_pred.reshape(256, 1200)
    Truth_pred = brain_pred.tolist()
    print("loss:", loss.item())
    io.savemat(outcome_path[i], {'Y': Truth_pred})