import torch.nn as nn
import torch.utils.data
from Project_Analogy.Model import Unet
from Project_Analogy.dataset import DataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(class_num=1, channel_num=1)
model.to(device=device)
model.train()

epochs = 50000
batch_size = 128
lr = 0.00001

optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.MSELoss(reduction="mean")
dataset = DataSet()
dataset.train()
loader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    for x_data, y_data in loader:
        optimizer.zero_grad()
        x_data = x_data.to(device=device, dtype=torch.float32)
        y_data = y_data.to(device=device, dtype=torch.float32)
        pred = model(x_data)
        loss = loss_func(pred, y_data)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0 and epoch > 5:
        torch.save(model.state_dict(), str(epoch)+'model.pth')
    print("epoch", epoch, "loss:", loss.item())

torch.save(model.state_dict(), 'model.pth')


