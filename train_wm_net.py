import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
from cifar10 import CIFAR10
from model import CNN

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
)

SAVE_PATH = 'wm_model.pth'
Generator_PATH = 'generator.pth'
WaterMark_PATH = 'watermark.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:\n',device)

cnn = CNN().to(device)
k = 5
embed_dim = 128
NUM_EPOCH = 50
lr = 0.01
batch_size = 32
show_loss_gap = 100

train_set = CIFAR10('train', transform= transform)
train_data = Data.DataLoader(dataset= train_set, batch_size= batch_size, shuffle = True)
valid_set = CIFAR10('valid', transform= transform)
valid_data = Data.DataLoader(dataset= valid_set, batch_size= 600, shuffle = False)


criterion = nn.CrossEntropyLoss().to(device)
# wm_criterion = F.binary_cross_entropy()

optimizer = optim.SGD(cnn.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       patience=4,
                                                       verbose=1,
                                                       factor=0.5,
                                                       min_lr=1e-5)


b = torch.ones(1, embed_dim).to(device)
# random X
w = cnn.get_parameter('conv2.0.weight')
w = torch.mean(w, dim= 0).view(1, -1)
X_cols = embed_dim
X_rows = w.size()[1]
X = np.random.randn(X_rows, X_cols)
X = torch.tensor(X, dtype=torch.float32).to(device)

for epoch in range(NUM_EPOCH):
    print('EPOCH:', epoch)
    cnn.train()
    for index, (data, label) in enumerate(train_data):
        
        data = data.to(device)
        label = label.to(device)
        output = cnn(data)
        loss = criterion(output, label)

        w = cnn.get_parameter('conv2.0.weight')
        w = torch.mean(w, dim= 0).view(1, -1).to(device)
        regularized_loss = k * torch.sum(F.binary_cross_entropy(input= torch.sigmoid(torch.matmul(w, X)), target= b)).to(device)

        optimizer.zero_grad()
        Loss = loss + regularized_loss
        (Loss).backward()
        optimizer.step()

        if (index+1)%show_loss_gap == 0 :
            print('index:',(index+1)//100,'  ','Loss=', Loss.item(), 'loss=', loss.item(), 'regularized_loss=', regularized_loss.item())

    cnn.eval()
    for index, (data, label) in enumerate(valid_data) :
        data = data.to(device)
        label = label.to(device)
        output = cnn(data)
        _,pred = torch.max(output,1)
        accuracy = sum(pred == label)/len(label) * 100
        print('test_batch_',index,':Accuracy is ', accuracy.item())

    scheduler.step(loss)

print("Done...... Save results......")
torch.save(X, Generator_PATH)
torch.save(b, WaterMark_PATH)
print("WaterMark and Generator has been saved at {} and {}".format(Generator_PATH, WaterMark_PATH))

torch.save(cnn.state_dict(), SAVE_PATH)
print("The watermarked model is save at {}".format(SAVE_PATH))