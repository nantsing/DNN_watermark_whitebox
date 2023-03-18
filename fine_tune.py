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

NUM_EPOCH = 10
lr = 0.01
batch_size = 32
show_loss_gap = 100

WM_PATH = 'wm_model.pth'
SAVE_PATH = 'ft_model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:\n',device)
cnn = CNN().to(device)
cnn.load_state_dict(torch.load(WM_PATH))

train_set = CIFAR10('train', transform= transform, fine_tune= True)
train_data = Data.DataLoader(dataset= train_set, batch_size= batch_size, shuffle = True)
valid_set = CIFAR10('valid', transform= transform, fine_tune= True)
valid_data = Data.DataLoader(dataset= valid_set, batch_size= 600, shuffle = False)

# print(len(train_data))
# print(len(valid_data))

criterion = nn.CrossEntropyLoss().to(device)
# wm_criterion = F.binary_cross_entropy()
# fix other parameters, only modify conv2d_1
optimizer = optim.SGD(cnn.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       patience=4,
                                                       verbose=1,
                                                       factor=0.5,
                                                       min_lr=1e-5)

for epoch in range(NUM_EPOCH):
    print('EPOCH:', epoch)
    cnn.train()
    for index, (data, label) in enumerate(train_data):
        
        data = data.to(device)
        label = label.to(device)
        output = cnn(data)

        # print(label)
        # print(output)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index+1)%show_loss_gap == 0 :
            print('index:',(index+1)//show_loss_gap,'  ', 'loss=', loss.item())

    cnn.eval()
    for index, (data, label) in enumerate(valid_data) :
        data = data.to(device)
        label = label.to(device)
        output = cnn(data)
        _,pred = torch.max(output,1)

        accuracy = sum(pred == label)/len(label) * 100
        print('test_batch_',index,':Accuracy is ', accuracy.item())

    scheduler.step(loss)


torch.save(cnn.state_dict(), SAVE_PATH)
print("The fine-tuned model is save at {}".format(SAVE_PATH))