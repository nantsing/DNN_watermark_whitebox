import torch
from model import CNN
import data
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.models as models
from cifar10 import CIFAR10

save = 'wm_model.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
) 

cnn = CNN().to(device)
cnn.load_state_dict(torch.load(save))

test_set = CIFAR10('test', transform= transform)
test_data = Data.DataLoader(dataset=test_set, batch_size=500, shuffle=False)

total = 0
correct = 0
cnn.eval()
for index, (data, label) in enumerate(test_data):
    data = data.to(device)
    label = label.to(device)
    output = cnn(data)
    _, pred = torch.max(output.data, 1)
    
    total += len(label)
    correct += (pred == label).sum()

print("Eval Accuracy: {:.3f} %".format(correct/total*100))