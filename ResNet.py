import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

#//-------------------------------------------------------------------------
# Global Variable For training
# You just use the following hyper-parameters
BATCH_SIZE = 100                      #한번에 모델에 입력되는 데이터개수
NUM_EPOCH = 50                        #전체 데이터셋을 몇번 반복하여 학습할지 나타냄
LEARNING_RATE = 0.01                  #모델이 학습하는 속도 조절, 빠를수록 빠르지만 빠르다고 최적화 되는건 아님
CRITERION = nn.CrossEntropyLoss()     #손실함수, 크로스엔트로피로 확률로서 평가
#-----------------------------------------------------------------------------
# CIFAR10 Dataset
train_dataset = dsets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)   #데이터셋 로드 및 훈련과 데스트에 필요한 데이터 준비
#------------------------------------------------------------------------------
def fit(model,train_loader):
    model.train()
    device = next(model.parameters()).device.index
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    losses = []
    for i, data in enumerate(train_loader):
        image = data[0].type(torch.FloatTensor).cuda(device)
        label = data[1].type(torch.LongTensor).cuda(device)

        pred_label = model(image)
        loss = CRITERION(pred_label, label)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses)/len(losses)
    return avg_loss                                        #모델을 학습하고 평균 손실값 반환
#----------------------------------------------------------------------------------------------
def eval(model, test_loader):
    model.eval()
    device = next(model.parameters()).device.index
    pred_labels = []
    real_labels = []

    for i, data in enumerate(test_loader):
        image = data[0].type(torch.FloatTensor).cuda(device)
        label = data[1].type(torch.LongTensor).cuda(device)
        real_labels += list(label.cpu().detach().numpy())

        pred_label = model(image)
        pred_label = list(pred_label.cpu().detach().numpy())
        pred_labels += pred_label

    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)
    pred_labels = pred_labels.argmax(axis=1)
    acc = sum(real_labels==pred_labels)/len(real_labels)*100
    return acc                         #모델을 평가 모드로 설정하고, 실제 값과 예측값 비교하여 정확도를 반환한다.

#---------------------------------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.resblock1 = ResBlock(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.resblock2 = ResBlock(128)
        self.pool3 = nn.AvgPool2d(kernel_size=8, stride=8)

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)

        x = self.relu3(self.conv3(x))
        x = self.resblock1(x)
        x = self.pool2(x)

        x = self.relu4(self.conv4(x))
        x = self.resblock2(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


resnet_model = SimpleResNet().cuda()
train_loss4 = []
test_accuracy4 = []
for epoch in tqdm(range(NUM_EPOCH)):
    train_loss4.append(fit(resnet_model, train_loader))
    test_accuracy4.append(eval(resnet_model, test_loader))

print("\n🔍 [SimpleCNN] 각 레이어별 파라미터 수 (Weight / Bias)\n" + "-"*60)
total_weight = 0
total_bias = 0

for name, param in cnn_model.named_parameters():
    param_count = param.numel()
    if 'weight' in name:
        print(f"{name:<35} | Weight Params: {param_count}")
        total_weight += param_count
    elif 'bias' in name:
        print(f"{name:<35} | Bias Params:   {param_count}")
        total_bias += param_count

print("-"*60)
print(f"총 가중치(Weights): {total_weight:,}")
print(f"총 바이어스(Biases): {total_bias:,}")
print(f"총 파라미터 수: {total_weight + total_bias:,}")


# 학습 손실 그래프화
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss4, label='Train Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 테스트 정확도 그래프화
plt.subplot(1,2,2)
plt.plot(test_accuracy4, label='Test Accuracy', color='green')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()