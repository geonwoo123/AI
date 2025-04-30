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
    return acc

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Fully-connected layer
        self.fc1_1 = nn.Linear(3*32*32, 8*28*28)
        self.act1_1 = nn.ReLU()
        self.fc1_2 = nn.Linear(8*28*28, 8*24*24)
        self.act1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.fc2_1 = nn.Linear(8*12*12, 16*8*8)
        self.act2_1 = nn.ReLU()
        self.fc2_2 = nn.Linear(16*8*8, 16*4*4)
        self.act2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Output layer
        self.out = nn.Linear(16*2*2, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)

        x = self.act1_1(self.fc1_1(x))
        x = self.act1_2(self.fc1_2(x))
        x = x.view(-1, 8, 24, 24)
        x = self.pool1(x)
        x = x.view(-1, 8*12*12)

        x = self.act2_1(self.fc2_1(x))
        x = self.act2_2(self.fc2_2(x))
        x = x.view(-1, 16, 4, 4)
        x = self.pool2(x)
        x = x.view(-1, 16*2*2)

        out = self.out(x)
        return out              #cnn아닌 mlp방식

#-------------------------------------------------------------
mlp_model = SimpleMLP().cuda()
train_loss1 = []
test_accuracy1 = []
for epoch in tqdm(range(NUM_EPOCH)):
    train_loss1.append(fit(mlp_model, train_loader))
    test_accuracy1.append(eval(mlp_model, test_loader))
summary(mlp_model, input_size = (3,32,32))
print("\n🔍 각 레이어별 파라미터 수 (Weight / Bias)\n" + "-"*50)
total_weight = 0
total_bias = 0

for name, param in mlp_model.named_parameters():
    param_count = param.numel()
    if 'weight' in name:
        print(f"{name:<30} | Weight Params: {param_count}")
        total_weight += param_count
    elif 'bias' in name:
        print(f"{name:<30} | Bias Params:   {param_count}")
        total_bias += param_count

print("-"*50)
print(f"총 가중치(Weights): {total_weight:,}")
print(f"총 바이어스(Biases): {total_bias:,}")
print(f"총 파라미터 수: {total_weight + total_bias:,}")


# 학습 손실 시각화
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss1, label='Train Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 테스트 정확도 시각화
plt.subplot(1,2,2)
plt.plot(test_accuracy1, label='Test Accuracy', color='green')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
