import torch
import torch.nn as nn # neural net 구성 요소
import torch.nn.functional as F # 딥러닝에 자주 사용되는 수학적 함수
import torch.optim as optim # 최적화 함수
import torchvision.datasets as dsets
import torchvision.transforms as transforms # 데이터 형태 지정 가능
import matplotlib.pyplot as plt

train_dataset = dsets.MNIST('dataset/', 
                                           train=True, 
                                           download=True,
                                           transform=transforms.ToTensor())
test_dataset = dsets.MNIST('dataset/', 
                                          train=False, 
                                          download=True,
                                          transform=transforms.ToTensor())
train_target = train_dataset.targets
test_target = test_dataset.targets
print(train_dataset.data.shape)
# train=True : training dataset 받는 코드
# train=False : test dataset 받는 코드
# print("Number of training data : ", len(train_dataset)) -> 60000
# print("Number of test data : ", len(test_dataset)) -> 10000
# print("Number of training targets : ", len(train_target)) -> 60000
# print("Number of testing targets : ", len(test_target)) -> 10000

print("Number of training targets : ", len(train_target))
print("Number of testing targets : ", len(test_target))
