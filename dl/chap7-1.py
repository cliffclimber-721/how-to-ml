import torch
import torch.nn as nn # neural net 구성 요소
import torch.nn.functional as F # 딥러닝에 자주 사용되는 수학적 함수
import torch.optim as optim # 최적화 함수
import torchvision.datasets as dsets
import torchvision.transforms as transforms # 데이터 형태 지정 가능
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import SGDClassifier
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader, TensorDataset

# train=True : training dataset 받는 코드
train_dataset = dsets.FashionMNIST('dataset/', 
                                           train=True, 
                                           download=True,
                                           transform=transforms.ToTensor())

# train=False : test dataset 받는 코드
test_dataset = dsets.FashionMNIST('dataset/', 
                                          train=False, 
                                          download=True,
                                          transform=transforms.ToTensor())

train_target = train_dataset.targets.numpy()
test_target = test_dataset.targets.numpy()

print(train_dataset.data.shape, train_target.data.shape)
print(test_dataset.data.shape, test_target.data.shape)

fig, axs = plt.subplots(1, 30, figsize=(10, 10))
for i in range(30):
    img, label = train_dataset[i]
    axs[i].imshow(img.squeeze().numpy(), cmap="gray_r")
    axs[i].axis('off')
    axs[i].set_title(f"{label}")
#plt.tight_layout()
#plt.show()

#print([train_target[i] for i in range(10)])

#print(np.unique(train_target, return_counts=True))

# logistic regression
train_scaled = train_dataset.data.float() / 255.0
train_scaled = train_scaled.reshape(-1, 28*28).numpy()
print(train_scaled.data.shape)

sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score'])) # 0.8313666666666666

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print(train_scaled.data.shape, train_target.data.shape)
print(val_scaled.data.shape, val_target.data.shape)

train_scaled = torch.tensor(train_scaled, dtype=torch.float32)
train_target = torch.tensor(train_target, dtype=torch.long)
val_scaled = torch.tensor(val_scaled, dtype=torch.float32)
val_target = torch.tensor(val_target, dtype=torch.long)

train_loader = DataLoader(TensorDataset(train_scaled, train_target), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_scaled, val_target), batch_size=32)

dense = nn.Sequential(
    nn.Linear(784, 10), # 이 코드가 Dense 레이어에 해당됨
    nn.Softmax()
)

model = nn.Sequential(dense)
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(dense.parameters(), lr=0.001)
acc = Accuracy(task="multiclass", num_classes=10)

class ModelCompile(nn.Module):
    def __init__(self, model, criterion, opt, acc):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.acc = acc

    def forward(self, x):
        return self.model(x)
    
    def compute_loss(self, predictions, labels):
        return self.criterion(predictions, labels)
    
    def compute_metrics(self, predictions, labels):
        results= {}
        for metric in self.metrics:
            results[metric.__class__.__name__] = metric(predictions, labels)
        return results

compiled_model = ModelCompile(
    model=dense,
    criterion=criterion,
    opt=opt,
    acc=[acc]
)

print(train_target[:10])

for epoch in range(50):
    running_loss = 0.0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward pass 및 최적화
        opt.zero_grad()
        loss.backward()
        opt.step()

    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels).item()
            val_acc += acc(val_outputs, val_labels)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    model.train()

print("Training complete!")