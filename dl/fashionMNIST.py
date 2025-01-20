# https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html 
# 해당 사이트 보고 chap7-1.py 코드 재작성했습니다. 내용은 크게 다를 건 없고 그대로 복사 붙여넣기 했습니다 ㅎㅎ

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 데이터셋 다운로드하기
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# batch size는 소그룹에 속하는 데이터 수를 의미함
# 데이터셋이 10000개라고 가정하면 이걸 64개씩 나눠 푼다고 생각하면 됨
# batch size가 크면 한 번에 처리해야할 데이터 양이 많아져서 학습속도가 느리고, 메모리 부족 문제가 발생할 수 있음
# 그렇다고 너무 작으면, 가중치가 자주 업데이트 되어있다보니까 훈련이 불안정해짐
batch_size=64

# Dataloader 는 데이터셋을 읽어와서 배치 단위로 데이터를 불러오는 거임
# 모델 학습을 좀더 효율적으로 진행할 수 있음
# 멀티스레딩/멀티프로세싱을 지원하고, 데이터를 병렬 처리할 수 있음
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for i, j in test_dataloader:
    print(f"Shape of i : [n, c, h, w]: {i.shape}")
    print(f"Shape of j : {j.shape} {j.dtype}")
    break

# 어떤 디바이스로 학습할건지 정의
device =(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 모델 만들기
class NeuralNetwork(nn.Module):
    def __init__(self):
        # 위에 사용했던 변수들을 상속받는 역할을 해주는 걸 super().__init__()이 함
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (i, j) in enumerate(dataloader):
        # i -> 입력 데이터, j -> 레이블 또는 정답 데이터(이미지 클래스 라벨)
        i, j = i.to(device), j.to(device)
        # 예측 오류 계산

        # 그래서 모델에 입력데이터 i를 전달해서 예측값을 생성함
        pred = model(i)
        loss = loss_fn(pred, j)

        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(i)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # 평가 먼저 한 다음,
    model.eval()
    test_loss, correct = 0, 0
    # auto grad 비활성화 -> gradient를 트래킹 하지 않아 필요한 메모리가 줄어들고 계산속도가 증가한다
    # item() 은 숫자값만 추출하게끔 하는 문법
    with torch.no_grad():
        for i, j in dataloader:
            i, j = i.to(device), j.to(device)
            pred = model(i)
            test_loss += loss_fn(pred, j).item()
            correct += (pred.argmax(1) == j).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n===============================================")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("DONE!")

torch.save(model.state_dict(), "mnistmodel.pth")
print("Saved mnistmodel.pth")