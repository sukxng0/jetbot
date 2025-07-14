import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from new_model import TrafficSignMobileNet  # model.py에 정의된 클래스

# 하이퍼파라미터
batch_size = 8
epochs = 30
learning_rate = 0.0005
image_size = 224  # MobileNetV2 입력 크기

# 전처리 정
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, hue = 0.1, saturation = 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 불러오기
data_dir = "Dataset"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 클래스 수 확인
num_classes = len(dataset.classes)
print("클래스 목록:", dataset.classes)

# 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = TrafficSignMobileNet(num_classes=num_classes).to(device)

# 손실 함수, 최적화기 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")

# 모델 저장
torch.save(model.state_dict(), "traffic_sign_mobilenet.pth")
print("학습 완료! 모델이 'traffic_sign_mobilenet.pth'로 저장되었습니다.")
