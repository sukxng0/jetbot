import torch.nn as nn
from torchvision.models import mobilenet_v2

class TrafficSignMobileNet(nn.Module):
    def __init__(self, num_classes=5):
        super(TrafficSignMobileNet, self).__init__()
        # Pretrained MobileNetV2 로드
        self.base_model = mobilenet_v2(pretrained=True)
        # 마지막 classifier 교체: 원래 (1280 → 1000) → (1280 → num_classes)
        self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel, num_classes)

    def forward(self, x):
        return self.base_model(x)
