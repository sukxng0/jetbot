import cv2
import torch
import torch.nn.functional as F
import time
import numpy as np
from new_model import TrafficSignMobileNet  # 사전에 정의된 모델 클래스

# 클래스 이름
class_names = ['green_light', 'left_sign', 'red_light', 'right_sign', 'stop_sign']

# 디바이스 설정
device = torch.device("cpu")
print(f"Using device: {device}")

# 모델 로딩
model = TrafficSignMobileNet(num_classes=len(class_names))
model.load_state_dict(torch.load("traffic_sign_mobilenet.pth", map_location=device))
model.to(device)
model.eval()

# OpenCV 기반 전처리 함수 (빠름)
def preprocess_with_cv2(frame):
    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5  # Normalize
    return tensor.unsqueeze(0).to(device)

# GStreamer 카메라 파이프라인
def gstreamer_pipeline(
        capture_width=640, capture_height=480,
        display_width=640, display_height=480,
        framerate=5, flip_method=0):
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

# 카메라 시작
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 시간 관리
last_inference_time = 0
last_display_time = 0
last_terminal_time = 0

# 출력 유지 변수
detected_label = None
detected_confidence = 0.0

# 프레임 주기 설정
inference_interval = 0.2  # 추론 5FPS
display_interval = 0.1    # 디스플레이 10FPS
terminal_interval = 5.0   # 터미널 출력 5초마다

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    now = time.time()

    # 추론 주기 체크
    if now - last_inference_time > inference_interval:
        input_tensor = preprocess_with_cv2(frame)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        confidence_val = confidence.item()
        pred_idx = pred.item()

        if confidence_val > 0.3:
            detected_label = class_names[pred_idx]
            detected_confidence = confidence_val
        else:
            detected_label = None
            detected_confidence = 0.0

        last_inference_time = now

    # 프레임에 결과 표시
    if detected_label:
        text = f"{detected_label} ({detected_confidence*100:.1f}%)"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.rectangle(frame, (5, 5), (310, 60), (0, 255, 0), 2)

    # 디스플레이 주기 체크
    if now - last_display_time > display_interval:
        cv2.imshow("Traffic Sign Detection", frame)
        last_display_time = now

    # 터미널 출력
    if detected_label and (now - last_terminal_time > terminal_interval):
        print(f"[{time.strftime('%H:%M:%S')}] Detected: {detected_label} ({detected_confidence*100:.1f}%)")
        last_terminal_time = now

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
