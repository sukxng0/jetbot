import cv2
import torch
import torch.nn.functional as F
import time
import numpy as np
from new_model import TrafficSignMobileNet
from jetbotmini import Robot

# 클래스 이름 정의
class_names = ['green_light', 'left_sign', 'red_light', 'right_sign', 'stop_sign']

# 디바이스 설정
device = torch.device("cpu")
print(f"Using device: {device}")

# 모델 로딩
model = TrafficSignMobileNet(num_classes=len(class_names))
model.load_state_dict(torch.load("traffic_sign_mobilenet.pth", map_location=device))
model.to(device)
model.eval()

# 로봇 초기화
robot = Robot()
robot.set_motors(0.0, 0.0)

# 영상 전처리
def preprocess_with_cv2(frame):
    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)

# 카메라 파이프라인
def gstreamer_pipeline(capture_width=640, capture_height=480,
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

# 변수 초기화
last_inference_time = 0
last_display_time = 0
last_label_time = 0
label_cooldown = 5.0  # 표지판 반응 쿨다운 (초)

detected_label = None
detected_confidence = 0.0

is_driving = False
is_stopping = False

current_speed = 0.0
target_speed = 0.0
max_speed = 0.4
accel_time = 2.0
accel_step = 0.05
accel_interval = accel_time * accel_step / max_speed

decel_time = 2.0
decel_step = accel_step
decel_interval = accel_interval

last_accel_time = 0
last_decel_time = 0

detection_pause_until = None

# 오른쪽 바퀴가 더 빠른 경우 → 속도 줄이기
motor_bias = -0.03

inference_interval = 0.2
display_interval = 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    now = time.time()

    # === 추론 ===
    if not is_stopping and (detection_pause_until is None or now >= detection_pause_until):
        if now - last_inference_time > inference_interval:
            input_tensor = preprocess_with_cv2(frame)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)

            confidence_val = confidence.item()
            pred_idx = pred.item()
            last_inference_time = now

            # === 쿨다운 이후에만 반응 ===
            if confidence_val > 0.3 and (now - last_label_time > label_cooldown):
                detected_label = class_names[pred_idx]
                detected_confidence = confidence_val
                last_label_time = now
            else:
                detected_label = None
                detected_confidence = 0.0

            # 초록불 감지 후 주행 시작
            if detected_label == "green_light" and not is_driving:
                print("✅ 초록불 감지 → 10초 탐지 중지 & 2초 가속")
                is_driving = True
                target_speed = max_speed
                last_accel_time = now
                detection_pause_until = now + 10  # 10초 동안 탐지 중지

            # 멈춤표지 감지 시 감속
            elif detected_label == "stop_sign" and is_driving and not is_stopping:
                print("🛑 멈춤표지 감지 → 감속 후 종료")
                is_stopping = True
                last_decel_time = now
                detection_pause_until = None

    # === 가속 ===
    if is_driving and not is_stopping:
        if current_speed < target_speed and now - last_accel_time > accel_interval:
            current_speed += accel_step
            current_speed = min(current_speed, target_speed)
            last_accel_time = now
            print(f"[가속] 속도: {current_speed:.2f}")
        robot.set_motors(current_speed, current_speed + motor_bias)

    # === 감속 ===
    if is_stopping:
        if current_speed > 0 and now - last_decel_time > decel_interval:
            current_speed -= decel_step
            current_speed = max(current_speed, 0.0)
            last_decel_time = now
            print(f"[감속] 속도: {current_speed:.2f}")
        robot.set_motors(current_speed, current_speed + motor_bias)
        if current_speed == 0.0:
            print("🛑 완전 정지 → 프로그램 종료")
            break

    # === 영상 출력 ===
    if detected_label:
        text = f"{detected_label} ({detected_confidence * 100:.1f}%)"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.rectangle(frame, (5, 5), (320, 60), (0, 255, 0), 2)

    cv2.imshow("Traffic Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정지 후 종료
robot.set_motors(0.0, 0.0)
cap.release()
cv2.destroyAllWindows()
