import os
import csv
import copy
import pyautogui
import time

import cv2
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
import tools

# 마우스 이벤트 발생 간격 설정
MOUSE_EVENT_INTERVAL = 1  # 초 단위
pyautogui.FAILSAFE = False

# 메인 함수 정의
def main():
    # 인자 가져오기
    args = tools.get_args()

    # 카메라 및 캡처 크기 설정
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # 정적 이미지 모드, 최소 신뢰도 설정 가져오기
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # 모델 이름 가져오기
    model_name = args.model

    # 바운딩 박스 사용 여부 설정
    use_brect = not args.unuse_brect

    # 카메라 준비
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Mediapipe Hands 모듈 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # 키포인트 분류기 모델 로드
    model_path = os.path.join('model', model_name, 'keypoint_classifier.tflite')
    keypoint_classifier = KeyPointClassifier(model_path=model_path)

    # 레이블 로드
    label_path = os.path.join('model', model_name, 'keypoint_classifier_label.csv')
    with open(label_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    # FPS 측정 모듈
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 화면 크기 가져오기
    screen_width, screen_height = pyautogui.size()
    
    # 마우스 커서 이동 범위 설정
    mouse_move_range = 400
    
    # 검지의 이전 위치 초기화
    prev_index_y = None
    
    # 마우스 이벤트 마지막 실행 시간 초기화
    last_mouse_event_time = 0
    
    while True:
        fps = cvFpsCalc.get()
    
        # 키 처리 (ESC: 종료)
        key = cv2.waitKey(10)
        if key == 27:  # ESC 또는 q
            break
    
        # 카메라 캡처
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)  # 미러 표시
        debug_image = copy.deepcopy(image)
    
        # 검출 수행
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
    
        # 키포인트 분류
        brect = None
        landmark_list = None
        handedness = None
        hand_sign_id = 0
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 바운딩 박스 계산
                brect = tools.calc_bounding_rect(debug_image, hand_landmarks)
                
                # 랜드마크 계산
                landmark_list = tools.calc_landmark_list(debug_image, hand_landmarks)
        
                # 상대 좌표 및 정규화된 좌표로 변환
                pre_processed_landmark_list = tools.pre_process_landmark(landmark_list)
                # 키포인트 분류
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                # 손가락 끝점 검출
                for idx, landmark in enumerate(landmark_list):
                    if idx == 8:  # 검지 끝점
                        index_tip = landmark
                
                if hand_sign_id == 0:
                    # 검지 끝점이 검출되었을 때
                    if index_tip is not None:
                        index_y = index_tip[1]  # 검지 끝점의 y 좌표
                    
                        # 이전 검지의 위치가 존재할 경우
                        if prev_index_y is not None:
                            # 이전 위치와 현재 위치의 차이 계산
                            y_diff = index_y - prev_index_y
                    
                            # 스크롤 방향 결정 및 스크롤
                            if y_diff > 5:
                                pyautogui.scroll(-100)  # 위로 스크롤
                            elif y_diff < -5:
                                pyautogui.scroll(100)  # 아래로 스크롤
                    
                        # 현재 검지의 위치를 이전 위치로 업데이트
                        prev_index_y = index_y
    
                # 손목 랜드마크를 기반으로 마우스 커서 이동
                if hand_sign_id == 5 or hand_sign_id == 7:  # 6번째와 8번째 동작일 때만 마우스 움직이기
                    # 손목 랜드마크 인덱스
                    wrist_index = 0
                    wrist_landmark = hand_landmarks.landmark[wrist_index]
    
                    # 손목 위치 가져오기
                    wrist_x = int(wrist_landmark.x * debug_image.shape[1])
                    wrist_y = int(wrist_landmark.y * debug_image.shape[0])
    
                    # 마우스 커서 이동
                    mouse_x = np.interp(wrist_x, [0, debug_image.shape[1]], [-mouse_move_range, screen_width + mouse_move_range])
                    mouse_y = np.interp(wrist_y, [0, debug_image.shape[0]], [-mouse_move_range, screen_height + mouse_move_range])
                    pyautogui.moveTo(mouse_x, mouse_y)
                    
                # 현재 시간 가져오기
                current_time = time.time()
                # 마우스 이벤트 발생 간격을 확인하여 일정 시간이 지나면 마우스 이벤트 발생
                if current_time - last_mouse_event_time >= MOUSE_EVENT_INTERVAL:
                    # 4번째 동작일 때 (음량 줄이기)
                    if hand_sign_id == 3:
                       pyautogui.press('volumedown')
                       last_mouse_event_time = current_time
                    # 5번째 동작일 때 (음량 키우기)
                    elif hand_sign_id == 4:
                       pyautogui.press('volumeup')
                       last_mouse_event_time = current_time
                    # 3번째 동작일 때 (홈 화면으로 이동)
                    elif hand_sign_id == 2:
                       #pyautogui.hotkey('win', 'd')
                       last_mouse_event_time = current_time
                    # 9번째 동작일 때
                    elif hand_sign_id == 8:
                        # 마우스 클릭 이벤트 발생
                        #pyautogui.hotkey('alt', 'f4')
                        last_mouse_event_time = current_time
                    # 7번째 동작일 때 (Alt + Tab)
                    elif hand_sign_id == 6:
                       # Alt + Tab 키 누르기 (창 전환)
                       #pyautogui.hotkey('alt', 'tab')
                       last_mouse_event_time = current_time 
                       
                # 2번째 동작이 감지되면 마우스 클릭
                if hand_sign_id == 1:
                    #pyautogui.click()
                    last_mouse_event_time = current_time                 
    
        
    # 그리기
        debug_image = tools.draw_bounding_rect(use_brect, debug_image, brect)
        if landmark_list is not None:  # landmark_list가 None이 아닐 경우에만 그리기 함수 호출
            debug_image = tools.draw_landmarks(debug_image, landmark_list)
        debug_image = tools.draw_info_text(debug_image, model_name, brect, handedness, keypoint_classifier_labels[hand_sign_id], fps)
        
        cv2.imshow('Hand motion recognition', debug_image)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
