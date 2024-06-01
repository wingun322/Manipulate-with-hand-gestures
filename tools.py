import argparse
import cv2
import copy
import numpy as np
import itertools

import mediapipe as mp

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)  # 카메라 장치 번호
    parser.add_argument("--width", help='cap width', type=int, default=960)  # 캡처 너비
    parser.add_argument("--height", help='cap height', type=int, default=540)  # 캡처 높이

    parser.add_argument('--use_static_image_mode', action='store_true')  # 정적 이미지 모드 사용 여부
    parser.add_argument(
        "--min_detection_confidence",
        help='min_detection_confidence',
        type=float,
        default=0.7,
    )  # 최소 검출 신뢰도
    parser.add_argument(
        "--min_tracking_confidence",
        help='min_tracking_confidence',
        type=int,
        default=0.5,
    )  # 최소 추적 신뢰도

    parser.add_argument(
        "--model",
        type=str,
        default='com_manipulate',
    )  # 모델 이름

    parser.add_argument('--unuse_brect', action='store_true')  # 바운딩 박스 사용 여부

    args = parser.parse_args()

    return args

# 바운딩 박스 계산 함수 정의
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    if landmark_array.size == 0:  # 랜드마크 배열이 비어있을 경우
        return None

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# 바운딩 박스 그리기 함수 정의
def draw_bounding_rect(use_brect, image, brect):
    if use_brect and brect is not None:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)

    return image

# 랜드마크 리스트 계산 함수 정의
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # 키포인트
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# 랜드마크 전처리 함수 정의
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 상대 좌표로 변환
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 1차원 리스트로 변환
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # 정규화
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# 이전 포즈와 현재 포즈가 같은지 비교 함수 정의
def is_same_landmarks(landmarks1, landmarks2):
    threshold = 0.05  # 차이의 허용 범위

    if not landmarks1 or not landmarks2:  # 하나라도 None인 경우
        return False

    for i in range(len(landmarks1)):
        if abs(landmarks1[i][0] - landmarks2[i][0]) > threshold or abs(landmarks1[i][1] - landmarks2[i][1]) > threshold:
            return False

    return True

# 정보 텍스트 그리기 함수 정의
def draw_info_text(image, model_name, brect, handedness, hand_sign_text, fps):
    if brect is not None:  # 바운딩 박스가 None이 아닐 경우
        cv2.rectangle(image, (brect[0], brect[1]), (brect[0] + 170, brect[1] - 22), (0, 0, 0), -1)
        
        if handedness is not None and handedness.classification:  # handedness가 None이 아니고 classification 속성이 있을 경우
            info_text = f'{model_name}:{handedness.classification[0].label[0:]}:{hand_sign_text}'
        else:
            info_text = f'{model_name}:Unknown:{hand_sign_text}'
        
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f'FPS:{fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# 랜드마크 그리기 함수 정의
def draw_landmarks(image, landmark_list):
    # 랜드마크 연결 선 그리기
    for i, landmark in enumerate(landmark_list):
        if i in [0, 1, 2, 5, 9, 13, 17]:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)
        else:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0), -1)

    for connection in mp.solutions.hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        cv2.line(image, (landmark_list[start_idx][0], landmark_list[start_idx][1]), 
                 (landmark_list[end_idx][0], landmark_list[end_idx][1]), (0, 255, 0), 2)

    return image