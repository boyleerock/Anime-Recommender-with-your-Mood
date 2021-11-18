import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

previous_time = 0
current_time = 0

path = "WIN_20180925_17_37_17_Pro_00025.png"
img = cv2.imread(path)

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results =hands.process(imgRGB)
# print(results.multi_hand_landmarks)
mpDraw = mp.solutions.drawing_utils

if results.multi_hand_landmarks:
    for handLandMark in results.multi_hand_landmarks:
        for id, lm in enumerate(handLandMark.landmark):
            # print(id, lm)
            height, width, channel = img.shape
            cx, cy = int(lm.x * width), int(lm.y * height) #position of the center
            print(id, cx, cy)
            # if id == 4:
            cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)

        mpDraw.draw_landmarks(img, handLandMark, mpHands.HAND_CONNECTIONS)

 # while True:
 #    success, img = cap.read()
 #    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 #    results =hands.process(imgRGB)
 #    # print(results.multi_hand_landmarks)
 #    mpDraw = mp.solutions.drawing_utils
 #
 #    if results.multi_hand_landmarks:
 #        for handLandMark in results.multi_hand_landmarks:
 #            for id, lm in enumerate(handLandMark.landmark):
 #                # print(id, lm)
 #                height, width, channel = img.shape
 #                cx, cy = int(lm.x * width), int(lm.y * height) #position of the center
 #                print(id, cx, cy)
 #                # if id == 4:
 #                cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)
 #
 #            mpDraw.draw_landmarks(img, handLandMark, mpHands.HAND_CONNECTIONS)
 #
 #    current_time = time.time()
 #    fps = 1/(current_time-previous_time)
 #    previous_time = current_time
 #
 #    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
 #                (255, 0, 255), 3)
 #
 #    cv2.imshow("Image", img)
 #    cv2.waitKey(1)