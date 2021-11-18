import cv2
import time
import HandTrackingModule as htm

wCam, hCam = 800, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector(detect_Confidence=0.7)

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        # if index_finder_mcp  and thumbs_tip is higher than wrist
        if lmList[0][2] > lmList[4][2] and lmList[5][2] > lmList[4][2]:
            print('thumbs up')
            cv2.putText(img, 'Like this Anime!', (45, 375), cv2.FONT_HERSHEY_PLAIN
                        , 3, (255, 255, 255), 6)
        elif lmList[0][2] <= lmList[4][2]:
            print('thumbs down')
            cv2.putText(img, 'Dislike:(', (45, 375), cv2.FONT_HERSHEY_PLAIN
                        , 6, (255, 255, 255), 8)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

