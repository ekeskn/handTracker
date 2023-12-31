import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('../PoseVideos/6.mp4')
# FOR WEBCAM:
#cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[10])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #print(int(fps))
    # -------------------SCALE---------------------------#
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    imS = cv2.resize(img, dim)

    # -----------------SHOW VIDEO-----------#v

    cv2.putText(imS, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", imS)
    cv2.waitKey(10)