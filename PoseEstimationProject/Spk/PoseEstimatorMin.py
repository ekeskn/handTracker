import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

#cap = cv2.VideoCapture('../PoseVideos/4.mp4')
#FOR WEBCAM:
cap = cv2.VideoCapture(0)
pTime = 0
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)




    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    print(int(fps))
    #-------------------SCALE---------------------------#
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    imS = cv2.resize(img, dim)

    #-----------------SHOW VIDEO-----------#v

    cv2.putText(imS, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image",imS)
    cv2.waitKey(10)