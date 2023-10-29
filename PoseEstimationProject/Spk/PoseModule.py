import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self,mode = False, upBody= False, model = 1, smooth = True, enable_seg = False,
                 smooth_seg=True, minDetect = 0.5, minTrack = 0.5):
        self.mode=mode
        self.upBdoy = upBody
        self.model = model
        self.smooth= smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.minDetect = minDetect
        self.minTrack = minTrack

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.upBdoy,self.smooth,
                                     self.enable_seg,self.smooth_seg,
                                     self.minDetect,self.minTrack)

    def findPose(self,img, draw=True):


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList




def main():
    cap = cv2.VideoCapture('../PoseVideos/4.mp4')
    # FOR WEBCAM:
    #cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[10])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        print(int(fps))
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

if __name__== "__main__":
    main()