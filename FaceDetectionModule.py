import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, minDetectionCon=0.75):
        self.minDetectionCon = minDetectionCon

        # import our mediapipe clases
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        # default is 0.5
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        # convert
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.faceDetection.process(imgRGB)
        # print(results)
        # extract the information
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fanciDraw(img, bbox)
                    # cv2.rectangle(img, bbox, (255,0,255), 2)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img, bboxs

    def fanciDraw(self, img, bbox, length=30, thickness=5, rectangleThickness=1):
        x, y, w, h  = bbox
        # diagonal point
        x1, y1 = x+w, y+h
        # drawing the lines
        cv2.rectangle(img, bbox, (255, 0, 255), rectangleThickness)
        # top left x,y corner
        cv2.line(img, (x,y), (x+length, y),(255,0,255), thickness)
        cv2.line(img, (x, y), (x , y+length), (255, 0, 255), thickness)
        # top right x1,y corner
        cv2.line(img, (x1, y), (x1 - length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + length), (255, 0, 255), thickness)
        # bottom left x,y1 corner
        cv2.line(img, (x, y1), (x + length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - length), (255, 0, 255), thickness)
        # bottom right x1,y1 corner
        cv2.line(img, (x1, y1), (x1 - length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - length), (255, 0, 255), thickness)

        return img
def main():
    cap = cv2.VideoCapture('Videos/1.mp4')
    pTime = 0
    # create a object for the class
    detector = FaceDetector()
    while True:
        # this give us our frame
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        # img, bboxs = detector.findFaces(img, False)
        # print(bboxs)

        # display the actual value fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)

        cv2.imshow('Image', img)
        # we can chose the sped 1 fast 10 slow
        cv2.waitKey(10)

if __name__ == '__main__':
    main()