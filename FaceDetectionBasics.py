import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture('Videos/1.mp4')
pTime = 0

# import our mediapipe clases
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
# default is 0.5
faceDetection = mpFaceDetection.FaceDetection(0.75)



while True:
    # this give us our frame
    success, img = cap.read()

    # convert
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    # print(results)
    # extract the information
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # display the actual value fps
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,
                3, (0,255,0), 2)

    cv2.imshow('Image', img)
    # we can chose the sped 1 fast 10 slow
    cv2.waitKey(10)