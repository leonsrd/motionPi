from picamera.array import PiRGBArray
from imutils.io import TempFile
from picamera import PiCamera
import datetime
import dropbox
import imutils
import time
import cv2

client = dropbox.Dropbox("dropbox key")
print("[DROPBOX ACCOUNT LINKED SUCCESSFULLY]")
 
camera = PiCamera()
camera.resolution = ([640,480])
camera.framerate = 16
capture = PiRGBArray(camera, size=([640,480]))

print("[CAMERA SENSOR WARMING UP]")
time.sleep(3)

avg = None
motionCounter = 0

lastUploaded = datetime.datetime.now()

for f in camera.capture_continuous(capture, format="bgr", use_video_port=True):
    frame = f.array
    timestamp = datetime.datetime.now()
    detector = "nomotion"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if avg is None:
        avg = gray.copy().astype("float")
        capture.truncate(0)
            
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    
    thresh = cv2.threshold(frameDelta, 5, 255,
        cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cnts)
    
    for c in cont:
        if cv2.contourArea(c) > 5000:
            detector = "motion"
    
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.60, (0, 255, 255), 1)
    
    if detector == "motion":
        if (timestamp - lastUploaded).seconds >= 3:
            motionCounter += 1
            if motionCounter >= 8:
                TempImage = TempFile(ext=".jpg")
                cv2.imwrite(TempImage.path, frame)
                print("[MOTION DETECTED] {}".format(ts))
                path = "/{base_path}/{timestamp}.jpg".format(
                base_path="pi_sucam/img", timestamp=ts)
                client.files_upload(open(TempImage.path, "rb").read(), path)
                TempImage.cleanup()
                lastUploaded = timestamp
                motionCounter = 0
    
    cv2.imshow("live feed", frame)
    if cv2.waitKey(1) == ord("q"):
        break
    
    capture.truncate(0)