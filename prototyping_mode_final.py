from numba import jit

import cv2
import numpy as np
import time

from cam_utils import FPS, WebcamVideoStream

# ======================== output_image =======================
from output_image import * 

# ======================== processing, threading ========================

from threading import Thread, Lock
from multiprocessing import Queue, Pool, Process

# ============================ tensorflow  ============================
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ============================ 라즈베리 파이를 위한 라이브러리 ============================
#from pn532pi import Pn532, pn532
#from pn532pi import Pn532Spi
import pymysql
import binascii

#from picamera.array import PiRGBArray
#from picamera import PiCamera

# ============================ Audio ============================
import pygame
import playsound

#-------------------------------------------------------------------------#
progress = 0
process_STATUS = 0
detect_STATUS = 0
found_STATUS = 1
sound_STATUS = 0
change_STATUS = None
name = None
noMaskStack = 0

detect_Count = 0
rkfh = 1200
tpfh = 900

frame = None
prototxtPath = r"models\deploy.prototxt"
weightsPath = r"models\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("start")
maskNet = load_model(r"models\mask_detector.model")
print("finish")
#PN532_SPI = Pn532Spi(Pn532Spi.SS0_GPIO8)
#nfc = Pn532(PN532_SPI)

#pygame.mixer.init()
#pygame.mixer.music.load("sound/maskvoice.wav")
#------------------------------------------------------------------#

@jit(nopython=False, cache=True)
def please(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

@jit
def detect_and_predict_mask():
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []
    HeadMax = 30000
    isHead = 0


    for i in range(0, detections.shape[2]):


        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            isHead = (endX-startX) * (endY-startY)
            print(isHead)
            if isHead > HeadMax :
                HeadMax = isHead
                
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:

        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

def soundPlay() :
    global sound_STATUS
    #pygame.mixer.music.play()
    playsound.playsound('sound/maskvoice.wav')
    time.sleep(10)
    sound_STATUS = 0

def codelay() : 
    global noMaskStack
    global progress
    global process_STATUS
    global found_STATUS
    global name
    print("[INFO] called codelay")
    lock = Lock() 
    if progress == 2 :
        print("[INFO] your progress == 2")   
        if noMaskStack <= 0 :
            noMaskStack = 5 
        while noMaskStack >= 0 :
            time.sleep(1)
            noMaskStack -=  1
        if progress == 2 :
            progress = 0
        process_STATUS = 0
        print("[INFO] your progress == 0")
                
    elif progress == 1 :
        print("[INFO] your progress == 1")
        # NFC 사용시 open
        #found_Loop()  # nfc 부분
        if progress == 1 :
            # NFC 사용시 close
            time.sleep(3.7)
            progress = 0
            
            
        process_STATUS = 0
                
    elif progress == 3 :
        print("[INFO] your progress == 3")
        detect_STATUS = 1
        time.sleep(7)
        progress = 0
        process_STATUS = 0
        name = None
        time.sleep(7)
        detect_STATUS = 0


def detect():
    global noMaskStack
    global detect_STATUS
    global progress
    global detect_Count
    if progress == 3 :
        time.sleep(7)
        detect_STATUS = 0
    else :
        try :
            (locs, preds) = detect_and_predict_mask()
            
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred           
                cv2.rectangle(frame, (startX, startY), (endX, endY), (183, 100, 100), 1)
        
                if mask > withoutMask:
                    if not progress == 3 :
                        progress = 1
                else:                    
                    if not progress == 3 :
                        progress = 2
                        noMaskStack = noMaskStack + 1
            
        finally:
            time.sleep(1)
            detect_STATUS = 0
#=============================== NFC (라즈베리 파이) =====================================
def RecordSQL(UID):
    global name
    try:
        conn = pymysql.connect(host="localhost",
                       user="myroot",
                       passwd="1234",
                       db="Project")
        cursor = conn.cursor()
        sql = "SELECT id, name from staff where rfid = '"+ UID + "';"
        cursor.execute(sql)
        
        id, name = cursor.fetchone()
        print(id, name)
        sql = "insert into record (staff_id) values ('"+str(id)+"');"
        cursor.execute(sql)
        conn.commit()
    
    finally:
        cursor.close()
        conn.close()

def NFC_setup():
    nfc.begin()

    versiondata = nfc.getFirmwareVersion()
    if (not versiondata):
        print("Didn't find PN53x board")
        raise RuntimeError("Didn't find PN53x board")  # halt

    #  Got ok data, print it out!
    print("Found chip PN5 {:#x} Firmware ver. {:d}.{:d}".format((versiondata >> 24) & 0xFF, (versiondata >> 16) & 0xFF,
                                                                (versiondata >> 8) & 0xFF))

    #  configure board to read RFID tags
    nfc.SAMConfig()

    print("Waiting for an ISO14443A Card ...")
    
    

def loop():
    #  Wait for an ISO14443A type cards (Mifare, etc.).  When one is found
    #  'uid' will be populated with the UID, and uidLength will indicate
    #  if the uid is 4 bytes (Mifare Classic) or 7 bytes (Mifare Ultralight)
    global found_STATUS
    global progress
    success, uid = nfc.readPassiveTargetID(pn532.PN532_MIFARE_ISO14443A_106KBPS)

    if (success):
        print("binascii : " , binascii.hexlify(uid))
        UID = binascii.hexlify(uid)
        UID = UID.decode('utf8')
        RecordSQL(UID)
        found_STATUS = 0
        progress = 3
        return True;
    elif found_STATUS == 0 :
        print("asdf")
        return True;
    else:
        print("...")
        found_STATUS = found_STATUS + 1
        return False

def found_Loop() :
    global found_STATUS
    global detect_Count
    print("start")
    found_STATUS = 1
    found = loop()
    while not found :
            found = loop()
            if found_STATUS >= 11:
                found = True
            if not progress == 1:
                found = True
    detect_Count = 0
            
#==========================  main ==============================
if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)

    cov_list = crawlText()  # 크롤링 정보 가지고 오기 
    img_rect= cv2.imread('background/rect.png')
    img_rect = cv2.resize(img_rect,(rkfh,tpfh), interpolation=cv2.INTER_CUBIC)
    pil_frame = PIL_image(cov_list, img_rect)
    #fps = FPS().start()

    # t1.join()
    
    # nfc NFC_setup
    #NFC_setup()

    print("[INFO] starting video stream...")
    info = None

    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame,(rkfh, tpfh), interpolation=cv2.INTER_CUBIC)

        if process_STATUS == 0 :

            if progress == 0 :
                if detect_STATUS == 0 :
                    detect_STATUS = 1
                    thread3 = Thread(target=detect, args=())
                    thread3.start()

                if not progress == change_STATUS :
                    change_STATUS = progress
            
            else :
                if( process_STATUS == 0):
                    process_STATUS = 1
                    thread1 = Thread(target=codelay, args=())  
                    thread1.start()

        if progress == 1:
            if detect_STATUS == 0 :
                detect_STATUS = 1
                thread3 = Thread(target=detect, args=())
                thread3.start()
            if not progress == change_STATUS :
                change_STATUS = progress
                info ="nfc 태그를 인식해 주세요"
                
        elif progress == 2:
            if detect_STATUS == 0 :
                detect_STATUS = 1
                thread3 = Thread(target=detect, args=())
                thread3.start()
            if sound_STATUS == 0 :
                sound_STATUS = 1
                thread4 = Thread(target=soundPlay, args=())
                thread4.start()
            if not progress == change_STATUS :
                change_STATUS = progress
                info ="마스크를 착용해주세요"
        elif progress == 3:
            if not progress == change_STATUS : 
                change_STATUS = progress
                if name == "직원 아님" :
                    info = "안녕하세요"
                else : 
                    info = name + "님 안녕하세요"

        # 화면 출력
        out_frame = outputFrame(progress, frame, pil_frame, info)

        cv2.imshow("mask classification prototyping model",out_frame)

        #fps.update()
        
        if cv2.waitKey(1) == 27 : #ESC
            break        




    #fps.stop()
    #pool.terminate()
    #input_q.join()
    #output_q.join()
    #video_capture.stop()
    video_capture.release()
    cv2.destroyAllWindows()
