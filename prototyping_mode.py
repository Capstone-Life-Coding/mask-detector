from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import multiprocessing
from multiprocessing import Queue, Pool
from cam_utils import FPS, WebcamVideoStream
import cv2
import squd2
import numpy as np
import imutils
import time
import os
import playsound
from threading import Thread, Lock
from PIL import Image,ImageDraw, ImageFont 

                    
                    
#-------------------------------------------------------------------------#             

def detect_and_predict_mask(frame, faceNet, maskNet):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))


	faceNet.setInput(blob)
	detections = faceNet.forward()
	


	faces = []
	locs = []
	preds = []


	for i in range(0, detections.shape[2]):


		confidence = detections[0, 0, i, 2]


		if confidence > 0.5:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")


			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

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

def codelay() : 
    global i    
    global progress
    lock = Lock() 
    while True :
        time.sleep(0.15)
        if progress == 2 : 
                    lock.acquire()
                    playsound.playsound('maskvoice.wav')
                    time.sleep(3)
                    progress = 0 
                    i = 1
                    lock.release()
        elif progress == 3 :
                    lock.acquire()
                    time.sleep(3)
                    progress = 0 
                    i = 1
                    lock.release()
                    
                    
#-------------------------------------------------------------------------#                   

if __name__ == '__main__':
    
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("mask_detector.model")
       
   
   
    i = 1   
    progress = 0
    flag = False
    flag2 = False
    
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=10)
    output_q = Queue(maxsize=10)
    pool = Pool(5,squd2.worker, (input_q,output_q))

    video_capture =  WebcamVideoStream(0 ,1200,1200).start()
    

    fps = FPS().start()
    thread1 = Thread(target=codelay, args=())   
    # t1.start()
    # t1.join()
    
    
    print("[INFO] starting video stream...")

    while True:
        
        info2 ="" 
        info =""
        frame = video_capture.read()        
        frame = imutils.resize(frame, width=1200,)
        
        t1 = time.strftime('%Y-%m-%d')        
        t2 = time.strftime('%H:%M:%S') 
        t3 = "info"
        cv2.putText(frame,t2,(550,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255, 255),1) 
        cv2.putText(frame,t1,(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255, 255),1) 
        cv2.putText(frame,t3,(1050,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255, 255),1)  

        
        
        if progress == 2 :            
            info2 ="Please where a Mask"
            cv2.putText(frame, info2, (350,730),
                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
            pill_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pill_image)
            draw.text((550,100), "임한규", font=ImageFont.truetype('malgun.ttf', 34), fill=(255, 255, 255))
            frame = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)
    
            
        elif progress == 3 :        
            info2 ="Success" 
            cv2.putText(frame, info2, (450,730),
                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)                    
    
        if i == 1 :
            info = "Confirmed person : 00  Weather : 00"
            cv2.putText(frame, info, (300,800),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            
        if progress == 0 :
            
            for (box, pred) in zip(locs, preds):
    
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred           
                cv2.rectangle(frame, (startX, startY), (endX, endY), (183, 100, 100), 1)
                i = 0
                if 1 == 0 :
                    info =" "
                    cv2.putText(frame, info, (500,800),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1) 
                

                if mask > withoutMask:
                    color = (0, 153, 0)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
                    label = 'Mask' 
                    progress = 3

 
                    
                    
                                           
                else:                    
                    color = (0, 0, 255)
                    label = 'No Mask' 
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)                   
                    progress = 2 
                    if flag2 == False :
                        flag2 = True
                        thread1.start()

                                 
        
        
        
        input_q.put(frame)
        cv2.imshow("mask classification prototyping model",output_q.get())
        fps.update()
        
        
        
        
        if cv2.waitKey(1) == 27 : #ESC
            break



    fps.stop()
    pool.terminate()
    #input_q.join()
    #output_q.join()
    video_capture.stop()
    cv2.destroyAllWindows()
