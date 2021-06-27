from PIL import Image,ImageDraw, ImageFont 
import cv2
import time
import numpy as np 

from my_package.crawl import  cov_weather

def PIL_image(cov_list,img_rect):
    cov_chk = cov_list[-1]
    pil_im=Image.new("RGB",(1200,900),(0,0,0))  # img 생성
    draw = ImageDraw.Draw(pil_im)        
    draw.text((910,40),"Mask Detector",font=ImageFont.truetype('NanumSquareRoundB.ttf', 32), fill=(225, 225, 225))
    if cov_chk:  # cov_list[-1] : True 일 경우 ,4글자 [구름많음] False : 2글자 [맑음, 흐림]
    # 4글자 인덱스 순서대로 :  날씨 이모티콘, 온도 (17º), 구름많음 or 구름조금, 일일확진자 91명
        draw.text((514,585), cov_list[0], font=ImageFont.truetype('NotoEmoji-Regular.ttf', 60), fill=(255, 255, 255))
        draw.text((584,605), cov_list[1], font=ImageFont.truetype('NanumSquareRoundB.ttf', 50), fill=(255, 255, 255))
        draw.text((542,667), cov_list[2]  , font=ImageFont.truetype('NanumSquareRoundB.ttf', 24), fill=(255, 255, 255))
        draw.text((514,730), cov_list[3] , font=ImageFont.truetype('NanumSquareRoundB.ttf', 22), fill=(255, 255, 255))
    else:
        # 두글자 인덱스 순서대로 :  날씨 이모티콘, 맑음 17º, 어제보다...  , 일일확진자 91명
        draw.text((486,517), cov_list[0] , font=ImageFont.truetype('NotoEmoji-Regular.ttf', 55), fill=(255, 255, 255))
        draw.text((550,533), cov_list[1] , font=ImageFont.truetype('NanumSquareRoundB.ttf', 45), fill=(255, 255, 255))
        draw.text((527,595), cov_list[2] , font=ImageFont.truetype('NanumSquareRoundB.ttf', 19), fill=(255, 255, 255))
        draw.text((520,650), cov_list[3] , font=ImageFont.truetype('NanumSquareRoundB.ttf', 22), fill=(255, 255, 255))
    pil_frame = np.array(pil_im)
    
    #pil_frame = cv2.add(img_rect, img)
    pil_frame += img_rect
    return pil_frame

# ======================================   ouput image  ===================================
def outputFrame(progress,frame, pil_frame, info): 
    t1, time_print , am_pm, date = current_time()
    if progress == 0:

        frame = cv2.addWeighted(frame, 0.4, pil_frame, 0.7, -1) # 촬영되는 영상과 이미지를 합쳐 투명도 주는 부분
        
        pill_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pill_image)
        # 대기화면 왼쪽 위 날짜 출력 부분  ex) 2020-10-26
        draw.text((70,40),t1,font=ImageFont.truetype('NanumSquareRoundB.ttf', 37), fill=(225, 225, 225))
        
        # 대기화면 중간에서 시간을 나타내는 부분들  ex 11 : 35 (AM)
        draw.text((423,180),time_print, font=ImageFont.truetype('Nunito-Bold.ttf', 110), fill=(255, 255, 255))
        draw.text((786,245),am_pm, font=ImageFont.truetype('NanumSquareRoundB.ttf', 40), fill=(255, 255, 255))
        draw.text((522,313),date, font=ImageFont.truetype('NanumSquareRoundB.ttf', 33), fill=(255, 255, 255))
        out_frame = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)
        
    # ----- 마스크 인식 후 NFC 태그 progress ----- 
    elif progress == 1:
        
        pill_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pill_image)
        draw.text((430,680), info, font=ImageFont.truetype('NanumSquareRoundB.ttf', 35), fill=(255, 255, 255))
        out_frame = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)  # 이 부분을 사각형 보다 늦게 빼버리면 색깔 사각형이 안보이게됨
        out_frame = cv2.rectangle(out_frame,(0,0),(1200,900), (0,230,230), 3)  # 노락색 사각형 그리기
        # cv2.cvtColor() 한 후 cv2.rectangle() 

    # -----  마스크 착용 x ----- 
    elif progress == 2:
        pill_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pill_image)
        draw.text((490,680), info, font=ImageFont.truetype('NanumSquareRoundB.ttf', 35), fill=(255,255,255))
        out_frame = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)
        out_frame = cv2.rectangle(out_frame,(0,0),(1200,900), (0,0,240), 3)
        # cv2.cvtColor() 한 후 cv2.rectangle() 

    # -----  NFC 태그로 등록된 사용자 확인 progress ----- 
    elif progress == 3:
        pill_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pill_image) 
        draw.text((520,680), info, font=ImageFont.truetype('NanumSquareRoundB.ttf', 35), fill=(255, 255, 255))
        out_frame = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)
        out_frame = cv2.rectangle(out_frame,(0,0),(1200,900), (143,255,133), 3)
        # 위와 동일 cvtColor 뒤에 사각형 그리기  
    return out_frame

# ======================================   Text Crawling  ===================================
def crawlText():
    cov = cov_weather()
    cov_chk = True
    cov_Comma_list = cov.split(',')
    cov_Space_list = cov_Comma_list[0].split()
    #  ============ 2020 / 10 ==============
    #day_covid = cov_Space_list[0] +" "+ cov_Space_list[1]   # 일일확진자 98명
    # current_weather = cov_Comma_list[1].strip()
    # ============= 2021 / 06  ==============
    day_covid = cov_Comma_list[0]
    current_weather = cov_Comma_list[2]
    temp = cov_Comma_list[1][-3:]

    if "구름" in current_weather or "흐림" in current_weather:
        unicode_text = u"\u2601"
    elif "맑음" in current_weather:
        unicode_text = u"\u2600"
    elif "비" in current_weather:
        unicode_text = u"\u2614"
    #if len(cov_Comma_list[-1].strip()) !=4:   # 구름많음, 구름조금
    print("cov_Comma_list : ", cov_Comma_list , "  cov_Space_list : ", cov_Space_list)
    if len(current_weather) >=4:   # 구름많음, 구름조금
         #  ============ 2020 / 10 ==============
        # 4글자 ex) 구름많음 일 경우   이모티콘은 0번쨰 인덱스
        #temp = cov_Space_list[3][2:]    # 온도
        #current_weather = cov_Comma_list[-1].strip() # 구름많음
        print("temp : ", temp)
        print("current_weather : ", current_weather)
        return [unicode_text, temp, current_weather, day_covid,cov_chk]
    
    else:
        # 2글자 ex) 흐림, 맑음 일 경우
         #  ============ 2020 / 10 ==============
        #current_weather = cov_Comma_list[-1].strip()  + " "+ cov_Space_list[3][2:] # 구름많음  12º
        cov_chk = False
        current_weather = current_weather + " " + temp
        weather_summary = cov_Comma_list[-1][:-2]  # 어제보다 ~~도 낮음
        return [unicode_text, current_weather, weather_summary, day_covid,cov_chk]

# ========================  current time ========================
def current_time() :
    t1 = time.strftime('%Y-%m-%d')       # 2020-10-17 
    time_print = time.strftime("%I : %M")  # 12시간 단위 출력
    am_pm = time.strftime('(%p)')           # AM, PM
    date = time.strftime(
        '%m월 %d일'.encode('unicode-escape').decode()   # 한글로 format 해줄 시 생기는 오류를 해결하기 위한 부분
    ).encode().decode('unicode-escape')     # 10월 17일
    return t1, time_print , am_pm, date