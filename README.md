# mask-detector
대학교 캡스톤 디자인 경진대회 - 마스크 검출기 - Object Detection, dlib_CNN

##. 프로젝트 개요
---

- 프로젝트명 : 마스크 착용 유/무 판별장치
- 팀명 : 생활코딩 | 팀원 : 김대현, 송용호, 이명훈, 임한규
- 프로젝트 기간 : 2020-09-01 ~ 2020-10-16
- 프로젝트 상세 : 사회맞춤형 캡스톤디자인 경진대회

</br></br>

## 2. 프로젝트 목표 및 요건
---
프로젝트 진행 당시 COVID-19의 재확산으로 마스크 착용이 중요한 사회 이슈로 떠오름
마스크를 착용하지 않는 경우 건물에 못들어가는 사회 지침에 따라 
건물에 들어가기 전 마스크 착용을 검출해 마스크를 착용 했는지 검사하는 인적 소요를 줄이고 
사람들로 하여금 마스크 착용을 일상생활과 같이 여기도록 프로젝트 주제를 선정함

1. 사람의 얼굴 인식
2. 마스크 착용 유/무를 검출
3. 착용한 경우 NFC 카드 요청, 데이터베이스에 등록된 경우 등록된 사람의 이름 출력

</br>

## 3. Project WorkFlow
---  

[이미지 넣기]

</br>


## 4. 순서도
[마스크 착용관련 Trhead 1 ]

사람을 식별한 후 마스크를 착용했는지 확인하는 thread

[마스크 착용관련 Trhead 2 ]

마스크를 착용하지 않은 경우 마스크를 착용해주세요 음성 출력,
착용한 경우 NFC 인식 스레드 실행

</br>

## 5. 프로젝트 환경구성
---

- 주요 환경 구성
  - 라즈베리 파이 </br> 카메라모듈, RFID 리더기</br>
  - OpenCV </br> opencv-python</br>
  - MaridDB
    


작성중...
