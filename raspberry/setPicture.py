import requests
import time
from picamera import PiCamera

num = 1
camera=PiCamera()
#카메라 화소 설정// 3280, 2464가 최대 화소
camera.resolution=(3280,2464)


while True:
    #db connect & get takePictureInfo
    #라즈베리가 카메라를 찍어야 되는지 찍지 말아야 되는지를 판별하기 위한 값을 서버에 요청.
    #서버로부터 반환값이 1이라면 사진을 찍어야 하고 0이라면 찍지 말아야한다.
    getInfo = requests.post('http://13.124.223.128/rasberryCommunicate/takePicture.php',{'setPicture':'get'}).text
    print(getInfo)

    #반환값이 1일 때 사진찍는 코드
    if getInfo=="1" :
        print("takePicture\n")

        #사진찍은 데이터를 저장할 때, 어느 방향을 찍은 사진인지 판별하기 위한 값을 서버에 요청.
        #반환 값은 0,90,180 중 하나.
        getRotation = requests.post('http://13.124.223.128/rasberryCommunicate/takePicture.php',{'setPicture':'rotation'}).text
        nowRotation = getRotation
        print("now rotation : "+getRotation)
        
        camera.start_preview()
        time.sleep(3)
	num=num+1

	    #찍힌 각도에 따라 이미지 파일 이름이 다르게 저장됨.
	    #팀장님 서있는 벽쪽 방향
        if getRotation == "0":
            camera.capture("result_1.jpg")
        #뒤쪽 책상을 찍는 방향
        elif getRotation == "90":
            camera.capture("result_2.jpg")
        #창가쪽 책상을 찍는 방향
        elif getRotation == "180":
            camera.capture("result_3.jpg")
        
            
        #camera.capture("arduinoTest"+nowRotation+"index"+str(num)+'.jpg')
        #camera.stop_preview()
        #num = num+1
        
        #서버에 업로드
        if getRotation == "0":
            img=open("result_1.jpg",'rb')
        elif getRotation == "90":
            img=open("result_2.jpg",'rb')
        elif getRotation == "180":
            img=open("result_3.jpg",'rb')
        
        #print(str(img))
        


        upload={'file':img}
        url=requests.post("http://13.124.223.128/getImage.php",files=upload)
        
        print("done!!\n")
        
        #db Update (taken picture)

        #모터가 다음 각도로 움직일 수 있도록 데이터베이스  db / camera / move 의 값을 0으로 변경
        takePicture = requests.post('http://13.124.223.128/rasberryCommunicate/takePicture.php',{'setPicture':'done'}).text
        print(takePicture)
    
    #after taking picture, moving motor
    else:
        if getInfo=="0" :
            print("already\n")
        
    
    time.sleep(1)
        





























































































