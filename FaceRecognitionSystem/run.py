import cv2
import numpy as np
import face_recognition
#os is used to process different file in a system
import os
from datetime import datetime

#find the image from this folder and find encoding
path = 'student_pictures' 
#import all images from student_pictures
images = []
#import all images name from student_pictures
classNames = []
my_List = os.listdir(path)
# printing images name in terminal
# print(my_List)

for cl in my_List:   #class in mylist  #cl is image name
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) #removing .jpg from image name
# print(classNames)


# def is defining a function
#looping through all images and find encodings
def find_Encodings(images):
    encodeList = []
    for img in images: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList



def mark_Attendance(name):
    with open('csv_of_student_attendance/main.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%HH:%MM:%SS')
            f.writelines(f'{name},{dtString}\n'
                        )


encodeListKnown = find_Encodings(images)

print('Launching Camera for Attendance...')

#capture image from the webcam
cap = cv2.VideoCapture(0)

#capturing each frame
while True:
    success, img = cap.read()
    #resizing image since we are capturing image in real time
    # scale 0.25 and 0.25 
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
#zip combines content ,,, encodecurframe raw facecurframe sangai loop garney
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        #numpy minimum distance match
        matchIndex = np.argmin(faceDistance)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            mark_Attendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

