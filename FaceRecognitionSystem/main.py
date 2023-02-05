# We are using a library called face_recognition which helps us to detect, recognize, train and encode any 
# faces. 
# HOG representation --> detection of face_recognition
# Dlib --> Landmarks
# Open face--> previously trained neural network--> encoding 128 measurement
# library used 
# 1. Cmake 
# 2. Dlib 
# 3. face_recognition
# 4. Numpy
# 5.Open cv
# ---------------------------------------------------------------------------------------------------------------
# IMPORTING THE LIBRARY
# ---------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import face_recognition
# ---------------------------------------------------------------------------------------------------------------
# LOADING THE IMAGES 
# ---------------------------------------------------------------------------------------------------------------
# loading rajesh hamal image from training_rajesh_hamal folder
imgRajesh = face_recognition.load_image_file('training_rajesh_hamal/Rajesh Hamal.jpg')
#converting rajesh hamal image into rgb
imgRajesh = cv2.cvtColor(imgRajesh, cv2.COLOR_BGR2RGB)
# loading rajesh hamal image from training_rajesh_hamal folder for test image
imgTest = face_recognition.load_image_file('training_rajesh_hamal/Rajesh Hamal Test.jpg')
#converting rajesh hamal image into rgb for test image
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
# ---------------------------------------------------------------------------------------------------------------
# FIND FACE IN IMAGE AND ENCODE FACE 
# ---------------------------------------------------------------------------------------------------------------
#imapge ma face find garney
faceLoc = face_recognition.face_locations(imgRajesh)[0]
#found face lai encode garney
encodeRajesh = face_recognition.face_encodings(imgRajesh)[0]
#drawing circle around face [cv2.circle (image_name,(x1,y1),(x2,y2)) color]
cv2.rectangle(imgRajesh, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]),(0, 171, 255), 2)
# ---------------------------------------------------------------------------------------------------------------
# FIND FACE IN IMAGE AND ENCODE FACE FOR TEST IMAGE
# ---------------------------------------------------------------------------------------------------------------
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]),(0, 171, 255), 2)
# ---------------------------------------------------------------------------------------------------------------
# CALLING COMPARE_FACES FUNCTION TO COMPARE RAJESH AND RAJEST TEST AND SAVING IT INTO RESULT
# ---------------------------------------------------------------------------------------------------------------
results = face_recognition.compare_faces([encodeRajesh], encodeTest)
#finding face distance
faceDistance = face_recognition.face_distance([encodeRajesh], encodeTest)
#terminal ma result(true or false) and facedistance print garney (usually 0.44)
# print(results, faceDistance)
#image ma name print garna lai , facedistance ko code lai round of garney,font,scale,
# cv2.putText(imgTest, f'{results} {(faceDistance[0])}', (50, 50), cv2.FONT_ITALIC ,1, (0, 171, 255), 2)



# ---------------------------------------------------------------------------------------------------------------
# FINALLY DISPLAY TRAINED IMAGE
# ---------------------------------------------------------------------------------------------------------------
cv2.imshow('Picture Of Rajesh Hamal', imgRajesh)
cv2.imshow('Picture Of Rajesh Hamal For Testing Purpose', imgTest)


cv2.waitKey(0)