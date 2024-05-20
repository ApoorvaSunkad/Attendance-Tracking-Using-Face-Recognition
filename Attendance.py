#First, import all the neccessary libraries required.

import cv2
import numpy as np
import face_recognition
import face_recognition_models

"""
Loading image is an hectic task over here since we have to store of all them separately in a variable
hence we can create a list of images and then add here

we will ask our program to find the image folder then  find no. of images it has
and import them and find encodings for them.

"""

#to find image folder importing os
import os
from datetime import datetime

path = ''  # images file path
images = []

classNames = [] # to output the images list

myList = os.listdir(path) # we will fetch all images from this folder
print(myList) # output - ['Satya Nadella.jpg', 'Sudha Murthy.jpg', 'Sundar Pichai.jpg']

#using these names we will import the images one by one
for cl in myList: #looping through the list
    curImg = cv2.imread(f'{path}/{cl}') #importing each of classes
    images.append(curImg) # appending them in images folder
    classNames.append(os.path.splitext(cl)[0]) # using split text -> getting only the name instead of .jpg and appending them in classNames

print(classNames) #['Satya Nadella', 'Sudha Murthy', 'Sundar Pichai']


# --------------STEP 2 - ENCODING PROCESS of each image


#we will create a function that will compute all the encodings

def findEncodings(images):
    # create an empty list to store encodings
    encodeList = []
    for img in images:
        # first convert them into RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # next step is to find the encodings
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode) # appending in encode list
    return encodeList

#once the face is detected we will mark the attendance
def markAttendance(name):
    #we will just mark there name and time they have arrived at
    with open('../FaceRecognition_Project/AttendanceTracker.csv', 'r+') as f:
        # now we will read in all the lines that we have currently in our data
        # reason for this is if somebody has arrived we don't want to repeat it.
        #creating the list
        myDataList = f.readline()
        #empty list to put all names found
        nameList = []
        for line in myDataList:
            entry = line.split(',') # split by commas
            nameList.append(entry[0]) # first element of entry is name

        #once we have all the names in the nameList we will check if the name is present or not
        if name not in nameList:
            now = datetime.now() # gives the date and time
            datenTime = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datenTime}')


encodeListKnown = findEncodings(images) #Function call - to get encodings of the stored images
print('Encoding Complete') # to know that encoding is complete



# --------- STEP 3 -> Find the matches between our encodings
# But we don't have any images to match so those images will be coming from WEBCAM

# Initializing the webcam

cap = cv2.VideoCapture(0) # 0 as ID

while True: # to get each frame one by one
    success, img = cap.read()
    #because we are doing it in real time we want to reduce our image size this helps in speeding the process
    imgSmall = cv2.resize(img,(0,0),None, 0.25, 0.25)
    #convert to RGB
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)


    # In webcam we can find multiple faces to find that we can get location then we will send into these locations to our encodings
    facesinCurrFrame = face_recognition.face_locations(imgSmall)
    # finding encoding of webcam
    encodeinCurrFrame = face_recognition.face_encodings(imgSmall,facesinCurrFrame)


    #FINDING THE MATCHES
    #Iterating through all the faces found in the current frame and then we will compare with all the encodings that we have found

    for encodeFace, faceLoc in zip(encodeinCurrFrame,facesinCurrFrame):
        # Working
        # Why Zip? => because we want everything in same loop hence zip is used.
        # One by one it will fetch one face location from facesinCurrFrame list and then it will fetch the encodings from encodeinCurrFrame
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace) #finding face distance - this will give us a list of face distances
        # print(faceDist)

        matchIndex = np.argmin(faceDist) # to get the minimum index


        #once we get indexes now we know which person it is then we can show a box around the face and also display the name.
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)

            #to create a bounding box around the face
            y1,x2,y2,x1 = faceLoc #this faceLoc gives the location of face detected
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            # Whenever we find a match we will call the mark attendance function to mark attendance
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)