# Attendance-Tracking-Using-Face-Recognition

As the name suggests in this project we will track the attendance
of the people by detecting the faces using a webcam and record the attendance live in an excel sheet. 

Examples of Face Recognition:

We have seen the Facebook and Google's Google photos detecting the faces of our friends or relatives, it automatically suggests us to confirm that is this the same person which is already present in it's feed?

Also when we open up our phone's camera to take a picture it detects the faces present in that whole frame
we might have seen a rectangluar frame appearing on our faces. These pretty much work on face recognition models.

This project using python as the main language and its libraries.

Python Libraries: cmake, dlib, numpy, open cv, face-recognition, face-recognition models, os,date and time, etc

Code Editor: Pycharm

The steps for face recognition:

1. Loading the images and converting them into RGB format.

2. we will train the model with encodings of normal image and then we will compare with
test image to see whether it can detect the image correctly

3. Comparing the faces and finding the encodings between them, we are getting 128 encodings of both the images which basically are of faces
we are using the machine learning classification algorithm which is LINEAR SVM Algorithm at backend to check whether both the images are equal or not.

After face recognition and encoding of the face step completes the WEBCAM is turned on then we can show up the images to match,
if it matches it shows the name of the person and also the attendance is tracked with time in an Excel sheet.


Reference Article: https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78