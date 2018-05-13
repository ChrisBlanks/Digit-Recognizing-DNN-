#
# Christopher Blanks 5/13/2018
#
# Purpose:
#
# This script automates the whole process of gathering image data,
# preprocessing the data to mimic the MNIST dataset, passing the
# the data into the DNN, and outputing the guess of the DNN. The
# expedited_net function speeds up the process (approximately 20 seconds).
#
#
# IMPORTANT:
#
# Before running, you will have to:
#
# - In the load_data() in the mnist_loader script, you will have to change the
#   line ' f = gzip.open(/my/file/path,'rb') ' to your own absolute file path
#   to the pickled mnist data set file
#
# - Verify that your directory is setup correctly, so that you can access
#   each test picture, the '97_percent_Accuracy' network, the 'pixel_data' csv
#   file, the mnist pickled file, and the extra data sets
#
# - Verify that the pictures taken by the PiCamera end up in the directory and
#   that any generated csv file is made in the directory as well
#
# Notes:
#
# - To run this in the command prompt on the RasPi3, enter the following:
#
#   cd /home/Handwriting_DNN
#   sudo python3.5 /home/Handwriting_DNN/Demonstration.py
# 
# - The if statement is probably redundant 
# 


from Preprocessing import pic_proc2
from picamera import PiCamera
from Hand_Alg_2 import expedited_net

import time


camera = PiCamera() # Sets the PiCamera Library to the camera variable
camera.resolution = (640, 480)
camera.framerate = 30


user_start = int(input('Press 1 to take a picture and start the program.\n'))

while user_start == True:
    camera.start_preview(fullscreen=False, window = (100,20,640,480))
    time.sleep(3) # Allows Camera to Power Up
    camera.capture("New_pic.jpg")
    camera.stop_preview()
    
    pic = "New_pic.jpg"
    list_temp = pic_proc2(pic)
    expedited_net(list_temp)
    user_start = int(input('Press 0 to stop. Press 1 to continue.\n'))
    if user_start == 0:
        break

camera.close()

