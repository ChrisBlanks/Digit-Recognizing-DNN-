#
# Christopher Blanks 5/13/2018
#
# Purpose:
#
# This script is responsible for the preprocessing part of the automated process.
# It attempts to binarize the image data, so that only grayscale values remain. In
# additon, it flips the pixels values, so that black will be 255 and white will be 0.
# There certainly is room for improvement (maybe not converting image into black and white
# and find another method to binarize the values).
#
#
# Notes:
# - The first pic_proc uses open CV to preprocess and writes the values into a csv format
#
# - The second pic_proc uses the built_in PIL to preprocess and return a list to be used
#   in another script
#
# - Couldn't get a decent version of opencv on the RasPi, so the pic_proc2 is the one in current use



from PIL import Image
#import cv2


def pic_proc(pic):
    """ OpenCV method of preprocessing """

    #pic_sel = str(input('What image file would you like to select?\n>'))
    pic_sel = pic
    try:
        im = cv2.imread(pic_sel)
    except:
        print('Did not work. Try again!')
        raise NameError
    
    new_size = cv2.resize(im, (28, 28)) #size for neural network
    gray_image = cv2.cvtColor(new_size, cv2.COLOR_BGR2GRAY)
    
    #threshold determines whether black or white for each pixel
    thresh = 127
    im_bw = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1] 
    cv2.imwrite('blackwhite.png', im_bw)
    cv2.imshow('image',im_bw)
    
    count = 0
    with open('output_file3.csv','w+') as f:
        f.write('-1,') #placeholder value
        for i in range(new_size.shape[0]):
            for j in range(new_size.shape[1]):
                count = count+1
                if count<784:
                    gray_val = 255 - float(im_bw[i][j]) #flips values in order to match the MNIST set
                    f.write('{},'.format(gray_val)) 


def pic_proc2(picture):
    """ PIL alternative to opencv method; Creates a list instead of a CSV file."""

    try:
        im = Image.open(picture).convert('1') #should be greyscale
    except:
        print('Did not work. Try again!')
        raise NameError
    im.save('before_thumbnail.png')
    
    im.thumbnail((28,28), Image.ANTIALIAS)
    im = im.resize((28,28)) #resizes the thumbnail to the correct value
    
    #im.show(im) #have to install imagemagick to display imagesave
    im.save('dnn_pic.png')
    
    pixels = list(im.getdata()) #gathers pixel data in an internal PIL data structure

    pixel_list = [-1] #placeholder value to replace the label

    for i in pixels:
        temp = 255 - i #Flips the binary values so that black will be 255 and white is 0
        pixel_list.append(temp)

    return pixel_list
    


# test execution when the file is executed
#pic_proc2('r3.jpg')

    
    
