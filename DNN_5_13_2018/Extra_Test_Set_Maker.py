#
# Christopher Blanks 5/13/2018
#
# Purpose:
#
# This script has the function "extra_test_set_maker" that  accepts images represented in
# a csv data format and converts them into a two-entry tuple. This is no ordinary two
# entry tuple.
#
# The first entry needs to be a ndarray with all of the entries for a file,
# which each sub-entry is a 784-entry array
# of values for a 28 x 28 image. The values
# need to be mapped to be decimals between 0 and 1.
#
# The next entry needs to be a ndarray with a type int64 number that represents
# the label of the image.
#
# The other functions deal with unlabeled data. "unlabeled_data_formatter()" does this
# with the non-expedited process. The two auto_unlabel() are for the expedited process
# and take different arguments.
#


import matplotlib.pyplot as plt
import numpy as np


def extra_test_set_maker():
    """ Prompts the user to see if they want to use another data set for testing."""
    print('Hello User. When answering yes or no questions, please enter 1 for yes and 0 for no.')
    print('For other questions, please enter the desired paramater value, file name, or other entry.\n')
    key = int(input('Would you like to use an extra test set for this session?\n'))
    if key == 0:
        return []
    elif key == 1:
        pass
    else:
        raise ValueError
    file_name = str(input('Please, enter the CSV file name. \n'))

    f = open(file_name ,'r')
    line_store = f.readlines()
    f.close()

    extra_test_set = []

    for line in line_store:
        linebits = line.split(',')
        imarray = (np.asfarray(linebits[1:]).reshape((784,1)))
        imarray = np.divide(imarray,255)
        label = linebits[0]

        image_tuple = (imarray, np.int64(label))
        extra_test_set.append(image_tuple)
    return extra_test_set


def unlabeled_data_formatter():
    """ This function formats csv files into an unlabeled data format. Used for the non-expedited process. """
    unlabeled_file_name = str(input("\nPlease, enter the file name of the unlabeled data:\n>"))

    u = open(unlabeled_file_name,'r')
    line_store2 = u.readlines()
    u.close()

    unlabeled_data = []
    
    for line in line_store2:
        linebits2 = line.split(',')
        if linebits2[-1] == '':
            linebits2[-1] = '0.0'
        print(linebits2)
        print(len(linebits2))
        imarray2 = (np.asfarray(linebits2[1:]).reshape((784,1)))
        imarray2 = np.divide(imarray2,255)
        print(imarray2)

        image_tuple2 = (imarray2, np.int64(-1)) #Put in a placeholder value for the second element
        unlabeled_data.append(image_tuple2)
    return unlabeled_data

def auto_unlabel2(file):
    """ This function is when the user is loading a csv file (When the pic_proc() is used)"""
    u = open(file,'r')
    line_store2 = u.readlines()
    u.close()

    unlabeled_data = []
    
    for line in line_store2:
        linebits2 = line.split(',')
        if linebits2[-1] == '':
            linebits2[-1] = '0.0'
        print(linebits2)
        print(len(linebits2))
        imarray2 = (np.asfarray(linebits2[1:]).reshape((784,1)))
        imarray2 = np.divide(imarray2,255)
        print(imarray2)

        image_tuple2 = (imarray2, np.int64(-1)) #Put in a placeholder value for the second element
        unlabeled_data.append(image_tuple2)
    return unlabeled_data

def auto_unlabel(unlabeled_list):
    """ This function is for lists of 1-Dimension and 785 elemets. (When the pic_proc2() is used) """
    unlabeled_data = []
    imarray2 = (np.asfarray(unlabeled_list[1:]).reshape((784,1)))
    imarray2 = np.divide(imarray2,255)
    image_tuple2 = (imarray2, np.int64(-1)) #Put in a placeholder value for the label position
    unlabeled_data.append(image_tuple2)
    return unlabeled_data

    
