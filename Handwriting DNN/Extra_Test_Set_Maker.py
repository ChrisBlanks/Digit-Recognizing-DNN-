#
# Christopher Blanks 3/24/2018
#
# Purpose:
#
# This script needs to accept images represented in some data format
# and convert them to a two-entry tuple. This is no ordinary two
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
# Note: Should add a function that asks the user for a file name to retrieve a CSV from
# and adjusts ndarray shape depending on data set size
#
import matplotlib.pyplot as plt
import numpy as np


def extra_test_set_maker():
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

# Purpose:
#
# Splits each line in CSV document and places the contents into individual arrays.
# Splits the image data from the label

    for line in line_store:
        linebits = line.split(',')
        imarray = (np.asfarray(linebits[1:]).reshape((784,1)))
        imarray = np.divide(imarray,255)
        label = linebits[0]

        image_tuple = (imarray, np.int64(label))
        extra_test_set.append(image_tuple)
    return extra_test_set



