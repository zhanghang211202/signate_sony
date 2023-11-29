import numpy as np
import os
import matplotlib.pyplot as plt
import csv

def data_cleaning(X_train, y_train):
    new_x_train = []
    new_y_train = []
    # calculate the number of pixels that are equal to 1 in each image
    # if the number of pixels==1 is less than 60, then add the image to the new_X_train
    # return the new_X_train
    # for i in range(X_train.shape[0]):
    #     if np.sum(X_train[i] >= 200) < 50:
    #         new_x_train.append(X_train[i])
    #         new_y_train.append(y_train[i])
    # new_x_train = np.array(new_x_train)
    # new_y_train = np.array(new_y_train)

    # save all the images in X_train in folder name all
    # for i in range(X_train.shape[0]):
    #     if not os.path.exists('data/seperate_data/test'):
    #         os.makedirs('data/seperate_data/test')
    #     plt.imsave('data/seperate_data/test/'+str(i)+'.png', X_train[i].reshape(20,20), cmap='gray')

    csv_file_path = 'labels.csv'
    # Writing index i and y_train[i] to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'y_train'])  # Writing header
        for i in range(X_train.shape[0]):
            writer.writerow([str(i), int(y_train[i])])
        # write str(i) and y_train[i] to a csv file



        # plt.imsave('data/seperate_data/test/'+str(i)+'.png', X_train[i].reshape(20,20), cmap='gray')

    # for each image in new_x_train, according to the label in new_y_train, save them in different folders
    # the folder name is the label
    # for i in range(new_x_train.shape[0]):
    #     if not os.path.exists('data/'+str(new_y_train[i])):
    #         os.makedirs('data/'+str(new_y_train[i]))
    #     plt.imsave('data/'+str(new_y_train[i])+'/'+str(i)+'.png', new_x_train[i].reshape(20,20), cmap='gray')

    return new_x_train, new_y_train

# read npy file in data/X_train.npy
X_train = np.load('data/X_train.npy')

# read npy file in data/y_train.npy
y_train = np.load('data/y_train.npy')

x_test = np.load('data/X_test.npy')

X_train, y_train = data_cleaning(x_test, y_train)

# draw all images in X_train using for loop
# every time draw 10 images
for j in range(0, x_test.shape[0], 10):
    plt.figure()
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i+j].reshape(20,20),cmap='gray')
        # show the label of y_train
        # plt.title(y_train[i+j])
    plt.show()

