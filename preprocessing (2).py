import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class preprocess_data:
    def visualize_images(self ,dir_path,nimages):
        fig,axs=plt.subplots(nrows=2, ncols=2, figsize = (10,10))
        dpath=dir_path
        count=0
        for i in os.listdir(dpath):
            #get th elist of images in a given class
            train_class=os.listdir(os.path.join(dpath,i))
            #plot the images
            for j in range(nimages):
                img=os.path.join(dpath,i,train_class[j])
                img=cv2.imread(img)
                axs[count][j].title.set_text(i)
                axs[count][j].imshow(img)
            count+=1
        fig.tight_layout()
        plt.show(block=True)

    #write method to preprocess the data
    def preprocess(self,dir_path):
        dpath=dir_path
        #count the number of images in teh dataset
        train=[]
        labels=[]
        for i in os.listdir(dpath):
            #get the list of images in a given class
            train_class=os.listdir(os.path.join(dpath,i))
            for j in train_class:
                img=os.path.join(dpath,i,j)
                train.append(img)
                labels.append(i)
        print("number of images:{}\n".format(len(train)))
        print("number of images labels:{}\n".format(len(labels)))
        food_df=pd.DataFrame({'Image': train,'Labels': labels})
        print(food_df)
        return food_df, train, labels

    def generate_train_test_images(self, retina_df, train, label):
        train, test = train_test_split(retina_df, test_size=0.2)
        print(test)
        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, validation_split=0.15)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_dataframe(
            train,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            subset='training')
        validation_generator = train_datagen.flow_from_dataframe(
            train,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            subset='validation')
        test_generator = test_datagen.flow_from_dataframe(
            test,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32)

        print("Train Shape : ", train.shape)
        print("Test Shape : ", test.shape)
        print("Length : ", len(train))
        return train_generator, test_generator, validation_generator
