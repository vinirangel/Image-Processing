import numpy as np
import tensorflow as tf
import keras  
import os
import glob 
from skimage import io 
import skimage
import random 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import os
import shutil
import splitfolders

dataset_pth = r"C:\Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset"
output_pth = r"C:\Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset"

splitfolders.ratio(dataset_pth, output=output_pth, seed=1337, ratio=(.8, 0.2))

# dataset_pth = "C:/Users/Vinicius/Desktop/CS Files/PI/dataset"
# dataset_pth = "C:\Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/New Dataset/ASC-H"
# "directory = os.fsencode(dataset_pth)



# df = pd.read_csv("classifications.csv", delimiter=',')
# # print(df.head())

# id = []
# names = []
# coordinateX = []
# coordinateY = []
# classes = []
# index = 0
# # tf.keras.preprocessing.

# for idx, row in df.iterrows():
#     # if(index == 10):
#     id.append(row["cell_id"])
#     names.append(row["image_filename"])
#     classes.append(row['bethesda_system'])
#     coordinateX.append(row['nucleus_x'])
#     coordinateY.append(row['nucleus_y'])
#     # print(row["image_filename"])

# # for index in range(len(id)):
# #     # print(names[index], index, coordinateX[index], coordinateY[index])
# #     if ".png" in names[index]:
# #         filepath = "C:/Users/Vinicius/Desktop/CS Files/PI/dataset/" + names[index]
# #         # print(filepath)
# #         try:
# #             img = cv2.imread(filepath)
# #             # cv2.imshow(filepath, img)
# #             # cv2.waitKey(0)
# #             img = img[(coordinateY[index]-100):(coordinateY[index]-100) + 100, (coordinateX[index]-100):(coordinateX[index]-100) + 100]
# #             # tmp = "_" + (str)(id[index]) + ".png"
# #             # newfilename = names[index].replace(".png", tmp)
# #             # print(newfilename)
# #             newfilename = (str)(id[index]) + ".png"
# #             if classes[index] == "Negative for intraepithelial lesion":
# #                cv2.imwrite(os.path.join("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/New Dataset/Negative for intraepithelial lesion", newfilename), img) 
# #             elif classes[index] == "ASC-US":
# #                 cv2.imwrite(os.path.join("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/New Dataset/ASC-US", newfilename), img)
# #             elif classes[index] == "ASC-H":
# #                 cv2.imwrite(os.path.join("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/New Dataset/ASC-H", newfilename), img)
# #             elif classes[index] == "LSIL":
# #                 cv2.imwrite(os.path.join("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/New Dataset/LSIL", newfilename), img)
# #             elif classes[index] == "HSIL":
# #                 cv2.imwrite(os.path.join("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/New Dataset/HSIL", newfilename), img)
# #             elif classes[index] == "SCC":
# #                 cv2.imwrite(os.path.join("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/New Dataset/SCC", newfilename), img)

# #             # cv2.imwrite(os.path.join("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/New Dataset", newfilename), img)
# #         except:
# #             continue

# # for file in os.listdir(directory):

# validation_files = random.sample(glob.glob("C:\Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/New Dataset/ASC-H/*.png"), 15)

# for f in enumerate(validation_files, 1):
#     dest = os.path.join("C:\Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Preprocessed Dataset/Validation", str(f[0]))
#     if not os.path.exists(dest):
#         print(dest)
#         # os.makedirs(dest)
#     shutil.copy(f[1], dest)"