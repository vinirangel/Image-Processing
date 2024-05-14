import io
import PySimpleGUI as psg
from PIL import Image
import shutil
import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import numpy as np
# from skimage import io, feature
import skimage
import glob
import cv2
import os
import scipy
import xgboost as xgb
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing

# dataset = np.loadtxt('classifications.csv', delimiter=",")
# # split data into X and y
# X = dataset[:,0:8]
# Y = dataset[:,8]
# X.view

# # split data into train and test sets
# seed = 7
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

main_layout = [[psg.Button("Carregar Imagem", key="-LOAD-"), psg.VerticalSeparator(pad=10, color="gray"), psg.Button("Classificação Binaria com XGBoost", key="-XGBOOST_BINARY-")], 
         [psg.Button("Visualizar Imagem", key="-VIEW-"), psg.VerticalSeparator(pad=8, color="gray"), psg.Button("Classificação XGBoost com 6 Classes", key="-XGBOOST_TRAINING-")],
         [psg.Button("Tons de Cinza", key="-GRAYSCALE-"), psg.VerticalSeparator(pad=8, color="gray"), psg.Button("Prever com o Classificador XGBoost", key="-XGBOOST_PREDICT-")],
         [psg.Button("Histograma Tons de Cinza", key="-GRAYSCALE_HISTOGRAM-")],
         [psg.Button("Histograma HSV", key="-HSV_HISTOGRAM-")],
         [psg.Button("Matriz de Co-ocorrencia", key="-COMATRIX-")],
         [psg.Button("Momentos Invariantes de Hu", key="-HU_MOMENTS-")],
]

window = psg.Window('Processamento de Imagens', main_layout, size=(715,250))

#  ADICIONAR POPUP CASO FILE ESTEJA VAZIO

file = "EMPTY"
# file_path = "/home/vinicius/Desktop/PI/sample.jpg"
# file_path = "C:/Users/Vinicius/Desktop/CS Files/PI/sample.jpg"
file_path = "C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Validation/ASC-H/816.png"
train_predictions = ""
val_predictions = ""
train_images = ""
test_images = ""
model = ""

# train_images = []
# train_labels = []

# for directory_path in glob.glob("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Training/*"):
#    label = directory_path.split("\\")[-1]
#    # print(label)
#    for img_path in glob.glob(os.path.join(directory_path, "*png")):
#       print(img_path)
#       img = cv2.imread(img_path)
#       img = cv2.resize(img, (100,100))
#       # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#       train_images.append(img)
#       train_labels.append(label)

# train_images = np.array(train_images)
# train_labels = np.array(train_labels)

# test_images = []
# test_labels = []

# for directory_path in glob.glob("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Validation/*"):
#    label = directory_path.split("\\")[-1]
#    # print(label)
#    for img_path in glob.glob(os.path.join(directory_path, "*png")):
#       print(img_path)
#       img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#       img = cv2.resize(img, (100,100))
#       # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#       test_images.append(img)
#       test_labels.append(label)

# test_images = np.array(test_images)
# test_labels = np.array(test_labels)

# #Encode labels from test to integers
# le = preprocessing.LabelEncoder()
# le.fit(test_labels)
# test_labels_encoded = le.transform(test_labels)
# le.fit(train_labels)
# train_labels_encoded = le.transform(train_labels)

# base_model = tf.keras.applications.VGG16(include_top=False,
#                                                weights='imagenet')
# base_model.trainable = False
# base_model.summary()

# train_features = base_model.predict(train_images)
# train_features = train_features.reshape(train_features.shape[0], -1)
# val_features = base_model.predict(test_images)
# val_features = val_features.reshape(val_features.shape[0], -1)

# model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=120)

# # eval_set = [(train_features, train_labels_encoded), (val_features, test_labels_encoded)]
# eval_set = [(train_features, train_labels_encoded), (val_features, test_labels_encoded)]

# model.fit(
#     train_features, 
#     train_labels_encoded,  
#     eval_metric=["merror"], 
#     eval_set=eval_set, 
#     verbose=True
# )

# results = model.evals_result()

# train_error = results['validation_0']['merror']
# train_acc = [1.0 - i for i in train_error]

# val_error = results['validation_1']['merror']
# val_acc = [1.0 - i for i in val_error]

# # best_ntree_limit = model.best_ntree_limit()
# # print('Best ntree limit: ', best_ntree_limit)

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(train_acc, label='Train Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([0.3,1.1])
# # plt.axvline(best_ntree_limit-1, color="gray", label="Optimal tree number")
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(train_error, label='Training Loss')
# plt.plot(val_error, label='Validation Loss')
# plt.legend()
# plt.title('Training and Validation Loss')
# # plt.axvline(best_ntree_limit-1, color="gray", label="Optimal tree number")
# plt.ylabel('merror')
# plt.ylabel('Number of Trees')

# plt.show()

# train_predictions = model.predict(train_features)
# val_predictions = model.predict(val_features)

# print ("Training Accuracy = ", accuracy_score(train_labels_encoded, train_predictions))
# print ("Validation Accuracy = ", accuracy_score(test_labels_encoded, val_predictions))

while True:
   event, values = window.read()
   print(event, values)

   if event == "-LOAD-":
      file_path = psg.popup_get_file('Selecione uma Imagem',  title="Selecionar Imagem")
      file = Image.open(file_path)
      print ("File selected", file)

   if event == "-VIEW-":
      # file_path=psg.popup_get_file('Select a file',  title="File selector")
      print("File Path: ", file_path)
      file = Image.open(file_path)
      # print ("File selected", file)
      # file = Image.open("/home/vinicius/Desktop/PI/dataset/0a2a5a681410054941cc56f51eb8fbda.png")
      png_bio = io.BytesIO()
      file.save(png_bio, format="PNG")
      png_data = png_bio.getvalue()
      window2 = psg.Window("Visualizar Imagem", [[psg.Image(png_data, subsample=2, key="-IMAGE-")],
                                                 [psg.Button("Zoom In", key="-ZOOM IN-")]
                                                 ], size=(720,480), finalize=True)
      event2, values2 = window2.read()

      if event2 == "-ZOOM IN-":
         print("ZOOM ZOOM ZOOM")
         window2['-IMAGE-'].update(png_data)

      if event2 == psg.WIN_CLOSED:
         print("CLOSE")
         window2.close()

   if event == "-GRAYSCALE-":
      # file_path=psg.popup_get_file('Select a file',  title="File selector")
      gray_scale_img = "gray_scale.png"
      shutil.copy(file_path, gray_scale_img)
      file = Image.open(gray_scale_img).convert('L')
      file.save(gray_scale_img, "PNG")
      png_bio = io.BytesIO()
      file.save(png_bio, format="PNG")
      png_data = png_bio.getvalue()
      window3 = psg.Window("Tons de Cinza", [[psg.Image(png_data, subsample=2, key="-IMAGE-")],
                                             [psg.Button("Zoom In", key="-ZOOM IN-")]
                                             ], size=(720,480), finalize=True)
      
      event3, values3 = window3.read()

      if event3 == "-ZOOM IN-":
         print("ZOOM ZOOM ZOOM")
         window3['-IMAGE-'].update(png_data)

      if event3 == psg.WIN_CLOSED:
         print("CLOSE")
         window3.close()

   if event == "-GRAYSCALE_HISTOGRAM-": 
      img = cv2.imread(file_path,0) 
      # calculate frequency of pixels in range 0-255 
      histogram = cv2.calcHist([img],[0],None,[256],[0,256]) 
      plt.plot(histogram)
      plt.show() 

   if event == "-HSV_HISTOGRAM-":

      # img = cv2.imread(file_path,0) 
      # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      # h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
      # hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
      # #hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
      # #hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
      # plt.plot(hist_h, color='r', label="hue")
 
      img = cv2.imread(file_path)
      assert img is not None, "file could not be read"
      hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
      hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
      plt.imshow(hist,interpolation = 'nearest')
      plt.show()

   if event == "-COMATRIX-":
      gray_scale_img = "gray_scale.png"
      shutil.copy(file_path, gray_scale_img)
      file = Image.open(gray_scale_img).convert('L', colors=16)
      file.save(gray_scale_img, "PNG")
      img_array = np.asarray(file)
      # print("IMAGE ARRAY \n", img_array.shape)
      
      #Calcula as matrizes de Co Ocorrencia
      #C1,1
      glcm1 = skimage.feature.graycomatrix(img_array, distances=[1,1], angles=[3*np.pi/2], symmetric=False, normed=False)
      print("C1,1 \n", glcm1[0:5, 0:5])
      #C2,2
      glcm2 = skimage.feature.graycomatrix(img_array, distances=[2,2], angles=[3*np.pi/2], symmetric=False, normed=False)      
      # print("C2,2 \n", glcm2[0:5, 0:5])
      #C4,4
      glcm4 = skimage.feature.graycomatrix(img_array, distances=[4,4], angles=[3*np.pi/2], symmetric=False, normed=False)
      # print("C4,4 \n", glcm4[0:5, 0:5])
      #C8,8
      glcm8 = skimage.feature.graycomatrix(img_array, distances=[8,8], angles=[3*np.pi/2], symmetric=False, normed=False)
      #C16,16
      glcm16 = skimage.feature.graycomatrix(img_array, distances=[16,16], angles=[3*np.pi/2], symmetric=False, normed=False)
      #C32,32
      glcm32 = skimage.feature.graycomatrix(img_array, distances=[32,32], angles=[3*np.pi/2], symmetric=False, normed=False)

      #Calcular a Entropia 
      entropy1 = skimage.measure.shannon_entropy(glcm1)
      print("Entropy 1,1 ",entropy1)
      entropy2 = skimage.measure.shannon_entropy(glcm2)
      # print("Entropy 2,2 ",entropy2)
      entropy4 = skimage.measure.shannon_entropy(glcm4)
      entropy8 = skimage.measure.shannon_entropy(glcm8)
      entropy16 = skimage.measure.shannon_entropy(glcm16)
      entropy32 = skimage.measure.shannon_entropy(glcm32)

      #Calcular a Homogeneidade
      homogeneity1 = skimage.feature.graycoprops(glcm1, 'homogeneity')
      print("HOMOGENEITY1,1 ", homogeneity1)
      homogeneity2 = skimage.feature.graycoprops(glcm2, 'homogeneity')
      homogeneity4 = skimage.feature.graycoprops(glcm2, 'homogeneity')
      homogeneity8 = skimage.feature.graycoprops(glcm8, 'homogeneity')
      homogeneity16 = skimage.feature.graycoprops(glcm16, 'homogeneity')
      homogeneity32 = skimage.feature.graycoprops(glcm32, 'homogeneity')

      #Calcular o Contraste
      contrast1 = skimage.feature.graycoprops(glcm1, 'contrast')
      print("CONTRAST1,1 ", contrast1)
      contrast2 = skimage.feature.graycoprops(glcm2, 'contrast')
      contrast4 = skimage.feature.graycoprops(glcm4, 'contrast')
      contrast8 = skimage.feature.graycoprops(glcm8, 'contrast')
      contrast16 = skimage.feature.graycoprops(glcm16, 'contrast')
      contrast32 = skimage.feature.graycoprops(glcm32, 'contrast')

   if event == "-HU_MOMENTS-":
      img = cv2.imread(file_path)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret,thresh = cv2.threshold(gray,170,255,0)
      contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      print("Number of contours detected:",len(contours))

      # compute HuMoments for all the contours detected in the image
      for i, cnt in enumerate(contours):
         x,y = cnt[0,0]
         moments = cv2.moments(cnt)
         hm = cv2.HuMoments(moments)
         cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
         # cv2.putText(img, f'Contour {i+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
         print(f"\nHuMoments for Contour {i+1}:\n", hm)

      cv2.imshow("Hu-Moments", img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

   if event == "-XGBOOST_BINARY-":
      # Treinamento Binario
      
      train_images = []
      train_labels = []
      classes = ["Negative", "Other"]

      # for directory_path in glob.glob("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Training/*"):
      # label = "Negative for intraepithelial lesion"
      # classes.append("Negative for intraepithelial lesion")
      # classes.append("Other")
      for directory_path in glob.glob("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Training/*"):
         label = directory_path.split("\\")[-1]
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100,100))
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(hist)
            if "Negative for intraepithelial lesion" in label:
               train_labels.append(label)
            else:
               train_labels.append("Other")

      train_images = np.array(train_images)
      train_labels = np.array(train_labels)

      test_images = []
      test_labels = []

      for directory_path in glob.glob("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Validation/*"):
         label = directory_path.split("\\")[-1]
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100,100))
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            test_images.append(hist)
            if "Negative for intraepithelial lesion" in label:
               test_labels.append(label)
            else:
               test_labels.append("Other")

      test_images = np.array(test_images)
      test_labels = np.array(test_labels)

      #Encode labels from test to integers
      le = preprocessing.LabelEncoder()
      le.fit(test_labels)
      test_labels_encoded = le.transform(test_labels)
      le.fit(train_labels)
      train_labels_encoded = le.transform(train_labels)

      # print("TRAIN LABELS:", train_labels_encoded)
      # print("TEST LABELS:", test_labels_encoded)

      base_model = tf.keras.applications.VGG16(include_top=False,
                                                     weights='imagenet')
      base_model.trainable = False
      base_model.summary()

      train_images = train_images.reshape(train_images.shape[0], -1)
      test_images = test_images.reshape(test_images.shape[0], -1)

      # model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=120)
      model = xgb.XGBClassifier(objective='binary:logistic')

      # eval_set = [(train_images, train_labels_encoded), (test_images, test_labels_encoded)]

      model.fit(
          train_images,
          train_labels_encoded,  
          verbose=True
      )

      accuracy = model.score(test_images, test_labels_encoded)
      print("Accuracy: %.2f%%" % (accuracy * 100.0))

      train_predictions = model.predict(train_images)
      val_predictions = model.predict(test_images)

      print ("Training Accuracy = ", accuracy_score(train_labels_encoded, train_predictions))
      print ("Validation Accuracy = ", accuracy_score(test_labels_encoded, val_predictions))

      cf = tf.math.confusion_matrix(labels=test_labels_encoded, predictions=val_predictions).numpy()
      ax = plt.subplot()
      sns.heatmap(cf, annot=True, fmt='g', ax=ax)
      ax.set_title('Confusion Matrix')
      ax.set_xlabel('Predicted labels')
      ax.set_ylabel('True labels')
      ax.xaxis.set_ticklabels(classes) 
      ax.yaxis.set_ticklabels(classes)
      plt.xticks(rotation = 90)
      plt.yticks(rotation = 360)

      plt.show()

   if event == "-XGBOOST_TRAINING-":      
      # Treinamento com as 6 classes
      train_images = []
      train_labels = []
      classes = []

      for directory_path in glob.glob("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Training/*"):
         label = directory_path.split("\\")[-1]
         classes.append(directory_path.split("\\")[-1])
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100,100))
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(hist)
            train_labels.append(label)

      train_images = np.array(train_images)
      train_labels = np.array(train_labels)

      test_images = []
      test_labels = []

      for directory_path in glob.glob("C:/Users/Vinicius/Desktop/CS Files/PI/Image-Processing/Validation/*"):
         label = directory_path.split("\\")[-1]
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100,100))
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            test_images.append(hist)
            test_labels.append(label)

      test_images = np.array(test_images)
      test_labels = np.array(test_labels)

      #Encode labels from test to integers
      le = preprocessing.LabelEncoder()
      le.fit(test_labels)
      test_labels_encoded = le.transform(test_labels)
      le.fit(train_labels)
      train_labels_encoded = le.transform(train_labels)

      base_model = tf.keras.applications.VGG16(include_top=False,
                                                     weights='imagenet')
      base_model.trainable = False
      base_model.summary()

      # train_features = base_model.predict(train_images)
      # train_features = train_features.reshape(train_features.shape[0], -1)
      # val_features = base_model.predict(test_images)
      # val_features = val_features.reshape(val_features.shape[0], -1)

      train_images = train_images.reshape(train_images.shape[0], -1)
      test_images = test_images.reshape(test_images.shape[0], -1)

      # model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=120)
      model = xgb.XGBClassifier()

      # eval_set = [(train_features, train_labels_encoded), (val_features, test_labels_encoded)]
      eval_set = [(train_images, train_labels_encoded), (test_images, test_labels_encoded)]

      model.fit(
          train_images, 
          train_labels_encoded,  
          eval_metric=["merror"], 
          eval_set=eval_set, 
          verbose=True
      )

      results = model.evals_result()

      train_error = results['validation_0']['merror']
      train_acc = [1.0 - i for i in train_error]

      val_error = results['validation_1']['merror']
      val_acc = [1.0 - i for i in val_error]

      # best_ntree_limit = model.best_ntree_limit()
      # print('Best ntree limit: ', best_ntree_limit)

      plt.figure(figsize=(8, 8))
      plt.subplot(2, 1, 1)
      plt.plot(train_acc, label='Train Accuracy')
      plt.plot(val_acc, label='Validation Accuracy')
      plt.legend(loc='lower right')
      plt.ylabel('Accuracy')
      plt.ylim([0.3,1.1])
      # plt.axvline(best_ntree_limit-1, color="gray", label="Optimal tree number")
      plt.title('Training and Validation Accuracy')

      plt.subplot(2, 1, 2)
      plt.plot(train_error, label='Training Loss')
      plt.plot(val_error, label='Validation Loss')
      plt.legend()
      plt.title('Training and Validation Loss')
      # plt.axvline(best_ntree_limit-1, color="gray", label="Optimal tree number")
      plt.ylabel('merror')
      plt.ylabel('Number of Trees')

      # plt.show()

      train_predictions = model.predict(train_images)
      val_predictions = model.predict(test_images)

      print ("Training Accuracy = ", accuracy_score(train_labels_encoded, train_predictions))
      print ("Validation Accuracy = ", accuracy_score(test_labels_encoded, val_predictions))
      
      # cm = confusion_matrix(test_labels_encoded, val_predictions)
      # plt.figure(figsize = (10,8))
      # sns.heatmap(cm, annot=True)
      # plt.show()

      cf = tf.math.confusion_matrix(labels=test_labels_encoded, predictions=val_predictions).numpy()
      ax = plt.subplot()
      sns.heatmap(cf, annot=True, fmt='g', ax=ax)
      ax.set_title('Confusion Matrix')
      ax.set_xlabel('Predicted labels')
      ax.set_ylabel('True labels')
      ax.xaxis.set_ticklabels(classes) 
      ax.yaxis.set_ticklabels(classes)
      plt.xticks(rotation = 90)
      plt.yticks(rotation = 360)

      plt.show()

   if event == "-XGBOOST_PREDICT-":
      print(file_path)

      img = cv2.imread(file_path, cv2.IMREAD_COLOR)
      img = cv2.resize(img, (100,100))
      hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
      hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
      validation_images = np.array(hist)
      # print("BEFORE: ", hist)
      validation_images = np.expand_dims(hist, axis=0) #Expand dims so the input is (num images, x, y, c)
      # print("AFTER: ", test_images)
      # le = preprocessing.LabelEncoder()
      # le.fit(test_labels)
      # test_labels_encoded = le.transform(test_labels)
      validation_images = validation_images.reshape(validation_images.shape[0], -1)

      val_prediction = model.predict(validation_images)
      prediction = le.inverse_transform([val_prediction])

      # input_img = np.expand_dims(file_path, axis=0)
      # input_img_feature=VGG_model.predict(input_img)
      # input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
      # prediction = model.predict(input_img)
      # prediction = le.inverse_transform([prediction])
      print("The prediction for this image is: ", prediction)
      print("The actual label for this image is: ", file_path)      

   if event == psg.WIN_CLOSED:
      break

window.close()