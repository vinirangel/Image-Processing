import io
import PySimpleGUI as psg
from PIL import Image
import shutil
import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import numpy as np
import skimage
import glob
import cv2
import os
import time
import xgboost as xgb
from keras.applications import EfficientNetB0
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_sample_weight
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from tensorflow import keras

main_layout = [[psg.Button("Carregar Imagem", key="-LOAD-"), psg.VerticalSeparator(pad=10, color="gray"), psg.Button("Classificação Binaria com XGBoost", key="-XGBOOST_BINARY-")], 
         [psg.Button("Visualizar Imagem", key="-VIEW-"), psg.VerticalSeparator(pad=8, color="gray"), psg.Button("Classificação XGBoost com 6 Classes", key="-XGBOOST_TRAINING-")],
         [psg.Button("Tons de Cinza", key="-GRAYSCALE-"), psg.VerticalSeparator(pad=8, color="gray"), psg.Button("Prever com o Classificador XGBoost", key="-XGBOOST_PREDICT-")],
         [psg.Button("Histograma Tons de Cinza", key="-GRAYSCALE_HISTOGRAM-"),  psg.VerticalSeparator(pad=8, color="gray"), psg.Button("Classificação Binaria com EfficientNet", key="-EFFICIENTNET_BINARY-")],
         [psg.Button("Histograma HSV", key="-HSV_HISTOGRAM-"), psg.VerticalSeparator(pad=8, color="gray"), psg.Button("Classificação EfficientNet com 6 Classes", key="-EFFICIENTNET_TRAINING-")],
         [psg.Button("Matriz de Co-ocorrencia", key="-COMATRIX-"), psg.VerticalSeparator(pad=8, color="gray"), psg.Button("Prever com o Classificador EfficientNet", key="-EFFICIENTNET_PREDICT-")],
         [psg.Button("Momentos Invariantes de Hu", key="-HU_MOMENTS-")],
]

window = psg.Window('Processamento de Imagens', main_layout, size=(715,250))

#  ADICIONAR POPUP CASO FILE ESTEJA VAZIO

file = "EMPTY"
# file_path = "/home/vinicius/Desktop/PI/sample.jpg"
# file_path = "C:/Users/Vinicius/Desktop/CS Files/PI/sample.jpg"
file_path = "/home/vinicius/Desktop/PI/Image-Processing/Validation/ASC-H/816.png"
training_path = "/home/vinicius/Desktop/PI/Image-Processing/Training/*"
validation_path = "/home/vinicius/Desktop/PI/Image-Processing/Validation/*"
train_predictions = ""
val_predictions = ""
train_images = ""
test_images = ""
model = ""
start_time = ""

while True:
   event, values = window.read()
   print(event, values)

   if event == "-LOAD-":
      file_path = psg.popup_get_file('Selecione uma Imagem',  title="Selecionar Imagem")
      file = Image.open(file_path)
      print ("File selected", file)

   if event == "-VIEW-":
      img = plt.imread(file_path)
      imgplot = plt.imshow(img)
      plt.show()

   if event == "-GRAYSCALE-":
      gray_scale_img = "gray_scale.png"
      shutil.copy(file_path, gray_scale_img)
      file = Image.open(gray_scale_img).convert('L')
      file.save(gray_scale_img, "PNG")
      img = plt.imread(gray_scale_img)
      imgplot = plt.imshow(img, cmap='gray')
      plt.show()
      # png_bio = io.BytesIO()
      # file.save(png_bio, format="PNG")
      # png_data = png_bio.getvalue()
      # window3 = psg.Window("Tons de Cinza", [[psg.Image(png_data, subsample=2, key="-IMAGE-")],
      #                                        [psg.Button("Zoom In", key="-ZOOM IN-")]
      #                                        ], size=(720,480), finalize=True)
      
      # event3, values3 = window3.read()

      # if event3 == "-ZOOM IN-":
      #    print("ZOOM ZOOM ZOOM")
      #    window3['-IMAGE-'].update(png_data)

      # if event3 == psg.WIN_CLOSED:
      #    print("CLOSE")
      #    window3.close()

   if event == "-GRAYSCALE_HISTOGRAM-": 
      img = cv2.imread(file_path,0) 
      # calculate frequency of pixels in range 0-255 
      histogram = cv2.calcHist([img],[0],None,[256],[0,256]) 
      plt.plot(histogram)
      plt.show() 

   if event == "-HSV_HISTOGRAM-":
 
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
         print(f"\nHuMoments for Contour {i+1}:\n", hm)

      cv2.imshow("Hu-Moments", img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

   if event == "-XGBOOST_BINARY-":
      start_time = time.time()

      # Treinamento Binario      
      train_images = []
      train_labels = []
      classes = ["Negative", "Other"]

      for directory_path in glob.glob(training_path):
         label = directory_path.split("\\")[-1]
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100,100))
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
            train_images.append(hist)
            if "Negative for intraepithelial lesion" in label:
               train_labels.append(label)
            else:
               train_labels.append("Other")

      train_images = np.array(train_images)
      train_labels = np.array(train_labels)

      test_images = []
      test_labels = []

      for directory_path in glob.glob(validation_path):
         label = directory_path.split("\\")[-1]
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100,100))
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
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

      base_model = tf.keras.applications.VGG16(include_top=False,
                                                     weights='imagenet')
      base_model.trainable = False
      base_model.summary()

      train_images = train_images.reshape(train_images.shape[0], -1)
      test_images = test_images.reshape(test_images.shape[0], -1)

      model = xgb.XGBClassifier(objective='binary:logistic')

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
      print("--- %s TIME ---" % (time.time() - start_time))

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
      start_time = time.time()

      # Treinamento com as 6 classes
      train_images = []
      train_labels = []
      classes = []

      for directory_path in glob.glob(training_path):
         label = directory_path.split("\\")[-1]
         classes.append(directory_path.split("\\")[-1])
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100,100))
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
            train_images.append(hist)
            train_labels.append(label)

      train_images = np.array(train_images)
      train_labels = np.array(train_labels)

      test_images = []
      test_labels = []

      for directory_path in glob.glob(validation_path):
         label = directory_path.split("\\")[-1]
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100,100))
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist( [hsv], [0, 2], None, [16, 8], [0, 180, 0, 256] )
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

      train_images = train_images.reshape(train_images.shape[0], -1)
      test_images = test_images.reshape(test_images.shape[0], -1)

      model = xgb.XGBClassifier()
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

      plt.figure(figsize=(8, 8))
      plt.subplot(2, 1, 1)
      plt.plot(train_acc, label='Train Accuracy')
      plt.plot(val_acc, label='Validation Accuracy')
      plt.legend(loc='lower right')
      plt.ylabel('Accuracy')
      plt.ylim([0.3,1.1])
      plt.title('Training and Validation Accuracy')

      plt.subplot(2, 1, 2)
      plt.plot(train_error, label='Training Loss')
      plt.plot(val_error, label='Validation Loss')
      plt.legend()
      plt.title('Training and Validation Loss')
      plt.ylabel('merror')
      plt.ylabel('Number of Trees')

      plt.show()

      train_predictions = model.predict(train_images)
      val_predictions = model.predict(test_images)

      print ("Training Accuracy = ", accuracy_score(train_labels_encoded, train_predictions))
      print ("Validation Accuracy = ", accuracy_score(test_labels_encoded, val_predictions))
      print("--- %s TIME ---" % (time.time() - start_time))
      
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
      validation_images = np.expand_dims(hist, axis=0) #Expand dims so the input is (num images, x, y, c)
      validation_images = validation_images.reshape(validation_images.shape[0], -1)

      val_prediction = model.predict(validation_images)
      prediction = le.inverse_transform([val_prediction])

      print("The prediction for this image is: ", prediction)
      print("The actual label for this image is: ", file_path)

   if event == "-EFFICIENTNET_BINARY-":
      start_time = time.time()
      
      train_images = []
      train_labels = []
      classes = ["Negative", "Other"]

      for directory_path in glob.glob(training_path):
         label = directory_path.split("\\")[-1]
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100,100))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
            if "Negative for intraepithelial lesion" in label:
               train_labels.append(label)
            else:
               train_labels.append("Other")

      train_images = np.array(train_images)
      train_labels = np.array(train_labels)

      test_images = []
      test_labels = []

      for directory_path in glob.glob(validation_path):
         label = directory_path.split("\\")[-1]
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100,100))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            test_images.append(img)
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

      train_datagen = ImageDataGenerator(
         rescale=1.0/255.0,      # Rescale pixel values to [0, 1]
         rotation_range=25,      # Randomly rotate images by up to 25 degrees
         width_shift_range=0.3,  # Randomly shift the width of images
         height_shift_range=0.3, # Randomly shift the height of images
         horizontal_flip=True,   # Randomly flip images horizontally
         shear_range=0.3,        # Apply shear transformations
         zoom_range=0.4,         # Randomly zoom into images
         fill_mode='nearest'     # Fill empty pixels with the nearest value
      )

      test_datagen = ImageDataGenerator(rescale = 1.0 / 255)
      # Create data generators for training and validation data
      batch_size = 32
      image_size = (100, 100)

      train_generator = train_datagen.flow_from_directory(
         '/home/vinicius/Desktop/PI/Image-Processing/Training/',
         target_size=image_size,
         batch_size=batch_size,
         class_mode='categorical'
      )

      val_generator = test_datagen.flow_from_directory(
         '/home/vinicius/Desktop/PI/Image-Processing/Validation/',
         target_size=image_size,
         batch_size=batch_size,
         class_mode='categorical'
      )

      efficient_net = EfficientNetB0(
         weights='imagenet',
         # input_shape=(266,16,8),
         include_top=False,
         pooling='max',
         classes=2
      )

      model = Sequential([
         efficient_net,
         keras.layers.Flatten(),  # Add a Flatten layer to reshape the output
         keras.layers.Dense(512, activation='relu'),
         keras.layers.BatchNormalization(),  
         keras.layers.Dropout(0.5),  
         keras.layers.Dense(256, activation='relu'),
         keras.layers.BatchNormalization(),  
         keras.layers.Dropout(0.3),
         keras.layers.Dense(128, activation='relu'),
         keras.layers.BatchNormalization(),
         keras.layers.Dropout(0.3),
         keras.layers.Dense(6, activation='softmax'),
      ])

      model.build((None, image_size[0], image_size[1], 3))
    
      lr_schedule = keras.optimizers.schedules.ExponentialDecay(5e-4, decay_steps=10000, decay_rate=0.9)

      model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

      early_stopping = keras.callbacks.EarlyStopping( monitor='val_accuracy',patience=5,restore_best_weights=True)

      history = model.fit(train_generator, 
                           epochs=10, 
                           steps_per_epoch=10,
                           validation_data=val_generator, 
                           callbacks=[early_stopping])

      plt.plot(history.history["accuracy"])
      plt.plot(history.history["val_accuracy"])
      plt.title("model accuracy")
      plt.ylabel("accuracy")
      plt.xlabel("epoch")
      plt.legend(["train", "validation"], loc="upper left")

      predictions = model.predict(val_generator)
      loss, accuracy = model.evaluate(val_generator)
      print(f'Test loss: {loss:.4f}, accuracy: {accuracy:.4f}')
      print("--- %s TIME ---" % (time.time() - start_time))

      true_labels = val_generator.classes
      class_names = list(val_generator.class_indices.keys())

      # Calculate the confusion matrix
      confusion = confusion_matrix(true_labels, np.argmax(predictions, axis=-1))

      # Plot the confusion matrix
      plt.figure(figsize=(8, 6))
      sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
      plt.xlabel("Predicted Labels")
      plt.ylabel("True Labels")
      plt.title("Confusion Matrix")
      plt.show()

   if event == "-EFFICIENTNET_TRAINING-":
      start_time = time.time()
      
      # Treinamento com 6 Classes
      train_images = []
      train_labels = []
      classes = []
      labels = ['ASC-H','ASC-US','HSIL','LSIL','Negative for intraepithelial lesion',' SCC']

      for directory_path in glob.glob(training_path):
         label = directory_path.split("\\")[-1]
         classes.append(directory_path.split("\\")[-1])
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100,100))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
            train_labels.append(label)

      train_images = np.array(train_images)
      train_labels = np.array(train_labels)

      test_images = []
      test_labels = []

      for directory_path in glob.glob(validation_path):
         label = directory_path.split("\\")[-1]
         # print(label)
         for img_path in glob.glob(os.path.join(directory_path, "*png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100,100))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            test_images.append(img)
            test_labels.append(label)

      test_images = np.array(test_images)
      test_labels = np.array(test_labels)

      #Encode labels from test to integers
      le = preprocessing.LabelEncoder()
      le.fit(test_labels)
      test_labels_encoded = le.transform(test_labels)
      le.fit(train_labels)
      train_labels_encoded = le.transform(train_labels)

      # Gerador de Imagens para Oversampling
      train_datagen = ImageDataGenerator(
         rescale=1.0/255.0,      # Rescale pixel values to [0, 1]
         rotation_range=25,      # Randomly rotate images by up to 25 degrees
         width_shift_range=0.3,  # Randomly shift the width of images
         height_shift_range=0.3, # Randomly shift the height of images
         horizontal_flip=True,   # Randomly flip images horizontally
         shear_range=0.3,        # Apply shear transformations
         zoom_range=0.4,         # Randomly zoom into images
         fill_mode='nearest'     # Fill empty pixels with the nearest value
      )

      test_datagen = ImageDataGenerator(rescale = 1.0 / 255)
      # Create data generators for training and validation data
      batch_size = 32
      image_size = (100, 100)

      train_generator = train_datagen.flow_from_directory(
         '/home/vinicius/Desktop/PI/Image-Processing/Training/',
         target_size=image_size,
         batch_size=batch_size,
         class_mode='categorical',
         shuffle=True
      )

      val_generator = test_datagen.flow_from_directory(
         '/home/vinicius/Desktop/PI/Image-Processing/Validation/',
         target_size=image_size,
         batch_size=batch_size,
         class_mode='categorical',
         shuffle=True
      )

      efficient_net = EfficientNetB0(
         weights='imagenet',
         # input_shape=(266,16,8),
         include_top=False,
         pooling='max',
         classes=6
      )

      model = Sequential()
      model.add(efficient_net)
      model.add(Dense(units = 6, activation='relu'))
      model.add(Dense(units = 6, activation = 'relu'))
      model.add(Dense(units = 6, activation='sigmoid'))
      model.summary()

      model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
  
      class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                 classes = np.unique(train_labels_encoded),
                                                 y= train_labels_encoded)
      
      class_weights = dict(enumerate(class_weights))

      early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

      # Treinar o Modelo
      history = model.fit(
         train_generator,
         batch_size=1000,
         epochs = 1,
         steps_per_epoch = 1,
         validation_data = val_generator,
         # validation_steps = 3,
         class_weight = class_weights,
         callbacks=[early_stopping]
      )

      plt.plot(history.history["accuracy"])
      plt.plot(history.history["val_accuracy"])
      plt.title("model accuracy")
      plt.ylabel("accuracy")
      plt.xlabel("epoch")
      plt.legend(["train", "validation"], loc="upper left")

      predictions = model.predict(val_generator)
      loss, accuracy = model.evaluate(val_generator)
      print(f'Test loss: {loss:.4f}, accuracy: {accuracy:.4f}')
      print("--- %s TIME ---" % (time.time() - start_time))

      true_labels = val_generator.classes
      class_names = list(val_generator.class_indices.keys())

      # Calculate the confusion matrix
      confusion = confusion_matrix(true_labels, np.argmax(predictions, axis=-1))

      # Plot the confusion matrix
      plt.figure(figsize=(8, 6))
      sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
      plt.xlabel("Predicted Labels")
      plt.ylabel("True Labels")
      plt.title("Confusion Matrix")
      plt.show()

   if event == "-EFFICIENTNET_PREDICT-":
      print(file_path)

      img = cv2.imread(file_path, cv2.IMREAD_COLOR)
      img = cv2.resize(img, (100,100))
      validation_image = np.array(img)
      validation_image = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)

      pred = model.predict(validation_image)

      print("The prediction for this image is: ", pred)
      print("The actual label for this image is: ", file_path)   

   if event == psg.WIN_CLOSED:
      break

window.close()