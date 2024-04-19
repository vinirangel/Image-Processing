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
import cv2
import scipy

main_layout = [[psg.Button("Carregar Imagem", key="-LOAD-")], 
         [psg.Button("Visualizar Imagem", key="-VIEW-")],
         [psg.Button("Tons de Cinza", key="-GRAYSCALE-")],
         [psg.Button("Histograma Tons de Cinza", key="-GRAYSCALE_HISTOGRAM-")],
         [psg.Button("Histograma HSV", key="-HSV_HISTOGRAM-")],
         [psg.Button("Matriz de Co-ocorrencia", key="-COMATRIX-")],
         [psg.Button("Momentos Invariantes de Hu", key="-HU_MOMENTS-")],
]

window = psg.Window('Processamento de Imagens', main_layout, size=(715,250))

#  ADICIONAR POPUP CASO FILE ESTEJA VAZIO

file = "EMPTY"
file_path = "/home/vinicius/Desktop/PI/sample.jpg"

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
      hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 16, 0, 256] )
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
      # print("C1,1 \n", glcm1[0:5, 0:5])
      #C2,2
      glcm2 = skimage.feature.graycomatrix(img_array, distances=[2,2], angles=[3*np.pi/2], symmetric=False, normed=False)      
      # print("C2,2 \n", glcm2[0:5, 0:5])
      #C4,4
      glcm4 = skimage.feature.graycomatrix(img_array, distances=[4,4], angles=[3*np.pi/2], symmetric=False, normed=False)
      print("C4,4 \n", glcm4[0:5, 0:5])
      #C8,8
      glcm8 = skimage.feature.graycomatrix(img_array, distances=[8,8], angles=[3*np.pi/2], symmetric=False, normed=False)
      #C16,16
      glcm16 = skimage.feature.graycomatrix(img_array, distances=[16,16], angles=[3*np.pi/2], symmetric=False, normed=False)
      #C32,32
      glcm32 = skimage.feature.graycomatrix(img_array, distances=[32,32], angles=[3*np.pi/2], symmetric=False, normed=False)

      #Calcular a Entropia 
      entropy1 = skimage.measure.shannon_entropy(glcm1)
      # print("Entropy 1,1 ",entropy1)
      entropy2 = skimage.measure.shannon_entropy(glcm2)
      # print("Entropy 2,2 ",entropy2)
      entropy4 = skimage.measure.shannon_entropy(glcm4)
      entropy8 = skimage.measure.shannon_entropy(glcm8)
      entropy16 = skimage.measure.shannon_entropy(glcm16)
      entropy32 = skimage.measure.shannon_entropy(glcm32)

      #Calcular a Homogeneidade
      homogeneity1 = skimage.feature.graycoprops(glcm1, 'homogeneity')
      # print("HOMOGENEITY1,1 ", homogeneity1)
      homogeneity2 = skimage.feature.graycoprops(glcm2, 'homogeneity')
      homogeneity4 = skimage.feature.graycoprops(glcm2, 'homogeneity')
      homogeneity8 = skimage.feature.graycoprops(glcm8, 'homogeneity')
      homogeneity16 = skimage.feature.graycoprops(glcm16, 'homogeneity')
      homogeneity32 = skimage.feature.graycoprops(glcm32, 'homogeneity')

      #Calcular o Contraste
      contrast1 = skimage.feature.graycoprops(glcm1, 'contrast')
      # print("CONTRAST1,1 ", contrast1)
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
         cv2.putText(img, f'Contour {i+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
         print(f"\nHuMoments for Contour {i+1}:\n", hm)

      cv2.imshow("Hu-Moments", img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

   if event == psg.WIN_CLOSED:
      break

window.close()