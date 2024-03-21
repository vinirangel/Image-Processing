import io
import PySimpleGUI as psg
from PIL import Image
import shutil
import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import cv2

# layout = [[psg.Text(text='Hello World',
#    font=('Arial Bold', 20),
#    size=20,
#    expand_x=True,
#    justification='center')],
# ]

main_layout = [[psg.Button("Carregar Imagem", key="-LOAD-")], 
         [psg.Button("Visualizar Imagem", key="-VIEW-")],
         [psg.Button("Tons de Cinza", key="-GRAYSCALE-")],
         [psg.Button("Histograma Tons de Cinza", key="-GRAYSCALE_HISTOGRAM-")],
         [psg.Button("Histograma HSV", key="-HSV_HISTOGRAM-")]
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
      assert img is not None, "file could not be read, check with os.path.exists()"
      hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
      hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 16, 0, 256] )
      
      plt.imshow(hist,interpolation = 'nearest')
      plt.show()
      
   if event == psg.WIN_CLOSED:
      break
window.close()