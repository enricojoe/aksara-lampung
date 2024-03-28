import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tkinter import *
from tkinter import filedialog
import cv2 as cv
import numpy as np
from PIL import Image
import customtkinter
from glob import glob
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dropout, Dense

from model import KonversiAksaraModel

IMAGE_SIZE = 50

class WindowApp:
  def __init__(self, root, w, h) -> None:
    self.root = root
    self.root.title('Konversi Aksara Lampung')
    self.root.minsize(w, h)

    tabview = customtkinter.CTkTabview(self.root,
                                       width=self.root._current_width - 20,
                                       height=self.root._current_height - 60)
    tabview.place(x=10, y=50)
    
    tab_konversi = tabview.add('Konversi')
    tab_segmentasi = tabview.add('Segmentasi')

    tabview.set('Konversi')

    label_judul = customtkinter.CTkLabel(self.root, 
                                         text='Konversi Aksara Lampung ke Indonesia', 
                                         width=40, height=28, 
                                         fg_color='transparent', 
                                         font=('arial', 20))
    label_judul.place(relx=.5, y=30, anchor='center')
    
    self.buat_tab_konversi(tab_konversi)
    self.buat_tab_segmentasi(tab_segmentasi)

  def buat_tab_konversi(self, root):
    button = customtkinter.CTkButton(root, 
                                     text='Pilih Gambar...', 
                                     width=50, height=28,
                                     command=lambda: self.prediksi_aksara(gambar, hasil))
    button.place(x=15, rely=.5)

    self.rata_tengah = 125

    label_gambar = customtkinter.CTkLabel(root, 
                                          text='Gambar:', 
                                          width=40, height=28, 
                                          fg_color='transparent')
    label_gambar.place(x=self.rata_tengah, y=70)

    gambar = customtkinter.CTkLabel(root,
                                         text=None,
                                         width=450, height=450)
    gambar.place(x=self.rata_tengah, y=100)

    label_hasil = customtkinter.CTkLabel(root,
                                         text='Hasil:', 
                                         width=40, height=28, 
                                         fg_color='transparent')
    label_hasil.place(x=self.rata_tengah, y=560)

    hasil = customtkinter.CTkLabel(root, 
                                    text=None, 
                                    width=40, height=28, 
                                    fg_color='transparent')
    hasil.place(x=self.rata_tengah, y=600)

    self.konversi_aksara = KonversiAksara()

  def buat_tab_segmentasi(self, root):
    button = customtkinter.CTkButton(root, 
                                     text='Pilih Gambar...', 
                                     width=50, height=28,
                                     command=lambda: self.segmentasi_teks(gambar, root))
    button.place(x=15, rely=.5)

    rata_tengah = 125

    label_gambar = customtkinter.CTkLabel(root, 
                                          text='Gambar:', 
                                          width=40, height=28, 
                                          fg_color='transparent')
    label_gambar.place(x=rata_tengah, y=70)

    gambar = customtkinter.CTkLabel(root,
                                         text=None,
                                         width=450, height=450)
    gambar.place(x=rata_tengah, y=100)

    label_hasil = customtkinter.CTkLabel(root,
                                         text='Hasil Segmentasi:', 
                                         width=40, height=28, 
                                         fg_color='transparent')
    label_hasil.place(x=rata_tengah, y=560)

    hasil = customtkinter.CTkLabel(root, 
                                   text=None, 
                                   width=40, height=28, 
                                   fg_color='transparent')
    hasil.place(x=rata_tengah, y=600)

  def pilih_gambar(self, container_gambar) -> None:
    file_path = filedialog.askopenfilename(title='Pilih Gambar',
                                           filetypes =[('Image Files', '*.jpeg *.jpg *.png')])
    if file_path:
        cv_image = cv.imread(file_path)

        h, w = cv_image.shape[:2]
        new_h, new_w = 0, 0
        if h > w:
           new_w = (450/h) * w
           new_h = 450
        else:
           new_h = (450/w) * h
           new_w = 450

        image = customtkinter.CTkImage(light_image=Image.fromarray(cv_image),
                                       size=(new_w, new_h))

        container_gambar.configure(image=image)
        container_gambar.image = image

        return cv_image

  def prediksi_aksara(self, container_gambar, container_hasil) -> None:
    gambar = self.pilih_gambar(container_gambar)
    self.konversi_aksara.setGambar(gambar)
    text = self.konversi_aksara.uji_prediksi()

    container_hasil.configure(text=text.capitalize())

  def segmentasi_teks(self, container_gambar, root) -> None:
    gambar = self.pilih_gambar(container_gambar)
    pengolahan_gambar = PengolahanGambar(gambar)
    list_kata = pengolahan_gambar.word_segmentation()
    show_gambar = []
    jarak_x = 0
    for i, kata in enumerate(list_kata):
      h, w = kata.shape[:2]
      new_h, new_w = 0, 0
      
      new_w = (50/h) * w
      new_h = 50
      show_gambar.append(customtkinter.CTkImage(light_image=Image.fromarray(kata),
                                                size=(new_w, new_h)))
      label = customtkinter.CTkLabel(root, text=None, width=new_w, height=new_h, fg_color='transparent')
      label.place(x=125+jarak_x, y=600)
      label.configure(image=show_gambar[i])
      label.image = show_gambar[i]
      jarak_x = jarak_x + new_w + 10

      

class PengolahanGambar:
  def __init__(self, image) -> None:
    self.image = image
  
  def gray_to_thresh(self):
    gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9,9), 0)
    thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,11,2)
    return thresh
  
  def tambahMargin(self, persen = 0):
    h, w = self.image.shape[:2]
    new_h = int(h * persen / 2)
    new_w = int(w * persen / 2)
    return cv.copyMakeBorder(self.image, new_h, new_h, new_w, new_w, cv.BORDER_CONSTANT, None, value=0)
  
  def word_segmentation(self) -> list:
    thresh = self.gray_to_thresh()

    nlabels, labels, stats, _ = cv.connectedComponentsWithStats(thresh, None, None, None, 8, cv.CV_32S)
    sizes = stats[1:, -1]
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 150:
            img2[labels == i + 1] = 255

    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (20,10))
    dilation = cv.dilate(img2, kernel2, iterations=5)

    contours, _ = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])

    list_kata = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        crop_img = img2[y:y + h, x:x + w]
        list_kata.append(crop_img)
    return list_kata

class KonversiAksara:
  def __init__(self) -> None:
    self.keras_model = KonversiAksaraModel().buat_model()
    self.keras_model.load_weights('./model_checkpoint/cp.weights.h5')

  def get_kelas_induk(self, idx) -> str:
    kelas = ['A', 'Ba', 'Ca', 'Da', 'Ga', 'Gha', 'Ha', 
            'Ja', 'Ka', 'La', 'Ma', 'Na', 'Nga', 'Nya', 
            'Pa', 'Ra', 'Sa', 'Ta', 'Wa', 'Ya']
    return kelas[idx]
  
  def get_kelas_anak(self, idx) -> str:
    kelas = ['0', 'A', 'Ah', 'Ai', 'An', 'Ang', 'Ar', 
            'Au', 'E', 'E2', 'I', 'O', 'U']
    return kelas[idx]

  def setGambar(self, image) -> None:
    self.cv_image = image
    self.pengolahan_gambar = PengolahanGambar(image)

  def prediksi(self, img) -> str:
    gambar = PengolahanGambar(img)
    margin = gambar.tambahMargin(.1)
    resized = cv.resize(margin, (IMAGE_SIZE, IMAGE_SIZE))
    to_predict = resized.reshape(1, *resized.shape, 1) / 255.0
    predicted_induk, predicted_anak = self.keras_model.predict(to_predict, verbose=0)
    predicted_induk = self.get_kelas_induk(np.argmax(predicted_induk))
    predicted_anak = self.get_kelas_anak(np.argmax(predicted_anak))

    if predicted_anak[0].lower() == 'a':
        hasil = predicted_induk + predicted_anak[1:]
    elif str(predicted_anak) == '0':
        hasil = predicted_induk[0]
    else:
        hasil = predicted_induk[0] + predicted_anak
    return hasil.lower()
  
  def uji_prediksi(self) -> str:
    list_kata = self.pengolahan_gambar.word_segmentation()
    teks = []
    for kata in list_kata:
        hasil_kata = ""
        kernel_aksara = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
        dilation_aksara = cv.dilate(kata, kernel_aksara, iterations=3)
        
        contours, _ = cv.findContours(dilation_aksara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])
        for aksara in contours:
            x, y, w, h = cv.boundingRect(aksara)
            gambar_aksara = kata[y:y + h, x:x + w]
            contours, _ = cv.findContours(gambar_aksara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            if len(contours) > 1:
                for ctr in contours[1:]:
                    contour = np.vstack((contour, ctr))
                contour = cv.convexHull(contour)
            x, y, w, h = cv.boundingRect(contour)
            crop_img = gambar_aksara[y:y + h, x:x + w]
            predicted = self.prediksi(crop_img)
            hasil_kata += predicted
        teks.append(hasil_kata)
    return " ".join(teks)

def main():
  root = customtkinter.CTk()
  app = WindowApp(root, 800, 900)
  root.mainloop()

if __name__ == "__main__":
  main()