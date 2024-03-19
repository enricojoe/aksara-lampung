import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tkinter import *
from tkinter import filedialog, ttk
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import customtkinter
from glob import glob
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dropout, Dense

class WindowApp:
  def __init__(self, root, w, h) -> None:
    self.root = root
    self.root.title('Konversi Aksara Lampung')
    self.root.minsize(w, h)

    label_judul = customtkinter.CTkLabel(self.root, 
                                         text='Konversi Aksara Lampung ke Indonesia', 
                                         width=40, height=28, 
                                         fg_color='transparent', 
                                         font=('arial', 20))
    label_judul.place(relx=.5, y=30, anchor='center')
    
    button = customtkinter.CTkButton(self.root, 
                                     text='Pilih Gambar...', 
                                     width=50, height=28,
                                     command=self.pilih_gambar)
    button.place(x=15, rely=.5)

    self.rata_tengah = 125

    label_gambar = customtkinter.CTkLabel(self.root, 
                                          text='Gambar:', 
                                          width=40, height=28, 
                                          fg_color='transparent')
    label_gambar.place(x=self.rata_tengah, y=70)

    self.gambar = customtkinter.CTkLabel(self.root,
                                         text=None,
                                         width=450, height=450)
    self.gambar.place(x=self.rata_tengah, y=100)

    label_hasil = customtkinter.CTkLabel(self.root,
                                         text='Hasil:', 
                                         width=40, height=28, 
                                         fg_color='transparent')
    label_hasil.place(x=self.rata_tengah, y=560)

    self.hasil = customtkinter.CTkLabel(self.root, 
                                        text=None, 
                                        width=40, height=28, 
                                        fg_color='transparent')
    self.hasil.place(x=self.rata_tengah, y=600)

    self.konversi_aksara = KonversiAksara()

  def pilih_gambar(self) -> None:
    file_path = filedialog.askopenfilename(title='Pilih Gambar',
                                           filetypes =[('Image Files', '*.jpeg *.jpg *.png')])
    if file_path:
        cv_image = cv.imread(file_path)

        self.konversi_aksara.setGambar(cv_image)
        text = self.konversi_aksara.uji_prediksi()

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

        self.gambar.configure(image=image)
        self.gambar.image = image

        self.hasil.configure(text=text.capitalize())

class KonversiAksara:
  def __init__(self) -> None:
    self.keras_model = self.model()
    self.keras_model.load_weights('./konversi-aksara/aksara-lampung/model_checkpoint/cp.weights.h5')
  
  def model(self) -> None:
    cnn_model = Sequential([
      Input(shape=(28, 28, 1)),
      Conv2D(filters=32,kernel_size=3,activation='relu'),
      MaxPooling2D(pool_size=2),
      Dropout(0.2),
      Conv2D(filters=64,kernel_size=2,activation='relu'),
      MaxPooling2D(pool_size=2),
      Dropout(0.2),
      Conv2D(filters=128,kernel_size=3,activation='relu'),
      MaxPooling2D(pool_size=2),
      Dropout(0.35),
      Flatten(), # flatten out the layers
      Dense(1000,activation='relu'),
      Dropout(0.4),
      Dense(500,activation='relu'),
      Dense(258,activation = 'softmax')  
    ])
    cnn_model.compile(loss ='sparse_categorical_crossentropy', 
                      optimizer=Adam(0.0001),
                      metrics =['accuracy'])
    return cnn_model

  def getKelas(self, idx) -> str:
     kelas = ['A', 'Ah', 'Ai', 'An', 'Ang', 'Ar', 'Au', 
              'B', 'Ba', 'Bah', 'Bai', 'Ban', 'Bang', 'Bar', 'Bau', 'Be', 'Be2', 'Bi', 'Bo', 'Bu', 
              'C', 'Ca', 'Cah', 'Cai', 'Can', 'Cang', 'Car', 'Cau', 'Ce', 'Ce2', 'Ci', 'Co', 'Cu', 
              'D', 'Da', 'Dah', 'Dai', 'Dan', 'Dang', 'Dar', 'Dau', 'De', 'De2', 'Di', 'Do', 'Du', 
              'E', 'E2', 'G', 'Ga', 'Gah', 'Gai', 'Gan', 'Gang', 'Gar', 'Gau', 'Ge', 'Ge2', 'Gh', 
              'Gha', 'Ghah', 'Ghai', 'Ghan', 'Ghang', 'Ghar', 'Ghau', 'Ghe', 'Ghe2', 'Ghi', 'Gho', 'Ghu', 
              'Gi', 'Go', 'Gu', 'H', 'Ha', 'Hah', 'Hai', 'Han', 'Hang', 'Har', 'Hau', 'He', 'He2', 'Hi', 
              'Ho', 'Hu', 'I', 'J', 'Ja', 'Jah', 'Jai', 'Jan', 'Jang', 'Jar', 'Jau', 'Je', 'Je2', 'Ji', 
              'Jo', 'Ju', 'K', 'Ka', 'Kah', 'Kai', 'Kan', 'Kang', 'Kar', 'Kau', 'Ke', 'Ke2', 'Ki', 'Ko', 
              'Ku', 'L', 'La', 'Lah', 'Lai', 'Lan', 'Lang', 'Lar', 'Lau', 'Le', 'Le2', 'Li', 'Lo', 'Lu', 
              'M', 'Ma', 'Mah', 'Mai', 'Man', 'Mang', 'Mar', 'Mau', 'Me', 'Me2', 'Mi', 'Mo', 'Mu', 'N', 
              'Na', 'Nah', 'Nai', 'Nan', 'Nang', 'Nar', 'Nau', 'Ne', 'Ne2', 'Nga', 'Ngah', 'Ngai', 'Ngan', 
              'Ngang', 'Ngar', 'Ngau', 'Nge', 'Nge2', 'Ngi', 'Ngo', 'Ngu', 'Ni', 'No', 'Nu', 'Ny', 'Nya', 
              'Nyah', 'Nyai', 'Nyan', 'Nyang', 'Nyar', 'Nyau', 'Nye', 'Nye2', 'Nyi', 'Nyo', 'Nyu', 'O', 'P', 
              'Pa', 'Pah', 'Pai', 'Pan', 'Pang', 'Par', 'Pau', 'Pe', 'Pe2', 'Pi', 'Po', 'Pu', 'R', 'Ra', 
              'Rah', 'Rai', 'Ran', 'Rang', 'Rar', 'Rau', 'Re', 'Re2', 'Ri', 'Ro', 'Ru', 'S', 'Sa', 'Sah', 
              'Sai', 'San', 'Sang', 'Sar', 'Sau', 'Se', 'Se2', 'Si', 'So', 'Su', 'T', 'Ta', 'Tah', 'Tai', 
              'Tan', 'Tang', 'Tar', 'Tau', 'Te', 'Te2', 'Ti', 'To', 'Tu', 'U', 'W', 'Wa', 'Wah', 'Wai', 
              'Wan', 'Wang', 'War', 'Wau', 'We', 'We2', 'Wi', 'Wo', 'Wu', 'Y', 'Ya', 'Yah', 'Yai', 'Yan', 
              'Yang', 'Yar', 'Yau', 'Ye', 'Ye2', 'Yi', 'Yo', 'Yu']
     return kelas[idx]

  def setGambar(self, image) -> None:
    self.cv_image = image

  def prediksi(self) -> None:
    resized = cv.resize(self.cv_image, (28, 28))
    to_predict = resized.reshape(1, *resized.shape, 1)
    predicted = np.argmax(self.keras_model.predict(to_predict, verbose=0))
    return predicted
  
  def uji_prediksi(self) -> str:
    thresh = self.gray_to_thresh((9, 9), 150)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    morph_open = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (10,7))
    dilation = cv.dilate(morph_open, kernel2, iterations=5)

    contours, _ = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[1])

    list_kata = []
    for contour in contours[::-1]:
        x, y, w, h = cv.boundingRect(contour)
        crop_img = thresh[y:y + h, x:x + w]
        list_kata.append(crop_img)

    teks = []
    for kata in list_kata:
        hasil_kata = ""
        kernel_aksara = cv.getStructuringElement(cv.MORPH_RECT, (5,15))
        dilation_aksara = cv.dilate(kata, kernel_aksara, iterations=2)
        
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
            margin = self.tambahMargin(crop_img, .1)
            resized = cv.resize(margin, (28, 28))
            to_predict = resized.reshape(1, *resized.shape, 1)
            predicted = self.getKelas(np.argmax(self.keras_model.predict(to_predict, verbose=0)))
            hasil_kata += predicted
        teks.append(hasil_kata)
    return " ".join(teks)

  def tambahMargin(self, img, persen = 0):
    h, w = img.shape[:2]
    new_h = int(h * persen / 2)
    new_w = int(w * persen / 2)
    return cv.copyMakeBorder(img, new_h, new_h, new_w, new_w, cv.BORDER_CONSTANT, None, value=0)

  def gray_to_thresh(self, gray_kernel, thresh_value):
    gray = cv.cvtColor(self.cv_image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, gray_kernel, 0)
    _, thresh = cv.threshold(blur, thresh_value, 255, cv.THRESH_BINARY_INV)

    return thresh
  
  def line_segmentation(self) -> list:
    thresh = self.gray_to_thresh((9,9), 150)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    morph_open = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (150,25))
    morph_close = cv.morphologyEx(morph_open, cv.MORPH_CLOSE, kernel2)
    contours, _ = cv.findContours(morph_close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    baris = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        crop_img = self.cv_image[y:y + h, x:x + w]
        baris.append(crop_img)

    return baris[0]
  
  def word_segmentation(self, segmented_line):
    words = []
    for i, line in enumerate(segmented_line):
        thresh = self.gray_to_thresh(line, (7,7), 150)
        
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
        morph1 = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (40,35))
        morph2 = cv.morphologyEx(morph1, cv.MORPH_CLOSE, kernel2)
        contours, _ = cv.findContours(morph2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            crop_img = line[y:y + h, x:x + w]
            words.append(crop_img)

    return words

  def uji_model(self):
    data_uji = {'filename': [], 'data_train': [], 'label': []}
    test_root = "./hasil/"
    list_folder = glob(test_root + "*/")
    for folder in list_folder:
      for isi_folder in glob(folder + "*/"):
        if isi_folder.split("/")[-1] != 'desktop': 
          for aksara in glob(isi_folder + '*'):
              data_uji['filename'].append(aksara)
              data_uji['label'].append(aksara.split("/")[-2])
              image = cv.imread(aksara)
              gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
              invert = cv.bitwise_not(gray)
              resized = cv.resize(invert, (50, 50))
              data_uji['data_train'].append(resized)

    data_uji['normalized'] = np.array(data_uji['data_train']) / 255
    let = LabelEncoder()
    data_uji['coded_label'] = let.fit_transform(data_uji['label'])
    X_test = data_uji['normalized']
    y_test = data_uji['coded_label']
    image_rows = 50
    image_cols = 50
    image_shape = (image_rows,image_cols,1) 
    XX_test = X_test.reshape(X_test.shape[0],*image_shape)
    hasi = self.keras_model.evaluate(XX_test, y_test, verbose=0)
    print(hasi)


def main():
  root = customtkinter.CTk()
  app = WindowApp(root, 700, 700)
  root.mainloop()

if __name__ == "__main__":
  main()