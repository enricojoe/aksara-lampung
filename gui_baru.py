import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pytesseract

class AksaraLampungConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Konversi Aksara Lampung")

        self.label_title = tk.Label(root, text="Konversi Aksara Lampung", font=("Arial", 16))
        self.label_title.pack(pady=10)

        self.label_image = tk.Label(root)
        self.label_image.pack(pady=10)

        self.btn_browse = tk.Button(root, text="Browse Image", command=self.browse_image)
        self.btn_browse.pack(pady=5)

        self.btn_convert = tk.Button(root, text="Convert", command=self.convert_image)
        self.btn_convert.pack(pady=5)

        self.label_output = tk.Label(root, text="", font=("Arial", 12))
        self.label_output.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = Image.open(file_path)
            self.image = self.image.resize((300, 300), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(self.image)
            self.label_image.config(image=self.photo)

    def convert_image(self):
        if hasattr(self, 'image'):
            text = pytesseract.image_to_string(self.image, lang='lamp')
            self.label_output.config(text=text)
        else:
            self.label_output.config(text="Error: No image selected.")

def main():
    root = tk.Tk()
    app = AksaraLampungConverter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
