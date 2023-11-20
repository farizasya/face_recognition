import tkinter
from tkinter import messagebox
import cv2, os, numpy as np
import csv
from PIL import ImageTk, Image
import pandas as pd

window = tkinter.Tk()
window.title("Face Recognition")
window.geometry('600x300')
window.configure(bg='#333333')
    
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        messagebox.showerror('Some file missing', 'Please contact us for help')
        window.destroy()

###################################################################################

def TakeImages():
    check_haarcascadefile()
    columns = ['NO', 'NAMA']
    #assure_path_exists("NamaWajah/")
    assure_path_exists("DataWajah/")
    serial = 0
    exists = os.path.isfile("NamaWajah.csv")
    if exists:
        with open("NamaWajah.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("NamaWajah.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()
    nama = entry1.get()

    if  nama.isalpha() == "":
        messagebox.showerror("","Nama Harus Di Isi")
    elif ((nama.isalpha()) or (' ' in nama)):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                cv2.imwrite("DataWajah\ " + nama + "." + str(serial) + "." + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                cv2.imshow('Mengambil Foto', img)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            elif sampleNum>30:
                break

        cam.release()
        cv2.destroyAllWindows()
        row = [serial, nama]
        with open('NamaWajah.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        messagebox.showinfo("Sukses", "Berhasil Mengambil Data")
   
########################################################################################

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("Training/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("DataWajah")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        messagebox.showerror("Data Tidak Ada", "Silahkan Hubungi Admin !")
        return
    recognizer.save("Training\Trainner.yml")
    messagebox.showinfo("","Data Berhasil di Training")

############################################################################################3

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

##############################################################################################

def facerecognition():
    check_haarcascadefile()
    #assure_path_exists("Data_Karyawan/")
    counter=0
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    exists3 = os.path.isfile("Training\Trainner.yml")
    if exists3:
        recognizer.read("Training\Trainner.yml")
    else:
        messagebox.showerror(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    exists1 = os.path.isfile("NamaWajah.csv")
    if exists1:
        df = pd.read_csv("NamaWajah.csv")
    else:
        messagebox.showerror(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()
   
    while True:
        ret, im = cam.read() 
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 100:
                aa = df.loc[df['NO'] == serial]['NAMA'].values
                #ID = df.loc[df['NO'] == serial]['NIK'].values
                #ID = str(ID)
                #ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                #confidance = "  {0}%".format(round(136 - conf))
                
            else:
                Id = 'Tidak Diketahui'
                bb = str(Id)
                #confidance = "  {0}%".format(round(20 - conf))
            cv2.putText(im, str(bb), (x + 30, y - 10), font, 1, (255, 255, 255), 2)    
            #cv2.putText(im, str(confidance), (x + 5, y + h + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Face Rocognition', im)
        if (cv2.waitKey(1) == ord('q') ):
            cv2.destroyAllWindows()
            #window.destroy()
            break

frame = tkinter.Frame(bg='#333333')

entry1 = tkinter.Entry(frame, width=50)
entry1.insert(0, "Tuliskan Nama ")
entry1.bind("<FocusIn>", lambda e: entry1.delete('0', 'end'))

foto_button = tkinter.Button(
    frame, text="Ambil Foto", bg="#0f40a3", fg="#FFFFFF", font=("Arial", 16), command=TakeImages)
training_button = tkinter.Button(
    frame, text="Training Data", bg="#0f40a3", fg="#FFFFFF", font=("Arial", 16), command=TrainImages)
face_button = tkinter.Button(
    frame, text="Face Recognition", bg="#0f40a3", fg="#FFFFFF", font=("Arial", 16), command=facerecognition)

entry1.grid(row=1, column=1, columnspan=4, pady=30)
foto_button.grid(row=2, column=1,padx=10 )
training_button.grid(row=2, column=2,padx=10 )
face_button.grid(row=2, column=3,padx=10)

frame.pack()

window.mainloop()
