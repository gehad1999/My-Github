from PyQt5 import QtWidgets
from Qt import Ui_MainWindow
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget , QTableWidgetItem ,QVBoxLayout
import soundfile as sn
import numpy as np
import  os, glob
from PIL import Image
import imagehash
import matplotlib.pyplot as plot
from scipy.io import wavfile
import cv2
import librosa

from sklearn.metrics.pairwise import cosine_similarity
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(609,566)
        self.ui.Mix_2.clicked.connect(self.Compare)
        self.ui.tableWidget.setColumnCount(2)
        self.ui.tableWidget.setHorizontalHeaderLabels(["Similarity Factor" , "Song Name"])
        self.ui.tableWidget.setUpdatesEnabled(True)
       

        self.ui.Browse_Audio_1.clicked.connect(lambda :self.Browse(self.ui.lineEdit_Audio_1,self.ui.Browse_Audio_1))
        self.ui.Browse_Audio_2.clicked.connect(lambda :self.Browse(self.ui.lineEdit_Audio_2,self.ui.Browse_Audio_2))
        self.ui.Mix.clicked.connect(self.Mixing)
        self.ui.horizontalSlider.valueChanged.connect(self.slidervaluechange1)
        self.ui.Hash.stateChanged.connect(self.hash)
        self.VBoxLayout=QVBoxLayout()
        self.VBoxLayout.addWidget(self.ui.tableWidget)
        self.setLayout(self.VBoxLayout)
        self.show()
        self.data1_wav=[]
        self.data2_wav=[]
        self.path :str 
        self.fs=44100
       
    def slidervaluechange1(self,value1):
        self.slidervaluechange1 = value1/100
        return (self.slidervaluechange1)



    def Browse(self,lineEdit,button):
        
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath() , '*.wav')  
        if button==self.ui.Browse_Audio_1:
            [self.data1_wav,self.fs] = sn.read(self.fileName,dtype='int16')
        elif button==self.ui.Browse_Audio_2: 
            [self.data2_wav,self.fs] = sn.read(self.fileName,dtype='int16')   
        
        lineEdit.setText(self.fileName)


    def Mixing(self):
        rows, cols = (600000, 2) 
        arr=[]
        for i in range (cols): 
              col = [] 
              for j in range(rows): 
                     col.append(self.slidervaluechange1*np.array(self.data1_wav[j])+(1-self.slidervaluechange1)*np.array(self.data2_wav[j])) 
              arr.append(col) 
        data3=np.concatenate(arr)
        plot.specgram(data3[:,0],Fs=self.fs)
        path='./pHash/'
        plot.savefig (path+'mix.png' )        
    def score(self,str1:'str',str2:'str'):
        return (100-(str1-str2))/100

    def hash(self):
        rowPosition = self.ui.tableWidget.rowCount()

        with open('output.txt', 'w') as f:

         for image in glob.glob(os.path.join('./Spectrogram', '*.png')):
            img = cv2.imread(image,0)

            hash = imagehash.phash(Image.fromarray(img))

            f.write("{}\n".format(hash) )
            for i in glob.glob(os.path.join('./pHash', '*.png')):
                imgg = cv2.imread(i,0)
                otherhash = imagehash.phash(Image.fromarray(imgg))
                h=self.score(hash,otherhash)
                self.ui.tableWidget.resizeColumnsToContents()
                self.ui.tableWidget.insertRow(rowPosition)
                self.ui.tableWidget.setItem(rowPosition,0,QtWidgets.QTableWidgetItem(str(h)))
                self.ui.tableWidget.setItem(rowPosition,1,QtWidgets.QTableWidgetItem(str(image)))
                rowPosition+=1
        for j in os.listdir('./pHash'):
            os.remove('./pHash/'+j)      
    def feature(self):
        rowPosition = self.ui.tableWidget.rowCount()
        for filename in glob.glob(os.path.join('./Wave', '*.wav')):
            y, sr = librosa.load(filename)
            S, phase = librosa.magphase(librosa.stft(y))  
            s1=librosa.feature.spectral_rolloff(S=S)
            feature1 = s1[0:1000, 0:1000]
        
            for i in glob.glob(os.path.join('./feature', '*.wav')):
                u, sr = librosa.load(i)
                f, phase = librosa.magphase(librosa.stft(u))  
                s2=librosa.feature.spectral_rolloff(S=f)
                feature2 = s2[0:1000, 0:1000]
                val_out = cosine_similarity( feature1, feature2)
                self.ui.tableWidget.resizeColumnsToContents()
                self.ui.tableWidget.insertRow(rowPosition)
                self.ui.tableWidget.setItem(rowPosition,0,QtWidgets.QTableWidgetItem(str(val_out)))
                self.ui.tableWidget.setItem(rowPosition,1,QtWidgets.QTableWidgetItem(str(filename)))
                rowPosition+=1
        for j in os.listdir('./feature'):
            os.remove('./feature/'+j)  

    def Compare(self):
        if self.ui.Hash.isChecked():
            self.hash()
        elif self.ui.Feature.isChecked():
            self.feature()    




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()
if __name__ == "__main__":
    main()





