import sys
import matplotlib.pyplot as plt
from PyQt5 import uic,QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import*
from PyQt5.QtWidgets import QMainWindow
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout,QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import cv2
import numpy as np
from imageModel import ImageModel
from modesEnum import  Modes
import enum 
from PyQt5.QtWidgets import QMessageBox
import logging
import logging.config
#import mylib




class mixer(QMainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self   )
        super(mixer, self).__init__(self)
        self.LOG_FILENAME = 'tmp.log'
        logging.basicConfig(filename=self.LOG_FILENAME,level=logging.DEBUG)
        

        logging.debug('This message should go to the log file')
        uic.loadUi("qt.ui",self)
        self.setWindowTitle("PyQt5 & Matplotlib Example GUI")
        self.msg = QMessageBox()    
        self.pushButton.clicked.connect(self.Image1)
        self.pushButton_2.clicked.connect(self.Image2)
        self.checkBox_7.stateChanged.connect(self.selectCheckBox_7)  
        self.checkBox_6.stateChanged.connect(self.selectCheckBox_6)
        self.checkBox_5.stateChanged.connect(self.selectCheckBox_5)
        self.checkBox_8.stateChanged.connect(self.selectCheckBox_8)
        self.checkBox_9.stateChanged.connect(self.selectCheckBox_7)
        self.checkBox_10.stateChanged.connect(self.selectCheckBox_6)
        self.checkBox_11.stateChanged.connect(self.selectCheckBox_5)
        self.checkBox_12.stateChanged.connect(self.selectCheckBox_8)
        self.checkBox.stateChanged.connect(self.selectCheckBox)
        self.checkBox_17.stateChanged.connect(self.selectCheckBox)
        self.checkBox_2.stateChanged.connect(self.selectCheckBox_2)
        self.checkBox_32.stateChanged.connect(self.selectCheckBox_2)
        self.checkBox_18.stateChanged.connect(self.firstcomp)
        self.checkBox_19.stateChanged.connect(self.firstcomp)
        self.checkBox_22.stateChanged.connect(self.secondcomp)
        self.checkBox_23.stateChanged.connect(self.secondcomp)
        self.checkBox_27.stateChanged.connect(self.messege_forUser)
        self.checkBox_26.stateChanged.connect(self.messege_forUser)
        self.checkBox_24.stateChanged.connect(self.messege_forUser)
        self.checkBox_25.stateChanged.connect(self.messege_forUser)
        self.checkBox_33.stateChanged.connect(self.messege_forUser)
        self.checkBox_34.stateChanged.connect(self.messege_forUser)
        self.checkBox_28.stateChanged.connect(self.secondcomp)
        self.checkBox_29.stateChanged.connect(self.secondcomp)
        self.checkBox_30.stateChanged.connect(self.secondcomp)
        self.checkBox_31.stateChanged.connect(self.secondcomp)
        self.horizontalSlider.valueChanged.connect(self.slidervaluechange1)
        self.horizontalSlider_2.valueChanged.connect(self.slidervaluechange2)
        self.checkBox_20.stateChanged.connect(self.Output)
        self.checkBox_21.stateChanged.connect(self.Output)

        self.image1Path : str = 'results/test.jpg'
        self.image2Path : str = 'results/test2.jpg'
        
        self.slidervaluechange1 =1
        self.slidervaluechange2 = 1

    def slidervaluechange1(self,value1):
        self.slidervaluechange1 = value1/100
        return (self.slidervaluechange1)
        
    
    def slidervaluechange2(self,value2):
        self.slidervaluechange2= value2/100
        return(self.slidervaluechange2)
       
    def Image1(self):
        self.label.setPixmap(QtGui.QPixmap(self.image1Path))
      
    def Image2(self):
        self.label_2.setPixmap(QtGui.QPixmap(self.image2Path))
        
    def messege_forUser(self):
        if self.checkBox_27.isChecked():
            self.label_6.setText( "Which  first component will you choose? 2nd comp can be the Imag ,phase or uni_phase")
        elif  self.checkBox_26.isChecked() :
            self.label_6.setText("Which  first component will you choose? 2nd comp can be the Real,mag or uni_mag")
        elif  self.checkBox_24.isChecked() :
            self.label_6.setText("Which  first component will you choose? 2nd comp can be the phase,uni_phase or imag")
        elif  self.checkBox_25.isChecked() :
            self.label_6.setText("Which  first component will you choose? 2nd comp can be the magnitude,uni_mag or real")
        elif  self.checkBox_33.isChecked() :
            self.label_6.setText("Which  first component will you choose? 2nd comp can be the phase,imag or Uni_phase")
        elif  self.checkBox_34.isChecked() :
            self.label_6.setText("Which  first component will you choose? 2nd comp can be the magnitude,real or uni_mag")
    def firstcomp(self):
            if self.checkBox_18.isChecked():
                self.image1=ImageModel(self.image1Path)
                return(self.image1)
                
                   
            elif self.checkBox_19.isChecked():
                 self.image2=ImageModel(self.image2Path)
                 return(self.image2)
                

    def secondcomp(self):
        if self.checkBox_22.isChecked():
                self.imageToBeMixed=ImageModel(self.image1Path)
                return (self.imageToBeMixed)
                
        elif self.checkBox_23.isChecked():
                self.imageToBeMixed=ImageModel(self.image2Path)
                return (self.imageToBeMixed)
                

         
    def selectCheckBox_7(self):                                # +++                                   # +++   
        if self.checkBox_9.isChecked() or self.checkBox_7.isChecked():
            if self.checkBox_7.isChecked():
                 self.image1=ImageModel(self.image1Path)
                 
                 plt.figure()
                 plt.subplot(121),plt.imshow(self.image1.imgByte, cmap = 'gray')
                 plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                 plt.subplot(122),plt.imshow(self.image1.real , cmap = 'gray')
                 #plt.subplot(122),plt.imshow(mix , cmap = 'gray')
                 plt.title('REAL Spectrum'), plt.xticks([]), plt.yticks([])
                 plt.show()
            
            elif self.checkBox_9.isChecked():
                self.image2=ImageModel(self.image2Path)
            
                self.image2.realfft()
                plt.figure()
                plt.subplot(121),plt.imshow(self.image2.imgByte, cmap = 'gray')
                plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(self.image2.real , cmap = 'gray')
                plt.title('Real Spectrum'), plt.xticks([]), plt.yticks([])
                plt.show()
       
    def selectCheckBox_6(self):                                # +++   
       if self.checkBox_6.isChecked() or self.checkBox_10.isChecked():
            if self.checkBox_6.isChecked():
                 self.image1=ImageModel(self.image1Path)
                 self.image1.imagfft()
                 plt.figure()
                 plt.subplot(121),plt.imshow(self.image1.imgByte, cmap = 'gray')
                 plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                 plt.subplot(122),plt.imshow(self.image1.imaginary , cmap = 'gray')
                 plt.title('Imaginary Spectrum'), plt.xticks([]), plt.yticks([])
                 plt.show()
            
            elif self.checkBox_10.isChecked():
                self.image2=ImageModel(self.image2Path)
            
                self.image2.imagfft()
                plt.figure()
                plt.subplot(121),plt.imshow(self.image2.imgByte, cmap = 'gray')
                plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(self.image2.imaginary , cmap = 'gray')
                plt.title('Imaginary Spectrum'), plt.xticks([]), plt.yticks([])
                plt.show()
    def selectCheckBox_5(self):                                # +++   
        if self.checkBox_5.isChecked() or self.checkBox_11.isChecked():
            if self.checkBox_5.isChecked():
                 self.image1=ImageModel(self.image1Path)
                 self.image1.magfft()
                 plt.figure()
                 plt.subplot(121),plt.imshow(self.image1.imgByte, cmap = 'gray')
                 plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                 plt.subplot(122),plt.imshow(self.image1.magnitude , cmap = 'gray')
                 plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
                 plt.show()
            
            elif self.checkBox_11.isChecked():
                self.image2=ImageModel(self.image2Path)
            
                self.image2.magfft()
                plt.figure()
                plt.subplot(121),plt.imshow(self.image2.imgByte, cmap = 'gray')
                plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(self.image2.magnitude , cmap = 'gray')
                plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
                plt.show()
    def selectCheckBox_8(self):                                # +++   
        if self.checkBox_8.isChecked() or self.checkBox_12.isChecked():
            if self.checkBox_8.isChecked():
                 self.image1=ImageModel(self.image1Path)
                 self.image1.phasefft()
                 plt.figure()
                 plt.subplot(121),plt.imshow(self.image1.imgByte, cmap = 'gray')
                 plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                 plt.subplot(122),plt.imshow(self.image1.phase , cmap = 'gray')
                 plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
                 plt.show()
            
            elif self.checkBox_12.isChecked():
                self.image2=ImageModel(self.image2Path)
            
                self.image2.phasefft()
                plt.figure()
                plt.subplot(121),plt.imshow(self.image2.imgByte, cmap = 'gray')
                plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(self.image2.phase , cmap = 'gray')
                plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
                plt.show()






    def selectCheckBox(self):                                # +++                                   # +++   
        if self.checkBox.isChecked() or self.checkBox_17.isChecked():
            if self.checkBox.isChecked():
                 self.image1=ImageModel(self.image1Path)
                 self.image1.uni_Magnitudefft()
                 plt.figure()
                 plt.subplot(121),plt.imshow(self.image1.imgByte, cmap = 'gray')
                 plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                 plt.subplot(122),plt.imshow(self.image1.unimagnitude , cmap = 'gray')
                 plt.title('Uni_magnitude Spectrum'), plt.xticks([]), plt.yticks([])
                 plt.show()
            
            elif self.checkBox_17.isChecked():
                self.image2=ImageModel(self.image2Path)
            
                self.image2.uni_Magnitudefft()
                plt.figure()
                plt.subplot(121),plt.imshow(self.image2.imgByte, cmap = 'gray')
                plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(self.image2.unimagnitude , cmap = 'gray')
                plt.title('Uni_magnitude Spectrum'), plt.xticks([]), plt.yticks([])
                plt.show()
    def selectCheckBox_2(self):                                # +++                                   # +++   
        if self.checkBox_2.isChecked() or self.checkBox_32.isChecked():
            if self.checkBox_2.isChecked():
                 self.image1=ImageModel(self.image1Path)
                 self.image1.uni_Phasefft
                 plt.figure()
                 plt.subplot(121),plt.imshow(self.image1.imgByte, cmap = 'gray')
                 plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                 plt.subplot(122),plt.imshow(self.image1.uniphase , cmap = 'gray')
                 plt.title('Uni_Phase Spectrum'), plt.xticks([]), plt.yticks([])
                 plt.show()
            
            elif self.checkBox_32.isChecked():
                self.image2=ImageModel(self.image2Path)
            
                self.image2.uni_Phasefft
                plt.figure()
                plt.subplot(121),plt.imshow(self.image2.imgByte, cmap = 'gray')
                plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(self.image2.uniphase , cmap = 'gray')
                plt.title('Uni_Phase Spectrum'), plt.xticks([]), plt.yticks([])
                plt.show()
   
    def Output(self):


        if self.checkBox_20.isChecked():
                 
                 self.firstimage=self.firstcomp()
                 self.secondimage=self.secondcomp()
                 self.magnitudeOrRealRatio=self.slidervaluechange1
                 self.phaesOrImaginaryRatio=self.slidervaluechange2
                 if self.checkBox_24.isChecked() or self.checkBox_25.isChecked() or self.checkBox_33.isChecked() or self.checkBox_34.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio, Modes.magnitudeAndPhase)
                 elif self.checkBox_33.isChecked() or self.checkBox_34.isChecked():
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio, Modes.Uni_magnitudeAndPhase)
                 elif self.checkBox_24.isChecked() or self.checkBox_34.isChecked():
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.magnitudeAndUNI_Phase)
                 elif self.checkBox_27.isChecked() or self.checkBox_26.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.realAndImaginary)
                 elif self.checkBox_27.isChecked() or self.checkBox_25.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.realAndphase)
                 elif self.checkBox_24.isChecked() or self.checkBox_26.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.magnitudeAndImaginary)
                 elif self.checkBox_27.isChecked() or self.checkBox_34.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.realAndUni_phase)
                 elif self.checkBox_33.isChecked() or self.checkBox_26.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.Uni_magAndImaginary)
                 mix=np.array(self.mix)
                 mix= np.fft.ifftshift(mix)

                 mix=np.fft.ifft2(mix)
                 mix=np.abs(mix)
                 plt.figure()
                 plt.imshow(mix, cmap = 'gray')
                 plt.title('  Real And IMaginary '), plt.xticks([]), plt.yticks([])
                 plt.show()
        if self.checkBox_21.isChecked():
                 self.firstimage=self.firstcomp()
                 self.secondimage=self.secondcomp()
                 self.magnitudeOrRealRatio=self.slidervaluechange1
                 self.phaesOrImaginaryRatio=self.slidervaluechange2
                 if self.checkBox_24.isChecked() or self.checkBox_25.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio, Modes.magnitudeAndPhase)
                 elif self.checkBox_33.isChecked() or self.checkBox_34.isChecked():
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.Uni_magnitudeAndPhase)
                 elif self.checkBox_24.isChecked() or self.checkBox_34.isChecked():
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.magnitudeAndUNI_Phase)
                 elif self.checkBox_25.isChecked() or self.checkBox_33.isChecked():
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.UNI_magnitudeAndPhase)
                 elif self.checkBox_27.isChecked() or self.checkBox_26.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.realAndImaginary)
                 elif self.checkBox_27.isChecked() or self.checkBox_25.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.realAndphase)
                 elif self.checkBox_24.isChecked() or self.checkBox_26.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.magnitudeAndImaginary)
                 elif self.checkBox_27.isChecked() or self.checkBox_34.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.realAndUni_phase)
                 elif self.checkBox_33.isChecked() or self.checkBox_26.isChecked() :
                     self.mix=self.firstimage.mix(self.secondimage,self.magnitudeOrRealRatio,self.phaesOrImaginaryRatio,Modes.Uni_magAndImaginary)
                 mix=np.array(self.mix)
                 mix= np.fft.ifftshift(mix)
                 mix=np.fft.ifft2(mix)
                 mix=np.abs(mix)
                 plt.figure()
                 plt.imshow(mix, cmap = 'gray')
                 plt.title('  Magnitude And Phase  '), plt.xticks([]), plt.yticks([])
                 plt.show()
      


            










app = QtWidgets.QApplication([])
window = mixer()
window.show()
app.exec_()


