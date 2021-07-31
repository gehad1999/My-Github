import sys
import matplotlib.pyplot as plt
from PyQt5 import uic,QtWidgets

from PyQt5.QtWidgets import*
from PyQt5.QtWidgets import QMainWindow
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout,QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

import os
import wave
import numpy as np
import winsound
import soundfile as sn
from scipy import signal
from time import sleep
from scipy.io.wavfile import read
import sounddevice as sd
import math as pis
from scipy.io.wavfile import write



class Equalizer(QMainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self   )
        super(Equalizer, self).__init__(self)
        uic.loadUi("qt.ui",self)
        self.setWindowTitle("PyQt5 & Matplotlib Example GUI")
        #########################Check Box################################
        self.checkBox.setChecked(False)
        self.checkBox.stateChanged.connect(self.onClicked1)
        self.checkBox_2.setChecked(False)
        self.checkBox_2.toggled.connect(self.onClicked1)
        self.checkBox_3.setChecked(False)
        self.checkBox_3.toggled.connect(self.onClicked1)
        ####################### Initialization variables###################
        self.increment = 0
        self.controlled=0 # control length of hamming/ hanning
        self.HammFlag=0
        self.HannFlag=0
        self.RecFlag=0
        self.data_wav=[]
        self.F_signal=[]
        self.split_signal=[]
        self.abs_parts=[]
        self.arr_absBand_1=[]
        self.arr_absBand_2=[]
        self.arr_absBand_3=[]
        self.arr_absBand_4=[]
        self.arr_absBand_5=[]
        self.arr_absBand_6=[]
        self.arr_absBand_7=[]
        self.arr_absBand_8=[]
        self.arr_absBand_9=[]
        self.arr_absBand_10=[]
        self.conctenate=[]
        self.mul_sig_1=self.arr_absBand_1
        self.mul_sig_2=self.arr_absBand_2
        self.mul_sig_3=self.arr_absBand_3
        self.mul_sig_4=self.arr_absBand_4
        self.mul_sig_5=self.arr_absBand_5
        self.mul_sig_6=self.arr_absBand_6
        self.mul_sig_7=self.arr_absBand_7
        self.mul_sig_8=self.arr_absBand_8
        self.mul_sig_9=self.arr_absBand_9
        self.mul_sig_10=self.arr_absBand_10
        self.response=[]
        self.Inverse_freq=[]
        self.abs_inv=[]
        self.name=[]
        ###################### Button ########################################
        self.Browse.clicked.connect(self.open_file)
        self.Play.clicked.connect(self.Start_timer)
        self.signal_after_fft.clicked.connect(self.FFT_signal)
        self.inverse.clicked.connect(self. Start_timer_inverse)
        self.All_Bands.clicked.connect(self.Plot_All_Bands)
        self.SAVE.clicked.connect(self.export)
        self.Difference.clicked.connect(self.Diff)
        ###################### Slider ########################################
        self.slider=[self.verticalSlider_9,self.verticalSlider_8,self.verticalSlider_7,self.verticalSlider_6,self.verticalSlider_5,self.verticalSlider_4,self.verticalSlider_3,self.verticalSlider_2,self.verticalSlider_1,self.verticalSlider_0]
        self.sliderFUN=[self.valuechange_9,self.valuechange_8,self.valuechange_7,self.valuechange_6,self.valuechange_5,self.valuechange_4,self.valuechange_3,self.valuechange_2,self.valuechange_1,self.valuechange_0]
        for i in range(len(self.slider)):
            self.slider[i].valueChanged.connect(self.sliderFUN[i])
         ############################          Check Box         ##########################
    def onClicked1(self):
        if (self.checkBox.isChecked()):
            self.HammFlag=1 
        else:
            self.HammFlag=0
        if (self.checkBox_2.isChecked()):
             self.HannFlag=1 
        else:
            self.HannFlag=0
        if(self.checkBox_3.isChecked()):
            self.RecFlag=1 
        else:
            self.RecFlag=0 
        ################################## Time Signal ########################################
    def open_file(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath() , '*.wav')       
        [self.data_wav,fs] = sn.read(fileName,dtype='int16')

    def Start_timer(self):
        #fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath() , '*.wav')       
        #[self.data_wav,fs] = sn.read(fileName,dtype='int16')
        self.timer3 = pg.QtCore.QTimer()
        self.timer3.timeout.connect(self.moveplot3)
        self.timer3.start() 
    def moveplot3(self):
         self.increment+=1000
         self.update_figure()
    def update_figure(self):
        self.graphicsView.plot(self.data_wav[:self.increment])
    def SOUND_play(self):
       sd.play(self.data_wav[self.increment:],self.fs)    

    ######################### signal in fourier ###############################################################################
    def FFT_signal(self):
        self.F_signal=np.fft.fft(self.data_wav)
        self.mag_sig=np.abs(self.F_signal)
        self.mag_sig[30000:]=0
        self.graphicsView.plot(self.mag_sig) 

    ############################################# split signal to 10 Bands ######################################################
    def BandWidth(self):
        self.graphicsView.clear()
        self.split_signal=np.array_split(self.F_signal,10)
        self.abs_parts=np.abs(self.split_signal)
        self.arr_absBand_1=np.array(self.abs_parts[0])
        self.arr_absBand_2=np.array(self.abs_parts[1])
        self.arr_absBand_3=np.array(self.abs_parts[2])
        self.arr_absBand_4=np.array(self.abs_parts[3])
        self.arr_absBand_5=np.array(self.abs_parts[4])
        self.arr_absBand_6=np.array(self.abs_parts[5])
        self.arr_absBand_7=np.array(self.abs_parts[6])
        self.arr_absBand_8=np.array(self.abs_parts[7])
        self.arr_absBand_9=np.array(self.abs_parts[8])
        self.arr_absBand_10=np.array(self.abs_parts[9])
 

     ##################################### PLOT ALL BANDS #################################################################
    def Plot_All_Bands(self):
        self.FFT_signal()
        self.BandWidth()
        plt.subplot(521)
        plt.plot(self.mul_sig_10,color='red')
        plt.subplot(522)
        plt.plot(self.mul_sig_9,color='yellow')
        plt.subplot(523)
        plt.plot(self.mul_sig_8,color='black')
        plt.subplot(524)
        plt.plot(self.mul_sig_7,color='blue')
        plt.subplot(525)
        plt.plot(self.mul_sig_6,color='pink')
        plt.subplot(526)
        plt.plot(self.mul_sig_5,color='green')
        plt.subplot(527)
        plt.plot(self.mul_sig_4,color='brown')
        plt.subplot(528)
        plt.plot(self.mul_sig_3,color='red')
        plt.subplot(529)
        plt.plot(self.mul_sig_2,color='black')
        plt.show()




    ############################################### Hamming #################################################################    
    def humming_func (self,n):
        if (self.controlled==0):
            self.humming=np.hamming( len(self.arr_absBand_10))
        elif (self.controlled==1):
            self.humming=np.hamming( len(self.arr_absBand_9))
        elif (self.controlled==2):
            self.humming=np.hamming( len(self.arr_absBand_8))
        elif (self.controlled==3):
                self.humming=np.hamming( len(self.arr_absBand_7))
        elif (self.controlled==4):
                self.humming=np.hamming( len(self.arr_absBand_6))
        elif (self.controlled==5):
                self.humming=np.hamming( len(self.arr_absBand_5))
        elif (self.controlled==6):
                self.humming=np.hamming( len(self.arr_absBand_4))
        elif (self.controlled==7):
                self.humming=np.hamming( len(self.arr_absBand_3))
        elif (self.controlled==8):
                self.humming=np.hamming( len(self.arr_absBand_2))
        elif (self.controlled==9):
            self.humming=np.hamming( len(self.arr_absBand_1))

        self.fourier_hamming=np.fft.fft(self.humming)
        self.abs= np.abs(self.fourier_hamming)
        self.gain =n * np.log10(self.abs)
        self.arr_ham=np.array(self.gain) 
    ############################################# Hanning #####################################################################
    def HanningFunc(self,Hn_gain):
        if (self.controlled==0):
                self.Hanning = np.hanning(len(self.arr_absBand_10))
        elif (self.controlled==1):
            self.Hanning = np.hanning(len(self.arr_absBand_9))
        elif (self.controlled==2):
            self.Hanning = np.hanning(len(self.arr_absBand_8))
        elif (self.controlled==3):
                self.Hanning = np.hanning(len(self.arr_absBand_7))
        elif (self.controlled==4):
                self.Hanning = np.hanning(len(self.arr_absBand_6))
        elif (self.controlled==5):
               self.Hanning = np.hanning(len(self.arr_absBand_5))
        elif (self.controlled==6):
               self.Hanning = np.hanning(len(self.arr_absBand_4))
        elif (self.controlled==7):
                self.Hanning = np.hanning(len(self.arr_absBand_3))
        elif (self.controlled==8):
                self.Hanning = np.hanning(len(self.arr_absBand_2))
        elif (self.controlled==9):
            self.Hanning = np.hanning(len(self.arr_absBand_1))
        self.F_Hann=np.fft.fft(self.Hanning)
        self.abs_Hann=np.abs(np.fft.fftshift(self.F_Hann))
        self.HannGain=Hn_gain*np.log10(self.abs_Hann)
        self.arr_han=np.array(self.HannGain)      
     ############################################# Rectangular #####################################################################
    
    def RectangularFunc(self,Rec_gain):
        if (self.controlled==0):
             self.rectangular=signal.boxcar(2*len(self.arr_absBand_10))
        elif (self.controlled==1):
            self.rectangular=signal.boxcar(len(self.arr_absBand_9))
        elif (self.controlled==2):
           self.rectangular=signal.boxcar(len(self.arr_absBand_8))
        elif (self.controlled==3):
            self.rectangular=signal.boxcar(len(self.arr_absBand_7))
        elif (self.controlled==4):
            self.rectangular=signal.boxcar(len(self.arr_absBand_6))
        elif (self.controlled==5):
            self.rectangular=signal.boxcar(len(self.arr_absBand_5))
        elif (self.controlled==6):
             self.rectangular=signal.boxcar(len(self.arr_absBand_4))
        elif (self.controlled==7):
             self.rectangular=signal.boxcar(len(self.arr_absBand_3))
        elif (self.controlled==8):
             self.rectangular=signal.boxcar(len(self.arr_absBand_2))
        elif (self.controlled==9):
            self.rectangular=signal.boxcar(len(self.arr_absBand_1))

        self.rectangular=signal.boxcar(len(self.arr_absBand_1))
        self.F_Rec = np.fft.fft(self.rectangular) / (len(self.rectangular)/2.0)
        self.abs_Rec=np.abs(np.fft.fftshift(self.F_Rec / abs(self.F_Rec).max()))
        self.RecGain = Rec_gain * np.log10( self.abs_Rec)
        self.arr_rec=np.array(self.RecGain)     
   
        ###########################################  Slider  ######################################     
    
    def valuechange_9(self):
    
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=9
           if (size>=0 and size<=12) :
               self.gs1=0.5
               self.PLOT_band_1() 
               
           if (size>=13 and size<=38) :
               self.gs1=25
               self.PLOT_band_1() 
           if (size>=39 and size<=63) :
               self.gs1=50
               self.PLOT_band_1() 
           if (size>=64 and size<=87) :
               self.gs1=75
               self.PLOT_band_1() 
           if (size>=88 and size<=100) :
               self.gs1=100
               self.PLOT_band_1() 
    def valuechange_8(self):
        
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=8
           if (size>=0 and size<=12) :
               self.gs2=0.5
               self.PLOT_band_2()  
               
           if (size>=13 and size<=38) :
               self.gs2=25
               self.PLOT_band_2()  
           if (size>=39 and size<=63) :
               self.gs2=50
               self.PLOT_band_2()  
           if (size>=64 and size<=87) :
               self.gs2=75
               self.PLOT_band_2()  
           if (size>=88 and size<=100) :
               self.gs2=100
               self.PLOT_band_2()  
    def valuechange_7(self):
        
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=7
           if (size>=0 and size<=12) :
               self.gs3=0.5
               self.PLOT_band_3()  
               
           if (size>=13 and size<=38) :
               self.gs3=25
               self.PLOT_band_3()  
           if (size>=39 and size<=63) :
               self.gs3=50
               self.PLOT_band_3() 
           if (size>=64 and size<=87) :
               self.gs3=75
               self.PLOT_band_3()  
           if (size>=88 and size<=100) :
               self.gs3=100
               self.PLOT_band_3()  
    def valuechange_6(self):
        
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=6
           if (size>=0 and size<=12) :
               self.gs4=0.5
               self.PLOT_band_4() 
               
           if (size>=13 and size<=38) :
               self.gs4=25
               self.PLOT_band_4() 
           if (size>=39 and size<=63) :
               self.gs4=50
               self.PLOT_band_4() 
           if (size>=64 and size<=87) :
               self.gs4=75
               self.PLOT_band_4()  
           if (size>=88 and size<=100) :
               self.gs4=100
               self.PLOT_band_4()    
    def valuechange_5(self):
        
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=5
           if (size>=0 and size<=12) :
               self.gs5=0.5
               self.PLOT_band_5() 
               
           if (size>=13 and size<=38) :
               self.gs5=25
               self.PLOT_band_5() 
           if (size>=39 and size<=63) :
               self.gs5=50
               self.PLOT_band_5() 
           if (size>=64 and size<=87) :
               self.gs5=75
               self.PLOT_band_5() 
           if (size>=88 and size<=100) :
               self.gs5=100
               self.PLOT_band_5() 
    def valuechange_4(self):
        
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=4
           if (size>=0 and size<=12) :
               self.gs6=0.5
               self.PLOT_band_6() 
               
           if (size>=13 and size<=38) :
               self.gs6=25
               self.PLOT_band_6() 
           if (size>=39 and size<=63) :
               self.gs6=50
               self.PLOT_band_6() 
           if (size>=64 and size<=87) :
               self.gs6=75
               self.PLOT_band_6() 
           if (size>=88 and size<=100) :
               self.gs6=100
               self.PLOT_band_6()                                    
    def valuechange_3(self):
        
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=3
           if (size>=0 and size<=12) :
               self.gs7=0.5
               self.PLOT_band_7() 
               
           if (size>=13 and size<=38) :
               self.gs7=25
               self.PLOT_band_7() 
           if (size>=39 and size<=63) :
               self.gs7=50
               self.PLOT_band_7()  
           if (size>=64 and size<=87) :
               self.gs7=75
               self.PLOT_band_7() 
           if (size>=88 and size<=100) :
               self.gs7=100
               self.PLOT_band_7() 
    def valuechange_2(self):
        
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=2
           if (size>=0 and size<=12) :
               self.gs8=0.5
               self.PLOT_band_8() 
               
           if (size>=13 and size<=38) :
               self.gs8=25
               self.PLOT_band_8() 
           if (size>=39 and size<=63) :
               self.gs8=50
               self.PLOT_band_8() 
           if (size>=64 and size<=87) :
               self.gs8=75
               self.PLOT_band_8() 
           if (size>=88 and size<=100) :
               self.gs8=100
               self.PLOT_band_8() 
    def valuechange_1(self):
        
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=1
           if (size>=0 and size<=12) :
               self.gs9=0.5
               self.PLOT_band_9() 
               
           if (size>=13 and size<=38) :
               self.gs9=25
               self.PLOT_band_9() 
           if (size>=39 and size<=63) :
               self.gs9=50
               self.PLOT_band_9() 
           if (size>=64 and size<=87) :
               self.gs9=75
               self.PLOT_band_9() 
           if (size>=88 and size<=100) :
               self.gs9=100
               self.PLOT_band_9() 
    def valuechange_0(self):
        
           self.graphicsView.clear()
           size = self.verticalSlider_9.value()
           self.controlled=0
           if (size>=0 and size<=12) :
               self.gs10=0.5
               self.PLOT_band_10() 
               
           if (size>=13 and size<=38) :
               self.gs10=25
               self.PLOT_band_10() 
           if (size>=39 and size<=63) :
               self.gs10=50
               self.PLOT_band_10() 
           if (size>=64 and size<=87) :
               self.gs10=75
               self.PLOT_band_10() 
           if (size>=88 and size<=100) :
               self.gs10=100
               self.PLOT_band_10() 
    #################################################  PLOT #######################################################################
    def PLOT_band_1(self):
        self.BandWidth()
        self.humming_func(self.gs1)
        self.HanningFunc(self.gs1)
        self.RectangularFunc(self.gs1)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_1=np.abs(self.arr_absBand_1* self.arr_ham )
            self.band_1=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_1) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_1=np.abs(self.arr_absBand_1* self.arr_han)
            self.band_1=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_1) 
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_1=np.abs( self.arr_absBand_1 * self.arr_rec)
            self.band_1=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_1) 
     ######################## plot_band_2  ######################################### 
    def PLOT_band_2(self):
        self.BandWidth()
        self.humming_func(self.gs2)
        self.HanningFunc(self.gs2)
        self.RectangularFunc(self.gs2)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_2=np.abs(self.arr_absBand_2* self.arr_ham )
            self.band_2=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_2) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_2=np.abs(self.arr_absBand_2* self.arr_han)
            self.band_2=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_2) 
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_2=np.abs( self.arr_absBand_2 * self.arr_rec)
            self.band_2=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_2)      
########################################### plot band 3 #####################################################################
    def PLOT_band_3(self):
        self.BandWidth()
        self.humming_func(self.gs3)
        self.HanningFunc(self.gs3)
        self.RectangularFunc(self.gs3)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_3=np.abs(self.arr_absBand_3* self.arr_ham )
            self.band_3=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_3) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_3=np.abs(self.arr_absBand_3* self.arr_han)
            self.band_3=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_3) 
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_3=np.abs( self.arr_absBand_3 * self.arr_rec)
            self.band_3=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_3) 

########################################### plot band 4 #####################################################################
    def PLOT_band_4(self):
        self.BandWidth()
        self.humming_func(self.gs4)
        self.HanningFunc(self.gs4)
        self.RectangularFunc(self.gs4)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_4=np.abs(self.arr_absBand_4* self.arr_ham )
            self.band_4=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_4) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_4=np.abs(self.arr_absBand_4* self.arr_han)
            self.band_4=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_4) 
            
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_4=np.abs( self.arr_absBand_4 * self.arr_rec)
            self.band_4=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_4) 
########################################### plot band 5 #####################################################################
    def PLOT_band_5(self):
        self.BandWidth()
        self.humming_func(self.gs5)
        self.HanningFunc(self.gs5)
        self.RectangularFunc(self.gs5)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_5=np.abs(self.arr_absBand_5* self.arr_ham )
            self.band_5=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_5) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_5=np.abs(self.arr_absBand_5* self.arr_han)
            self.band_5=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_5) 
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_5=np.abs( self.arr_absBand_5 * self.arr_rec)
            self.band_5=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_5) 
########################################### plot band 6 #####################################################################
    def PLOT_band_6(self):
        self.BandWidth()
        self.humming_func(self.gs6)
        self.HanningFunc(self.gs6)
        self.RectangularFunc(self.gs6)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_6=np.abs(self.arr_absBand_6* self.arr_ham )
            self.band_6=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_6) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_6=np.abs(self.arr_absBand_6* self.arr_han)
            self.band_6=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_6) 
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_6=np.abs( self.arr_absBand_6 * self.arr_rec)
            self.band_6=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_6) 
########################################### plot band 7 #####################################################################
    def PLOT_band_7(self):
        self.BandWidth()
        self.humming_func(self.gs7)
        self.HanningFunc(self.gs7)
        self.RectangularFunc(self.gs7)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_7=np.abs(self.arr_absBand_7* self.arr_ham )
            self.band_7=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_7) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_7=np.abs(self.arr_absBand_7* self.arr_han)
            self.band_7=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_7) 
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_7=np.abs( self.arr_absBand_7 * self.arr_rec)
            self.band_7=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_7) 
   
 ########################################### plot band 8 #####################################################################
    def PLOT_band_8(self):
        self.BandWidth()
        self.humming_func(self.gs8)
        self.HanningFunc(self.gs8)
        self.RectangularFunc(self.gs8)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_8=np.abs(self.arr_absBand_8* self.arr_ham )
            self.band_8=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_8) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_8=np.abs(self.arr_absBand_8* self.arr_han)
            self.band_8=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_8) 
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_8=np.abs( self.arr_absBand_8 * self.arr_rec)
            self.band_8=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_8) 

########################################### plot band 9 #####################################################################
    def PLOT_band_9(self):
        self.BandWidth()
        self.humming_func(self.gs9)
        self.HanningFunc(self.gs9)
        self.RectangularFunc(self.gs9)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_9=np.abs(self.arr_absBand_9* self.arr_ham )
            self.band_9=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_9) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_9=np.abs(self.arr_absBand_9* self.arr_han)
            self.band_9=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_9) 
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_9=np.abs( self.arr_absBand_9 * self.arr_rec)
            self.band_9=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_9) 
    ########################################### plot band 10 #####################################################################
    def PLOT_band_10(self):
        self.BandWidth()
        self.humming_func(self.gs10)
        self.HanningFunc(self.gs10)
        self.RectangularFunc(self.gs10)
        if (self.HammFlag==1):
            self.HannFlag=0
            self.RecFlag=0
            self.mul_sig_10=np.abs(self.arr_absBand_10* self.arr_ham )
            self.band_10=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_10) 
        elif(self.HannFlag==1):
            self.HammFlag=0
            self.RecFlag=0
            self.mul_sig_10=np.abs(self.arr_absBand_10* self.arr_han)
            self.band_10=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_10) 
        elif(self.RecFlag==1): 
            self.HammFlag=0
            self.HannFlag=0
            self.mul_sig_10=np.abs( self.arr_absBand_10 * self.arr_rec)
            self.band_10=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
            self.graphicsView.plot(self.band_10) 
        
          ################################# Time Signal ########################################
    def Start_timer_inverse(self):
        self.FFT_signal()
        self.BandWidth()
        self.conctenate=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
        self.Inverse_freq=np.fft.ifft(self.conctenate)
        self.abs_inv=np.abs(self.Inverse_freq)
        self.timer3 = pg.QtCore.QTimer()
        self.timer3.timeout.connect(self.moveplot_inverse)
        self.timer3.start() 
    def moveplot_inverse(self):
         self.increment+=1000
         self.update_figure_inverse()
    def update_figure_inverse(self):
        self.graphicsView.plot(self.abs_inv[:self.increment])
    def SOUND_play(self):
       sd.play(self.abs_inv[self.increment:],self.fs)   
    def export(self):
        scaled=np.int16(self.abs_inv/np.max(np.abs(self.abs_inv))*32767)
        name = QtWidgets.QFileDialog.getSaveFileName(self, "open","",'Track(*.wav)')
        write(name,22050,scaled)

    def Diff (self):
         self.BandWidth()
         self.conctenate=np.concatenate([self.mul_sig_1,self.mul_sig_2,self.mul_sig_3,self.mul_sig_4,self.mul_sig_5,self.mul_sig_6,self.mul_sig_7,self.mul_sig_8,self.mul_sig_9,self.mul_sig_10])
         self.Inverse_freq=np.fft.ifft(self.conctenate)
         plt.subplot(211)
         plt.plot(self.F_signal,color='red')
         plt.subplot(212)
         plt.plot(self.Inverse_freq,color='green')
         plt.show()

app = QtWidgets.QApplication([])
window = Equalizer()
window.show()
app.exec_()

          