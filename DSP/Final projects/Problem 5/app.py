from PyQt5 import QtWidgets
from Qt import Ui_MainWindow
from PyQt5.QtWidgets import QGraphicsScene
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread ,pyqtSignal
import os 
import time
from PyQt5.QtGui import QPixmap, QImage ,QMovie
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.gg=None
        self.i=0
        self.h=0
       
        self.ui.WithMask.clicked.connect(lambda : self.Simulation(self.ui.WithMask))
        self.ui.WithOutMask.clicked.connect(lambda : self.Simulation(self.ui.WithOutMask))
        self.ui.horizontalSlider.valueChanged.connect(self.slider)   

    def Display(self,Path):
        movie = QMovie(Path)
        self.ui.label.setMovie(movie)
        movie.start() 
    

    def Simulation(self,Button):
        
        if Button == self.ui.WithOutMask:
            if self.ui.checkBox.isChecked():
                self.Display("coronalWithOutMask.gif")
            elif self.ui.checkBox_2.isChecked():
                self.Display('bandicam_withOutMask.gif')
            else:
                self.showMessageBox('Warning','You must choose From ComBox' )     
            
        elif Button==self.ui.WithMask:
            if self.ui.checkBox.isChecked():
               self.Display("coronalWithMask.gif")
            elif self.ui.checkBox_2.isChecked():
                self.Display("bandicam_withMask.gif")
            else:
                self.showMessageBox('Warning','You must choose Plane  From CheckBox' )       


    def showMessageBox(self,title,message):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setWindowTitle(title)
        msgBox.setText(message)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.exec_()


    def slider(self):
        size=self.ui.horizontalSlider.value()
        if size <10 :
          self.Plot_data(0.1)
        elif size >10 and size <20  :
            self.Plot_data(0.8)
        elif size >20 and size <30  :
            self.Plot_data(0.7)
        elif size >30 and size <40  :
            self.Plot_data(0.6)
        elif size >40 and size <50  :
            self.Plot_data(0.5)
        elif size >50 and size <60  :
            self.Plot_data(0.4)
        elif size >60 and size <70  :
            self.Plot_data(0.3)
        elif size >70 and size <80  :
            self.Plot_data(0.2)
        elif size >80 and size <90  :
            self.Plot_data(0.1)
        elif size >90  :
            self.Plot_data(1)    
    def Plot_data(self,ratio):
        data = ratio*(np.random.rayleigh(10,80))
        sort = np.sort(data)[::-1]                                            
        time = np.arange(1.,len(sort)+1) / len(sort)                           
        plt.plot(time, sort)
        plt.xlabel("time")
        plt.ylabel("strength")
        plt.savefig('plot.png')
        self.image=QPixmap('plot.png')
        self.ui.label_2.setScaledContents(True)
        self.ui.label_2.setPixmap(self.image)
   
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()