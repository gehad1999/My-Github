from PyQt5 import QtWidgets
from PyQt5 import *
from img import Ui_MainWindow
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from copy import copy
# from modesEnum import Modes
from ImgClass import ImageModel
# import functions as f
import cv2


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.img_1=''
        self.img_2 = ''
        self.ui.load1.triggered.connect(lambda: self.open_file(  0 , self.ui.graphicsView))
        self.ui.load2.triggered.connect(lambda: self.open_file( 1, self.ui.graphicsView_2))



    def open_file(self, img_number, viewBox):

        self.file_name = QFileDialog().getOpenFileName(self,'Open file', '/home',"signals(*.png , *.jpg )")
        
        if self.file_name[0] != '' :
            if img_number == 0 :

                self.img_1 = copy(self.file_name[0]) 
                self.img1Obj = ImageModel(self.img_1)
                
                self.plot(self.img1Obj.imgByte, self.img1Obj.imgType ,viewBox )
               
                
                
            else :

                self.img_2 = copy(self.file_name[0]) 
                self.img2Obj = ImageModel(self.img_2)

                self.plot(self.img2Obj.imgByte, self.img2Obj.imgType ,viewBox )
                
               

            
            # self.pick_option()
            
            # self.ui.comboBox.currentIndexChanged.connect(lambda: self.pick_option())
            

        else :
            
            pass

        if(self.img_1 != '' and self.img_2 != ''):
            self.pick_option()
            
            self.ui.comboBox.currentIndexChanged.connect(lambda: self.pick_option())
        else:
            pass




        
    

    def plot(self  , data , imgForm, widget):
      

        widget.canvas.figure.clear()

        widget.axes = widget.canvas.figure.add_subplot(111)

        if(imgForm == 'grayscale'):
            
            widget.axes.imshow(data, cmap = 'gray')

          
           
        elif(imgForm == 'RGB'):
            

            widget.axes.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        else:
            widget.axes.imshow(data)

        widget.canvas.figure.tight_layout()  
        widget.canvas.draw()

    

    def pick_option(self):
        
        if self.img_1 != '' :

            if (self.ui.comboBox.currentText() == ''):
                pass

            elif self.ui.comboBox.currentText() == "Matching with SSD"  :
                if(self.img_2 == ''):

                    QMessageBox.about(self,"Warning","please insert the second image" )
                else:
                    output = self.img1Obj.Harris_SIFT(self.img2Obj.grayscale, self.img2Obj.imgByte, 'SSD')
                    self.plot(output , 'anyform', self.ui.graphicsView_4)

                          
            elif(self.ui.comboBox.currentText() == "Matching with normalized cross correlation"):
                if(self.img_2 == ''):

                    QMessageBox.about(self,"Warning","please insert the second image" )
                else:

                    output = self.img1Obj.Harris_SIFT(self.img2Obj.grayscale, self.img2Obj.imgByte, 'normalized cross correlation')
                    self.plot(output , 'anyform', self.ui.graphicsView_4)
                        

                        
           
        
  


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()