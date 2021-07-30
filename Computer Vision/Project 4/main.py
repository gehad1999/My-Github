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
        self.ui.comboBox.currentIndexChanged.connect(lambda: self.pick_option())



    def open_file(self, img_number, viewBox):

        self.file_name = QFileDialog().getOpenFileName(self,'Open file', '/home',"signals(*.png , *.jpg )")
        
        if self.file_name[0] != '' :
            if img_number == 0 :

                self.img_1 = copy(self.file_name[0]) 
                self.img1Obj = ImageModel(self.img_1)
                
                self.plot(self.img1Obj.imgByte, self.img1Obj.imgType ,viewBox )
               
                
                
            else :

                pass
            

        else :
            
            pass

        if(self.img_1 != '' ):
            self.pick_option()
            
            
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

            elif self.ui.comboBox.currentText() == "Kmeans"  :
                        
                output = self.img1Obj.Kmeans()
                self.plot(output , 'anyform', self.ui.graphicsView_2)

            elif self.ui.comboBox.currentText() == "Multilevel_otsu"  :
                        
                output = self.img1Obj.Otsu()
                self.plot(output , 'grayscale', self.ui.graphicsView_2)

            elif self.ui.comboBox.currentText() == "Multilevel_otsu_local"  :
                        
                output = self.img1Obj.Otsu_local()
                self.plot(output , 'grayscale', self.ui.graphicsView_2)

            elif self.ui.comboBox.currentText() == "Optimal Thresholding"  :
                        
                output = self.img1Obj.optimal_thresholding()
                self.plot(output , 'grayscale', self.ui.graphicsView_2)

                          
            elif self.ui.comboBox.currentText() == "Optimal Thresholding local"  :
                        
                output = self.img1Obj.optimal_thresholding_local()
                self.plot(output , 'grayscale', self.ui.graphicsView_2)

                          
            elif(self.ui.comboBox.currentText() == "Mean Shift"):
                output = self.img1Obj.mean_shift()
                self.plot(output , 'anyform', self.ui.graphicsView_2)
            
            elif(self.ui.comboBox.currentText() == "Region Growing"):
                output = self.img1Obj.regionGrowing()
                self.plot(output , 'anyform', self.ui.graphicsView_2)
            
            elif(self.ui.comboBox.currentText() == "Agglomerative"):
                output = self.img1Obj.agglomerative()
                self.plot(output , 'anyform', self.ui.graphicsView_2)
            
                # output = self.img1Obj.Harris_SIFT(self.img2Obj.grayscale, self.img2Obj.imgByte, 'normalized cross correlation')
                # self.plot(output , 'anyform', self.ui.graphicsView_4)
                        

                        
           
        
  


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()