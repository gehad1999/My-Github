from PyQt5 import QtWidgets
from PyQt5 import *
from img import Ui_MainWindow
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from copy import copy
from ImgClass import ImageModel
import cv2


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.img_1=''
        self.img_2 = ''
        self.ui.load1.triggered.connect(lambda: self.open_file(  0 , self.ui.graphicsView))



    def open_file(self, img_number, viewBox):

        self.file_name = QFileDialog().getOpenFileName(self,'Open file', '/home',"signals(*.png , *.jpg )")
        
        if self.file_name[0] != '' :
            if img_number == 0 :

                self.img_1 = copy(self.file_name[0]) 
                self.img1Obj = ImageModel(self.img_1)
                
                self.plot(self.img1Obj.imgByte, self.img1Obj.imgType ,viewBox )
                                
            else :

                pass
               

            
            self.pick_option()
            
            self.ui.comboBox.currentIndexChanged.connect(lambda: self.pick_option())
            

        else :
            
            pass
        

    def plot(self  , data , imgForm, widget):
      

        widget.canvas.figure.clear()

        widget.axes = widget.canvas.figure.add_subplot(111)

        if(imgForm == 'grayscale'):
            
            widget.axes.imshow(data, cmap = 'gray')

          
           
        else:
            

            widget.axes.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

        widget.canvas.figure.tight_layout()
  
        widget.canvas.draw()

    def draw_hough_lines(self  , data , imgForm, lines, widget):
        h , w = data.shape[0], data.shape[1]
        widget.canvas.figure.clear()

        widget.axes = widget.canvas.figure.add_subplot(111)

        for i in range(len(lines)):
            line = lines[i]
            widget.axes.plot([line[0] , line[1] ], [line[2] , line[3]] , color='g', linewidth=2)

        if(imgForm == 'grayscale'):
            
            widget.axes.imshow(data, cmap = 'gray')

          
           
        else:
            

            widget.axes.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

        widget.canvas.figure.tight_layout()
        widget.axes.set_xlim(0 ,w)
        widget.axes.set_ylim(h,0)
        widget.canvas.draw()

    def draw_hough_circles(self  , data , imgForm, circles, widget):
        widget.canvas.figure.clear()

        widget.axes = widget.canvas.figure.add_subplot(111)

        if(imgForm == 'grayscale'):
            
            widget.axes.imshow(data, cmap = 'gray')
          
        else:            

            widget.axes.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        for circle in circles:

            widget.axes.add_artist(circle)
        widget.canvas.figure.tight_layout()
        widget.canvas.draw()

    def draw_snake(self, image , new_snaxels, old_snaxels , widget):

        widget.canvas.figure.clear()

        widget.axes = widget.canvas.figure.add_subplot(111)
   
        for s in new_snaxels:
           widget.axes.plot(s[0],s[1],'y.',markersize=10.0)

        for s in old_snaxels:
            widget.axes.plot(s[0],s[1],'r.',markersize=10.0)
               
        widget.axes.imshow(image ,cmap = 'gray')
        widget.canvas.draw()



    def pick_option(self):
        
        if self.img_1 != '' :

            if self.ui.comboBox.currentText() == "Canny edge detection"  :
                output_image = self.img1Obj.canny_edge_detection()
                self.plot(output_image , 'grayscale', self.ui.graphicsView_2)

            elif self.ui.comboBox.currentText() == "Detect lines"  :
                lines = self.img1Obj.hough_transform_lines()
                self.draw_hough_lines(self.img1Obj.imgByte , self.img1Obj.imgType,lines ,self.ui.graphicsView_2)       
  
            elif self.ui.comboBox.currentText() == "Detect circles"  :
                circles = self.img1Obj.hough_transform_circles()
                self.draw_hough_circles(self.img1Obj.imgByte , self.img1Obj.imgType,circles ,self.ui.graphicsView_2)
           
            elif self.ui.comboBox.currentText() == "Snakes"  :
                new_snaxels , old_snaxels = self.img1Obj.activeContourFromCircle()
                self.draw_snake(self.img1Obj.imageSnake , new_snaxels , old_snaxels,self.ui.graphicsView_2 )
        
  


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()