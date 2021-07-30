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
                self.check(self.img1Obj.imgType)
                
                
            else :

                self.img_2 = copy(self.file_name[0]) 
                self.img2Obj = ImageModel(self.img_2)

                self.plot(self.img2Obj.imgByte, self.img2Obj.imgType ,viewBox )
                self.check(self.img2Obj.imgType)
               

            
            self.pick_option()
            
            self.ui.comboBox.currentIndexChanged.connect(lambda: self.pick_option())
            

        else :
            
            pass
        
    def check(self, imgForm):  
        index = self.ui.comboBox.findText("Convert to Gray scale") 

        if(imgForm == 'grayscale'):
            
            self.ui.comboBox.model().item(index).setEnabled(False)
        else:
            self.ui.comboBox.model().item(index).setEnabled(True)


    def plot(self  , data , imgForm, widget):
      

        widget.canvas.figure.clear()

        widget.axes = widget.canvas.figure.add_subplot(111)

        if(imgForm == 'grayscale'):
            
            widget.axes.imshow(data, cmap = 'gray')

          
           
        else:
            

            widget.axes.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

          
        widget.canvas.draw()

    def plot_histo(self  , data , imgForm, widget):

        x_axis = np.arange(256)
        
       
        
        widget.canvas.figure.clear()

        widget.axes = widget.canvas.figure.add_subplot(111)
       

        if(imgForm == 'grayscale'):
            
            widget.axes.plot(x_axis, data  )
           

        else:
            widget.axes.plot(x_axis, data[0] )
            widget.axes.plot(x_axis, data[1] , 'g')
            widget.axes.plot(x_axis, data[2] , 'r')
            

      

        widget.canvas.draw()

    def pick_option(self):
        
        if self.img_1 != '' :

            if self.ui.comboBox.currentText() == "Histogram equalization"  :
                output_image = self.img1Obj.Histogram_Equalization()
                self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

                

            elif(self.ui.comboBox.currentText() == "Image normalization"):
                output_image = self.img1Obj.normalize_img()

                self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Frequency domain Low pass filter"):
                output_image = self.img1Obj.freq_lowpass_Filter()

                self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Frequency domain High pass filter"):
                output_image = self.img1Obj.freq_highpass_Filter()

                self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Convert to Gray scale"):
                output_image = self.img1Obj.to_grayScale()

                self.plot(output_image , 'grayscale', self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Global thresholding"):
                output_image = self.img1Obj.otsu()

                self.plot(output_image , 'grayscale', self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Local thresholding"):
                output_image = self.img1Obj.contrast_threshold()

                self.plot(output_image , 'grayscale', self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Histogram & Distribution curve"):
                output_image = self.img1Obj.get_Histogram()
                self.plot_histo(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)
                # print(output_image)

            elif(self.ui.comboBox.currentText() == "Salt & pepper noise"):
                output_image = self.img1Obj.salt_and_pepper_noise()
                self.img1Obj.noised_img = copy(output_image)
                # cv2.imwrite("noised_image.png", output_image)

                self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_4)

            elif(self.ui.comboBox.currentText() == "Median Low pass filter"):
                
                if(self.img1Obj.noised_img.size == 0):

                    QMessageBox.about(self,"Warning","please apply noise to the image first" )
                
                else:
                    output_image = self.img1Obj.median_filter()

                    self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Average  Low pass filter"):

                if(self.img1Obj.noised_img.size == 0):
                    
                    QMessageBox.about(self,"Warning","please apply noise to the image first" )
                
                else:
                    output_image = self.img1Obj.average_filter()

                    self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Gaussian Low pass filter"):

                if(self.img1Obj.noised_img.size == 0):
                    
                    QMessageBox.about(self,"Warning","please apply noise to the image first" )

                else:
                    output_image = self.img1Obj.gaussian_filter()

                    self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Cumulative curve"):
                output_image = self.img1Obj.cumulative_crve()

                self.plot_histo(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Sobel Mask"):
                output_image = self.img1Obj.sobel()

                self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Prewitt Mask"):
                output_image = self.img1Obj.prewitt()

                self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Roberts Mask"):
                output_image = self.img1Obj.roberts()

                self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)

            elif(self.ui.comboBox.currentText() == "Hybrid image"):
                if(self.img_2 == ''):

                    QMessageBox.about(self,"Warning","please insert the second image, it should be the same size as the first" )
                else:

                    if (self.img2Obj.height == self.img1Obj.height) & (self.img2Obj.width == self.img1Obj.width) & (self.img2Obj.imgType == self.img1Obj.imgType) :
                        img1 = self.img1Obj.freq_lowpass_Filter()
                        img2 = self.img2Obj.sobel()
                        output_image =  0.4 * img1 + 0.6 * img2
                        output_image = output_image.astype(np.uint8)
                        self.plot(output_image , self.img1Obj.imgType, self.ui.graphicsView_3)
                        

                    else :
                        QMessageBox.about(self,"Warning","please load images of the same size, and type (grayscale , RGB)" )
                        

                        
           
        
  


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()