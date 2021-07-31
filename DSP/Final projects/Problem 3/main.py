from PyQt5 import QtWidgets , QtGui
from output import Ui_MainWindow
import sys
from  PyQt5.QtWidgets  import QFileDialog , QPushButton
import numpy as np
import sounddevice as sd


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.connect)
        self.ui.pushButton_2.clicked.connect(self.connect)
        self.ui.pushButton_3.clicked.connect(self.connect)
        self.ui.pushButton_4.clicked.connect(self.connect)
        self.ui.pushButton_5.clicked.connect(self.connect)
        self.ui.pushButton_6.clicked.connect(self.connect)
        self.ui.pushButton_7.clicked.connect(self.connect)
        self.ui.pushButton_8.clicked.connect(self.connect)
        self.ui.pushButton_9.clicked.connect(self.connect)
        self.ui.pushButton_10.clicked.connect(self.connect)
        
        global time , fs
        fs = 8000
        time = np.linspace(0,1,num=fs)
        
    def play (self, frequency):
        
        Y = np.sin(2 * np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)

        Y += np.sin(2 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/2
        Y += np.sin(3 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/4
        Y += np.sin(4 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/8
        Y += np.sin(5 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/16
        Y += np.sin(6 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/32
        Y+= Y*Y*Y
        Y *= 1 + 16 * time * np.exp(-6 * time)
        sd.play(Y , fs)
        
    def connect (self):
        
        if self.ui.pushButton.clicked:
            
            self.play (740)
        elif self.ui.pushButton_2.clicked: 
            self.play (622.3)
        elif self.ui.pushButton_3.clicked: 
            self.play (554.4)
        elif self.ui.pushButton_4.clicked: 
            self.play (466.2)
        elif self.ui.pushButton_5.clicked: 
            self.play (415)
        elif self.ui.pushButton_6.clicked:
            
            self.play (370)    
        elif self.ui.pushButton_7.clicked: 
            self.play (311.1)
        elif self.ui.pushButton_8.clicked: 
            self.play (277.2)    
        elif self.ui.pushButton_9.clicked: 
            self.play (116.2)
        elif self.ui.pushButton_10.clicked: 
            self.play (69.3)
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()