
from PyQt5 import QtWidgets
from Qt import Ui_MainWindow
import sys
import matplotlib.pyplot as plt
import librosa
import pyqtgraph 
import librosa.display
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.Browse.clicked.connect(self.Browse)
        self.ui.Browse_2.clicked.connect(self.Seperate)
        self.ui.Browse_3.clicked.connect(self.coctail)
    def Browse(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath() , '*.wav')    
    def Seperate(self):
        sr = 16000
        n_fft = 4096
        len_hop = n_fft / 4
        data, sr = librosa.load(self.fileName, sr=sr, mono=False)
        music_wav, vocal_wav = data
        mix_wav = librosa.to_mono(data)
        # Write each wav
        librosa.output.write_wav('music.wav',np.asfortranarray(music_wav) , sr)
        librosa.output.write_wav('vocal.wav', np.asfortranarray(vocal_wav), sr)
        librosa.output.write_wav('mix.wav', np.asfortranarray(mix_wav), sr)
        #Plot waveforms
        plt.figure(1)
        plt.subplot(3, 1, 1)
        librosa.display.waveplot(np.asfortranarray(mix_wav), sr=sr, color='r')
        plt.title('mix')
        
        #
        plt.subplot(3, 1, 2)
        librosa.display.waveplot(np.asfortranarray(music_wav), sr=sr)
        plt.title('music')
        #
        plt.subplot(3, 1, 3)
        librosa.display.waveplot(np.asfortranarray(vocal_wav), sr=sr)
        plt.title('vocal')
        plt.tight_layout()
        plt.savefig('plot.png')
        image=QPixmap('plot.png')
        self.ui.label.setScaledContents(True)
        self.ui.label.setPixmap(image)
        
    def  coctail(self):
        np.random.seed(0)
        n_samples = 2000
        time = np.linspace(0, 8, n_samples)

        s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
        s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

        S = np.c_[s1, s2, s3]
        S += 0.2 * np.random.normal(size=S.shape)  # Add noise

        S /= S.std(axis=0)  # Standardize data
        # Mix data
        A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
        X = np.dot(S, A.T)  # Generate observations

        # Compute ICA
        ica = FastICA(n_components=3)
        S_ = ica.fit_transform(X)  # Reconstruct signals
        A_ = ica.mixing_  # Get estimated mixing matrix

        # We can `prove` that the ICA model applies by reverting the unmixing.
        assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

        # For comparison, compute PCA
        pca = PCA(n_components=3)
        H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

        # #############################################################################
        # Plot results

        plt.figure()

        models = [X, S, S_, H]
        names = ['Observations (mixed signal)',
                'True Sources',
                'ICA recovered signals',
                'PCA recovered signals']
        colors = ['red', 'steelblue', 'orange']

        for ii, (model, name) in enumerate(zip(models, names), 1):
            plt.subplot(4, 1, ii)
            plt.title(name)
            for sig, color in zip(model.T, colors):
                plt.plot(sig, color=color)
                plt.savefig('plot.png')
                image=QPixmap('plot.png')
                self.ui.label.setScaledContents(True)
                self.ui.label.setPixmap(image)


        plt.tight_layout()
        
 



    


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()