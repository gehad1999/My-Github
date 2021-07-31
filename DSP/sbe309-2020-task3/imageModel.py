## This is the abstract class that the students should implement  

from modesEnum import Modes
import numpy as np
#from main import mixer 
from modesEnum import Modes
import cv2


class ImageModel():

    """
    A class that represents the ImageModel"
    """

    def __init__(self):
        pass

    def __init__(self, imgPath: str):
        self.imgPath = imgPath
        ###
        # ALL the following properties should be assigned correctly after reading imgPath 
        self.imgByte = cv2.imread(self.imgPath)
        self.imgByte= cv2.cvtColor(self.imgByte, cv2.COLOR_BGR2GRAY)  # for the first error has appeared for me from testtask , it hhas been solved by adding this line#
        #print(self.imgByte.shape)

        self.dft = np.fft.fft2(self.imgByte)
        #self.dft = np.fft.fftshift(self.dft)
        self.real=np.real(self.dft)
        self.imaginary=np.imag(self.dft)
        self.magnitude =np.abs(self.dft)            ##########for the 2nd assertion error,,i make sure that self.magnitude = the magnitude of Fourier not 20*log(magnitude) or something else
        self.phase = np.angle(self.dft)
        self.uniphase=np.zeros(self.phase.shape,dtype=None ,order='C')
        self.unimagnitude=np.ones(self.magnitude.shape,dtype=None ,order='C')
       
    def fourierTransform(self):
              
              self.imgByte = cv2.imread(self.imgPath)
              self.imgByte= cv2.cvtColor(self.imgByte, cv2.COLOR_BGR2GRAY)    # for the first error has appeared for me from testtask , it hhas been solved by adding this line#
              self.dft = np.fft.fft2(self.imgByte)
              self.dft = np.fft.fftshift(self.dft)
    def realfft(self):
              
              self.fourierTransform()
              self.real=np.real(self.dft)
              return (self.real)
              
    def imagfft(self):
             
              self.fourierTransform()
              self.imaginary=np.imag(self.dft)
              return (self.imaginary)
             
    def magfft(self):
             
              self.fourierTransform()
              self.magnitude =np.abs(self.dft)
              return(self.magnitude)
             
    def phasefft(self):
              self.fourierTransform()
              self.phase = np.angle(self.dft)
              return(self.phase)
    def uni_Phasefft(self):
              self.fourierTransform()
              self.uniphase = np.angle(self.dft)
              self.uniphase=np.zeros(self.uniphase.shape,dtype=None ,order='C')
              return (self.uniphase)
    def uni_Magnitudefft(self):
              self.fourierTransform()
              self.unimagnitude = np.abs(self.dft)
              self.unimagnitude=np.ones(self.unimagnitude.shape,dtype=None ,order='C')
              return (self.unimagnitude)
              
   
    def mix(self, imageToBeMixed: 'ImageModel', magnitudeOrRealRatio: float, phaesOrImaginaryRatio: float, mode: 'Modes' ) -> np.ndarray:
        """
        a function that takes ImageModel object mag ratio, phase ration 
        """
        ### 
        # implement this function
        if  mode==Modes.magnitudeAndPhase:
            self.magnitudeImg2=imageToBeMixed.magfft()
            self.phaseImg2=imageToBeMixed.phasefft()
            self.mixMag=np.add(self.magfft() * magnitudeOrRealRatio,self.magnitudeImg2*(1- magnitudeOrRealRatio))
            self.mixPhase=np.add((1-phaesOrImaginaryRatio)* self.phasefft() ,(phaesOrImaginaryRatio)* self.phaseImg2)
            self.mix=np.multiply(self.mixMag,np.exp(self.mixPhase))
            self.mix=np.fft.ifft2(self.mix)
            self.mix=np.abs(self.mix)
         
        elif  mode==Modes.Uni_magnitudeAndPhase:
            self.Uni_MagImg2=imageToBeMixed.uni_Magnitudefft()
            self.Uni_phaseImg2=imageToBeMixed.uni_Phasefft()
            self.mixReal=np.add(self.uni_Magnitudefft() * magnitudeOrRealRatio,self.Uni_MagImg2*(1- magnitudeOrRealRatio))
            self.mixImag=np.add((1-phaesOrImaginaryRatio)* self.uni_Phasefft() ,(phaesOrImaginaryRatio)* self.Uni_phaseImg2)
            self.mix=np.multiply(self.mixReal,np.exp(self.mixImag)) 
            self.mix=np.fft.ifft2(self.mix)
            self.mix=np.abs(self.mix)
        elif  mode==Modes.magnitudeAndUNI_Phase:
            self.Uni_MagImg2=imageToBeMixed.magfft()
            self.Uni_phaseImg2=imageToBeMixed.uni_Phasefft()
            self.mixReal=np.add(self.magfft() * magnitudeOrRealRatio,self.Uni_MagImg2*(1- magnitudeOrRealRatio))
            self.mixImag=np.add((1-phaesOrImaginaryRatio)* self.uni_Phasefft() ,(phaesOrImaginaryRatio)* self.Uni_phaseImg2)
            self.mix=np.multiply(self.mixReal,np.exp(self.mixImag)) 
            self.mix=np.fft.ifft2(self.mix)
            self.mix=np.abs(self.mix)
        elif  mode==Modes.UNI_magnitudeAndPhase:
            self.Uni_MagImg2=imageToBeMixed.uni_Magnitudefft()
            self.Uni_phaseImg2=imageToBeMixed.phasefft()
            self.mixReal=np.add(self.uni_Magnitudefft() * magnitudeOrRealRatio,self.Uni_MagImg2*(1- magnitudeOrRealRatio))
            self.mixImag=np.add((1-phaesOrImaginaryRatio)* self.phasefft() ,(phaesOrImaginaryRatio)* self.Uni_phaseImg2)
            self.mix=np.multiply(self.mixReal,np.exp(self.mixImag)) 
            self.mix=np.fft.ifft2(self.mix)
            self.mix=np.abs(self.mix)
        elif mode==Modes.realAndImaginary:

            self.realImg2=imageToBeMixed.realfft()
            self.imagImg2=imageToBeMixed.imagfft()
            self.mixReal=np.add(self.realfft() * magnitudeOrRealRatio,self.realImg2*(1- magnitudeOrRealRatio))
            self.mixImag=np.add((1-phaesOrImaginaryRatio)* self.imagfft() ,(phaesOrImaginaryRatio)* self.imagImg2)
            self.mix=np.multiply(self.mixReal,np.exp(self.mixImag)) 
            self.mix=np.fft.ifft2(self.mix)
            self.mix=np.abs(self.mix)
        elif mode==Modes.realAndphase:
            self.realImg2=imageToBeMixed.realfft()
            self.imagImg2=imageToBeMixed.phasefft()
            self.mixReal=np.add(self.realfft() * magnitudeOrRealRatio,self.realImg2*(1- magnitudeOrRealRatio))
            self.mixImag=np.add((1-phaesOrImaginaryRatio)* self.phasefft() ,(phaesOrImaginaryRatio)* self.imagImg2)
            self.mix=np.multiply(self.mixReal,np.exp(self.mixImag)) 
            self.mix=np.fft.ifft2(self.mix)
            self.mix=np.abs(self.mix)
        elif mode==Modes.magnitudeAndImaginary:
            self.realImg2=imageToBeMixed.magfft()
            self.imagImg2=imageToBeMixed.imagfft()
            self.mixReal=np.add(self.magfft() * magnitudeOrRealRatio,self.realImg2*(1- magnitudeOrRealRatio))
            self.mixImag=np.add((1-phaesOrImaginaryRatio)* self.imagfft() ,(phaesOrImaginaryRatio)* self.imagImg2)
            self.mix=np.multiply(self.mixReal,np.exp(self.mixImag)) 
            self.mix=np.fft.ifft2(self.mix)
            self.mix=np.abs(self.mix)
        elif mode==Modes.realAndUni_phase:
            self.realImg2=imageToBeMixed.realfft()
            self.imagImg2=imageToBeMixed.phasefft()
            self.mixReal=np.add(self.realfft() * magnitudeOrRealRatio,self.realImg2*(1- magnitudeOrRealRatio))
            self.mixImag=np.add((1-phaesOrImaginaryRatio)* self.phasefft() ,(phaesOrImaginaryRatio)* self.imagImg2)
            self.mix=np.multiply(self.mixReal,np.exp(self.mixImag)) 
            self.mix=np.fft.ifft2(self.mix)
            self.mix=np.abs(self.mix)
        elif mode==Modes.Uni_magAndImaginary:
            self.realImg2=imageToBeMixed.uni_Magnitudefft()
            self.imagImg2=imageToBeMixed.imagfft()
            self.mixReal=np.add(self.uni_Magnitudefft() * magnitudeOrRealRatio,self.realImg2*(1- magnitudeOrRealRatio))
            self.mixImag=np.add((1-phaesOrImaginaryRatio)* self.imagfft() ,(phaesOrImaginaryRatio)* self.imagImg2)
            self.mix=np.multiply(self.mixReal,np.exp(self.mixImag)) 
            self.mix=np.fft.ifft2(self.mix)
            self.mix=np.abs(self.mix)
        return self.mix
        ###
        pass