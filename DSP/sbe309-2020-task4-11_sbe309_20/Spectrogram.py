import matplotlib.pyplot as plot

from scipy.io import wavfile
import  os, glob


    
def spectrogram ():
    for filename in glob.glob(os.path.join('./Wave', '*.wav')):
    
        samplingFrequency, signalData = wavfile.read(filename) 
        plot.specgram(signalData[:,0],Fs=samplingFrequency)
        name = os.path.splitext(os.path.basename(filename))[0]
        plot.savefig ('./Spectrogram/'+ os.path.join(name) )
        


if __name__ == "__main__":
    spectrogram()