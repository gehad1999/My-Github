import matplotlib.pyplot as plot
from scipy.io import wavfile
import  os, glob
import librosa
import scipy.spatial
def change_to_be_hex(r):
    return int(str(r),base=16)

def xor_two_str(str1,str2):
    a = change_to_be_hex(str1)
    b = change_to_be_hex(str2)
    return bin(a ^ b).count('1')
def distanceHamming(str1,str2):
    return scipy.spatial.distance.hamming(str1,str2)
def main ():
    for filename in glob.glob(os.path.join('./Wave', '*.wav')):
        y, sr = librosa.load(filename)
        S, phase = librosa.magphase(librosa.stft(y))  
        spectral_flatness=librosa.feature.spectral_rolloff(S=S)
        
        for i in glob.glob(os.path.join('./feature', '*.wav')):
            u, sr = librosa.load(i)
            f, phase = librosa.magphase(librosa.stft(u))  
            second=librosa.feature.spectral_rolloff(S=f)

            print(spectral_flatness == second)




if __name__ == "__main__":
    main()