import numpy as np
from IPython.display import Audio
import sounddevice as sd
class GuitarString:
    def __init__(self, pitch, starting_sample, sampling_freq, stretch_factor):
        """Inits the guitar string."""
        self.pitch = pitch
        self.starting_sample = starting_sample
        self.sampling_freq = sampling_freq
        self.stretch_factor = stretch_factor
        self.init_wavetable()
        self.current_sample = 0
        self.previous_value = 0
        
    def init_wavetable(self):
        """Generates a new wavetable for the string."""
        wavetable_size = self.sampling_freq // int(self.pitch)
        self.wavetable = (2 * np.random.randint(0, 2, wavetable_size) - 1).astype(np.float)
        
    def get_sample(self):
        """Returns next sample from string."""
        if self.current_sample >= self.starting_sample:
            current_sample_mod = self.current_sample % self.wavetable.size
            r = np.random.binomial(1, 1 - 1/self.stretch_factor)
            if r == 0:
                self.wavetable[current_sample_mod] =  0.5 * (self.wavetable[current_sample_mod] + self.previous_value)
            sample = self.wavetable[current_sample_mod]
            self.previous_value = sample
            self.current_sample += 1
        else:
            self.current_sample += 1
            sample = 0
        return sample
fs = 8000
freqs = [90]
unit_delay = fs//3
delays = [unit_delay * _ for _ in range(len(freqs))]
stretch_factors = [2 * f/98 for f in freqs]
strings = []
for freq, delay, stretch_factor in zip(freqs, delays, stretch_factors):
     string = GuitarString(freq, delay, fs, stretch_factor)
     strings.append(string)
     
guitar_sound = [sum(string.get_sample() for string in strings) for _ in range(fs * 6)]

sd.play(guitar_sound , fs)
sd.play(guitar_sound , fs)
#display(Audio(guitar_sound , rate = fs))