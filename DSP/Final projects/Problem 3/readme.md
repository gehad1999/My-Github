# Problem 3
## Piano
## Code
First we used thouse Equation to generate the sound 
```python

Y = np.sin(2 * np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)

Y += np.sin(2 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/2
Y += np.sin(3 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/4
Y += np.sin(4 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/8
Y += np.sin(5 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/16
Y += np.sin(6 *2* np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time)/32
Y+= Y*Y*Y
Y *= 1 + 16 * time * np.exp(-6 * time)
``` 
Then we linked every button in our GUI with the right frequency of that button and put them in a function called connect
```python
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
```

## Our GUI
![GUI Image](ui.png)
## Problems 
1- it was hard to play the sound we tried a lot of equations until finally we found the equation above

2- We first worked with math library It did not work but when we used numpy library it worked just fine 

3- choosing the right frequency of the button also took lots of efforts

## Guitar 
First we found code for virtual guitar and it gave sound like guitar but with no equation we tried to add the equations but It did not work 

## Code
```python

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


```
