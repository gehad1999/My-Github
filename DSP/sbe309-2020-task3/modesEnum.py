## enum for different operation modes

import enum
from enum import Enum

class Modes(enum.Enum):
    magnitudeAndPhase  ='testMagAndPhaseMode'
    realAndImaginary ='testRealAndImagMode'
    Uni_magnitudeAndPhase ='testuni_phaesOrImaginaryRatio'
    magnitudeAndUNI_Phase='magnitudeAndUNI_Phase'
    UNI_magnitudeAndPhase='UNI_magnitudeAndPhase'
    realAndphase='realAndphase'
    realAndUni_phase='realAndUni_phase'
    Uni_magAndImaginary='Uni_magAndImaginary'
# How to use enum?
# Just type enumName.enumElement
# for example enum.magnitudeAndPhase or enum.realAndImaginary

# Why should we use enum instead of strings?
# Strings can be typed in different formats "lower, UPPER, camelCase, under_score" or spellings
# enum gives you the flexibilty to change the string to anything and no to ruin your program
# Try to change the value of any elemnt in enum and you'll find that the test still works, "I hope =D"
# You can assign integers, strings .... to your enum elements