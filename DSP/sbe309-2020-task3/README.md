<!--Headline-->
<!--Image-->
<!--UL-->
<!-- URLs-->

## Write your name and ID
### Name:gehad mohamed ahmed ali mohamed
### ID: 27

# MagnitudePhaseMixerTemplate
Starter Template for Magnitude Phase Mixer Task


### install opencv to run the test file

### Implement the ImageModel in imageModel.py and its mix function
### run testTask.py --> python testTask.py
### assign a valid path for image1Path and image2Path
### Now when you run testTask you should get the following line
### AssertionError: This is not a numpy array, check the return value of your implemented mix function

### when you implement the mix function correctly you should get the following 2 lines
### Modes.magnitudeAndPhase passed successfully
### Modes.realAndImaginary passed successfully

### Do not forget to update the dep.txt file


* These are  some of samples of my mixer programming software :
* You can display 1st image on its own label also you can select which component that you want to display in fourier transform using any checkbox button under the label of an image , the smae can be done for 2nd image by the   same way with its own objects to be used , this photo is a sample from results , in a (greyscale) of ImgByte (not colored scale) .
*![](/firstImg.PNG)
*  Actually for mixing you need the 1st image, you can choose which image firstly you want to take the 1st comp from it, after choosing your first image path then the first comp ,there'll be a new line that is apper to inform you which applicable 2nd comp you can choose from any image as a 2nd comp for mixing .
*![](/Messege_for_the_2nd_comp_type.PNG)
* Then the result of Mixing between  the two selected images with two selected components can be displayed through checking the checkbox with name refers to the components,when (the scale was colored scale) of ImgByte (not a grey scale) .
*![](/The_new_pic_from_mixing.PNG)
*![](/ex2output.PNG)

* Finally, you can see the logging which is uploaded into the file "temp.log".
*![](/the_logging_file.PNG)
### 1st An assertion error .
*![](/1stasser.PNG)
### After solving the 1st assertion error, the output of mix array is white or black as it observes in two second figures.
*![](/MagAndPhase.PNG)
*![](/RealAndIMag.PNG)
### there is a problem from inversefft of mix array ,, when i use the inverse for the real comp of only one image i get this result but when i use for mixing the only result that i don't get the expected pic from the mixed arrays, i test the function on only one array without doing any mixing and this was the result, the problem when i mix the output image is  like the figure above.
*![](/inverseimg1.PNG)
### 2nd assertion error that is displayed for me .
*![](/asserEr.PNG)
### But i didn't use the log to get the magnitude i use these lines for magnitude and phase from ImageModel.
*![](/arrays.PNG)


