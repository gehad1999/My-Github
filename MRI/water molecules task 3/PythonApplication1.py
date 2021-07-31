import PIL
from PIL import Image
import numpy as np
from numpy import asarray
from scipy.ndimage import zoom
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pylab
import random 
import matplotlib.animation
from matplotlib import animation, rc
from matplotlib import animation
from IPython.display import HTML
#It will print RGB values of the image. If the image is large redirect the output to a file using '>' 
#later open the file to see RGB values

FILENAME='wvsb.png' #image can be in gif jpeg or png format 
#'LA' for grey scale of image 'RGB' for rgb 
im=Image.open(FILENAME).convert('LA')
print (im)
data = asarray(im)
data = zoom(data, (0.01, 0.0067, 0.5))
print('shapenew',data.shape)
print(type(data))
print((data.ndim))
data2 = data.reshape((data.shape[0]), data.shape[1])

#data2 = data2.transpose()
# summarize shape
print('shape',data2.shape)
print('dim',data2.ndim)
print(data2[3][0],'data of row 4',data2[3].size)
#data3=data2.resize((4, 4))

# create Pillow image



rows, cols = (4, 4) 
arr=[]

for i in range (rows): 
   
   for j in range(cols):
      row = [] 
      T1=[]
      T2=[]
      row.append((data[i][j])) 
      arr.append(np.concatenate(row))
      print(i ,j, data[i][j] )
    
arr=np.concatenate(arr)
T1=list(arr)
T2=list(arr)

for index, item in enumerate(T1):
    if item==255 : # 255 is white color we assumed white color represents the water material 
        T1[index] =40000
        T2[index]= 20000
    
    if item==0 :   # 0 is black color we assumed black color represents the fat material 
        T1[index] = 60000
        T2[index]= 20000

    if item==78 :  # represents third material by third color 
        T1[index] =40000
        T2[index]= 10000


print('T1',T1)
print('T2',T2)
#print(data2)
print('1',asarray(arr).shape)
print('1',asarray(arr).size)
print('1',asarray(arr).dtype)
print('1',asarray(arr).ndim)
print('1',len(asarray(arr)))



dT = 100	
T = 100000
df = []
T1 = T1[0]
T2 = T2[0]
x=[0,1,2]
for d in range(len(x)):
    df.append(3001)
    
N = math.ceil(T/dT)+1
result=[None]*2


def freepression(T,T1,T2,df,i):
 #resultA,B=[],[]
 for d in range(3):
  phi = 2*math.pi*df[i]*T/1000
  Rz = [[math.cos(phi), -math.sin(phi), 0],
      [math.sin(phi), math.cos(phi) ,0],
      [0, 0, 1]]
  E1 = math.exp(-T/T1)	
  E2 = math.exp(-T/T2)
  B = [0, 0, 1-E1]
  A = [[E2, 0, 0],
       [0 ,E2, 0],
       [0, 0 ,E1]]
  resultA = np.dot(A,Rz)
 # d+=1
  return (resultA,B	)

def decay(i):
  A,B=[],[]
  A,B= freepression(dT,T1,T2,df,i)

  M = np.zeros((N,3))
 
  M[0,:]= np.array([1,0,0])
 
  for k in range (1,N):
    
    M[k,:] = np.dot(A,M[k-1,:]) + B
 
  return (M)

M = decay(0)
print('At T1[0]&T2[0] relates to the first point at row=0,col=0,from phantom 4*4 which represents color refers to one material')
print('T1[0]&T2[0]--> the first values in our lists to represent data[row=0][col=0] of our phantom of shape(4*4)')
print('At B0 in mT =',df,'\nFor this strength of MF T1[0]=',T1,' T2[0]=',T2)
#print (i)
xdata = []
ydata = []
timedata = np.arange(N)
axes = pylab.gca()
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
line,=axes.plot(xdata,ydata,'b-')
xdata2 = []
ydata2 = []
timedata2 = np.arange(N)
axes2 = pylab.gca()
axes2.set_xlim(-1,1)
axes2.set_ylim(-1,1)
line2,=axes2.plot(xdata2,ydata2,'r-')
xdata3 = []
ydata3 = []
timedata3 = np.arange(N)
axes3 = pylab.gca()
axes3.set_xlim(-1,1)
axes3.set_ylim(-1,1)
line3,=axes3.plot(xdata3,ydata3,'y-')
    
pylab.subplot(111)
for d in range (N):
    xdata.append(M[d,0])
    
    ydata.append(M[d,1])
    
    line.set_xdata(xdata)
    
    line.set_ydata(ydata)
    
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.01)
fig=plt.show()


