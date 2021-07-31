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


dT = 1	
T = 1000
df = []
x=[0,1,2]
for d in range(len(x)):
    df.append(random.randrange(1000,1500))
T1 = 600
T2 = 100
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

#i=0
#for i in range (len(df)):
M = decay(0)
G=  decay(1)
H=  decay(2)
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
    xdata2.append(G[d,0])
    xdata3.append(H[d,0])
    ydata.append(M[d,1])
    ydata2.append(G[d,1])
    ydata3.append(H[d,1])
    line.set_xdata(xdata)
    line2.set_xdata(xdata2)
    line3.set_xdata(xdata3)
    line.set_ydata(ydata)
    line2.set_ydata(ydata2)
    line3.set_ydata(ydata3)
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.01)
fig=plt.show()




#ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init_animation, frames=50)
#ani.save('Downloads/animation.gif', writer='imagemagick', fps=30)

  #xdata.append(M[i,0])
  
  #ydata.append(M[i,1])
  #line.set_xdata(xdata)
  #line.set_ydata(ydata)
#    plt.plot(timedata,xdata,'r')
#    plt.plot(timedata,ydata,'b')
   
  #plt.draw()
  #plt.pause(1e-17)
  #time.sleep(0.01)

#plt.show()




