import numpy as np
import cv2

class Point(object):
 def __init__(self,x,y):
  self.x = x
  self.y = y

 def getX(self):
  return self.x
 def getY(self):
  return self.y

class coordinate(object):
    def __init__(self, x, y):
            self.x = x
            self.y = y
    def getX(self):
        # Getter method for a Coordinate object's x coordinate.
        # Getter methods are better practice than just
        # accessing an attribute directly
        return self.x
    def getY(self):
        # Getter method for a Coordinate object's y coordinate
        return self.y

def Diff(img,currentPoint,tmpPoint):
 return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))


def regionGrow(img,seeds,thresh):
 height, weight = img.shape
 seedMark = np.zeros(img.shape)
 seedList = []
 for seed in seeds:
  seedList.append(seed)
 label = 1
 connects =  connects = [coordinate(-1, -1), coordinate(0, -1), coordinate(1, -1), coordinate(1, 0), coordinate(1, 1), 
    coordinate(0, 1), coordinate(-1, 1), coordinate(-1, 0)]
 while(len(seedList)>0):
  currentPoint = seedList.pop(0)

  seedMark[currentPoint.x,currentPoint.y] = label
  for i in range(8):
   tmpX = currentPoint.x + connects[i].x
   tmpY = currentPoint.y + connects[i].y
   if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
    continue
   diff = Diff(img,currentPoint,Point(tmpX,tmpY))
   if diff < thresh and seedMark[tmpX,tmpY] == 0:
    seedMark[tmpX,tmpY] = label
    seedList.append(Point(tmpX,tmpY))
 return seedMark