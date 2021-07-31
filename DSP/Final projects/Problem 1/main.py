import plotly.express as px
from plotly.offline import  plot
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5 import QtWidgets ,QtGui, QtCore
from qt import Ui_MainWindow  
import sys
import matplotlib.animation as animation

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)    
        self.df=pd.read_csv('covid-19.csv')
        self.ui.bubble.clicked.connect(self.bubble)
       # self.ui.map.clicked.connect(self.map_graph)
        self.ui.Box.currentIndexChanged.connect(self.bar)
#df['date']=pd.to_datetime( df['date'] ,format='%d/%m/%Y')
#df=df.sort_values('date', ascending=True)
#df['cases']=df['cases']*1000
#df=df.dropna()
#print(df.sample(10))
#print(df.cases)
    def bubble(self):
        #self.df.total_recovery=self.df.total_recovery * 1000
        self.fig = px.scatter(self.df, x="total_cases", y="total_deaths", animation_frame="date",
                              size="total_recovery", color="continent")
        self.fig.write_html("bubble.html")
        #plot(self.fig)
        
    
    #def map_graph(self):
#       # self.df=self.df.sort_values('cases', ascending=False)
##       self.df['date_frame']=pd.to_datetime( self.df['date_frame'] ,format='%d/%m/%y20')
##       self.df=self.df.sort_values(['date_frame','total_cases'], ascending=True)
#       self.choose
#       self.fig = px.bar(self.df, x="country", y="total_cases", color="continent",
#                animation_frame="date")
#       self.fig.write_html("bar.html")
#       # plot(self.fig)
    def bar(self):
        self.index=self.ui.Box.currentIndex()
        if(self.index==1):
            self.df['date_frame']=pd.to_datetime( self.df['date_frame'] ,format='%d/%m/%y20')
            self.df=self.df.sort_values(['date_frame','total_cases'], ascending=True)
            self.fig = px.bar(self.df, x="country", y="total_cases", color="continent",
                animation_frame="date")
            self.fig.write_html("bar_cases.html")
        if(self.index==2):
           self.df['date_frame']=pd.to_datetime( self.df['date_frame'] ,format='%d/%m/%y20')
           self.df=self.df.sort_values(['date_frame','total_deaths'], ascending=True)
           self.fig = px.bar(self.df, x="country", y="total_deaths", color="continent",
                animation_frame="date")
           self.fig.write_html("bar_deaths.html")
       
def main():
    app= QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec()

if __name__ == "__main__":
        main()        