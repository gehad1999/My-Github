# ------------------------------------------------------
# -------------------- mplwidget.py --------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import*

# from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.qt_compat import QtCore, QtWidgets

from matplotlib.figure import Figure

    
class MplWidget(QWidget):
    
    def __init__(self, parent = None):

        QWidget.__init__(self, parent)
        
        self.canvas = FigureCanvas(Figure(figsize=(15, 15)))
        
        # self.canvas = FigureCanvas(Figure( tight_layout= True))
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
       




        self.axes = self.canvas.figure.add_subplot(111)
        
        # self.canvas.axes = self.canvas.figure.add_subplot(111)
        # self.canvas.axes.set_autoscale_on(True)
        # self.canvas.axes.autoscale_view(True,True,True)
        # # self.canvas.figure.set_figheight(15)
        # # self.canvas.figure.set_figwidth(15)
        # self.canvas.figure.tight_layout()
        self.setLayout(vertical_layout)







# import matplotlib
# # Make sure that we are using QT5
# matplotlib.use('Qt5Agg')
# from PyQt5 import QtCore, QtWidgets


# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure

# class MplWidget(FigureCanvas):
#     """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = fig.add_subplot(111)

#         self.compute_initial_figure()

#         FigureCanvas.__init__(self, fig)
#         self.setParent(parent)

#         FigureCanvas.setSizePolicy(self,
#                                    QtWidgets.QSizePolicy.Expanding,
#                                    QtWidgets.QSizePolicy.Expanding)
#         FigureCanvas.updateGeometry(self)

#     def compute_initial_figure(self):
#         pass
        
