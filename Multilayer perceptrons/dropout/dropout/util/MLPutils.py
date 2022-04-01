import matplotlib.pyplot as plt
from IPython import display

class Animator3D:
    def __init__(self, xlabel=None, ylabel=None, zlabel = None, 
                 legend=None, 
                 xlim=None,ylim=None, zlim = None, 
                 xscale='linear', yscale='linear', zscale = 'linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), 
                 figsize=(10, 10)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        #display.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.figure(figsize=figsize), plt.axes(projection = '3d')
        #if nrows * ncols * nwidth == 1:
         #   self.axes = [self.axes,]
        # Use a lambda function to capture arguments
        display.set_matplotlib_formats('svg')
        plt.rcParams['figure.figsize'] = figsize
        self.config_axes = lambda: set_axes(self.axes, xlabel, ylabel, zlabel,
                xlim, ylim, zlim, 
                xscale, yscale, zscale, legend)
        self.X, self.Y, self.Z, self.fmts = None, None, None, fmts
        
    def add(self, x, y, z):
        # Add multiple data points into the figure
        if not hasattr(z, "__len__"):
            z = [z]
        n = len(z)
        if not hasattr(y, "__len__"):
            y = [y] * n
        if not hasattr(x, "__len__"):
            x = [x] * n 
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        if not self.Z:
            self.Z = [[] for _ in range(n)]
        for i, (a, b, c) in enumerate(zip(x, y, z)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
                self.Z[i].append(c)
        self.axes.cla()
        labels = ['text{}'.format(i) for i in range(len(self.Z))]
        for x, y,z,fmt, label in zip(self.X, self.Y,self.Z, self.fmts, labels):
            self.axes.plot(x, y, z, fmt)
            #self.axes.text(x, y, z, label)
           
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

class IteratorEx(object):
    def __init__(self, it):
        self.it = iter(it)
        self.sentinel = object()
        self.nextItem = next(self.it, self.sentinel)
        self.hasNext = self.nextItem is not self.sentinel

    def next(self):
        ret, self.nextItem = self.nextItem, next(self.it, self.sentinel)
        self.hasNext = self.nextItem is not self.sentinel
        return ret

    def __iter__(self):
        while self.hasNext:
            yield self.next()    

def set_axes(axes, 
                 xlabel, ylabel, zlabel,
                 xlim, ylim, zlim, 
                 xscale, yscale, zscale, 
                 legend):
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_zlabel(zlabel)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        axes.set_zlim(zlim)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_zscale(zscale)
        if legend:
            axes.legend(legend)
        axes.grid() 