# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from TestPAML.data import f2_data
 
def plot_f2(model, history):
    xs, ys, zs, grad, xs_c, ys_c, zs_c, grad_c = f2_data()
    
    plt.figure(1, dpi=600)
    plt.semilogy(history.history['loss'])
    plt.title('training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.show()
    
    zs_model = model.predict(np.hstack((xs, ys)))[0]
    X = xs.reshape((20,20))
    Y = ys.reshape((20,20))
    Z_MODEL = zs_model.reshape((20,20))
    Z = zs.reshape((20,20))
    plt.figure(2, dpi=600)
    ax2 = plt.axes(projection="3d")
    ax2.plot_wireframe(X, Y, Z_MODEL,linewidth=0.5, 
                     edgecolors="red", label="model")
    ax2.plot_wireframe(X, Y, Z, linewidth=0.5, 
                     edgecolors="blue", label="data")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    ax2.set_zlabel('$f_1$')

        
    zs_model = model.predict(np.hstack((xs, ys)))[1][:,0]
    X = xs.reshape((20,20))
    Y = ys.reshape((20,20))
    Z_MODEL = zs_model.reshape((20,20))
    Z = grad[:,0].reshape((20,20))
    plt.figure(4, dpi=600)
    ax2 = plt.axes(projection="3d")
    ax2.plot_wireframe(X, Y, Z_MODEL, linewidth=0.5, 
                     edgecolors="red", label="model")
    ax2.plot_wireframe(X, Y, Z, linewidth=0.5, 
                     edgecolors="blue", label="data")
    plt.xlabel('x')
    plt.ylabel('y')
    ax2.set_zlabel('$\partial_{x}f_1$')


    zs_model = model.predict(np.hstack((xs, ys)))[1][:,1]
    X = xs.reshape((20,20))
    Y = ys.reshape((20,20))
    Z_MODEL = zs_model.reshape((20,20))
    Z = grad[:,1].reshape((20,20))
    plt.figure(6, dpi=600)
    ax2 = plt.axes(projection="3d")
    ax2.plot_wireframe(X, Y, Z_MODEL, linewidth=0.5, 
                     edgecolors="red", label="model")
    ax2.plot_wireframe(X, Y, Z, linewidth=0.5, 
                     edgecolors="blue", label="data")
    plt.xlabel('x')
    plt.ylabel('y')
    ax2.set_zlabel('$\partial_{y}f_1$')


    plt.show()
