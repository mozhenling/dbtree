
"""
Created on Fri Mar 19 14:38:01 2021

@author: mozhenling
"""
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_pic(X, Y, Z, z_max, title, z_min=0, fontsize = 16):
    fig = plt.figure()
    plt.rc('font', size = fontsize)
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=('cool')) # cool looks good
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    plt.show()


def get_X_AND_Y(X_min, X_max, Y_min, Y_max, reso=0.01):
    resolution = reso * (X_max - X_min)
    X = np.arange(X_min, X_max, resolution)
    Y = np.arange(Y_min, Y_max, resolution)
    X, Y = np.meshgrid(X, Y)
    return (
     X, Y)


def Rastrigin(X=None, Y=None, objMin=True, is2Show=False, X_min=-5.52, X_max=5.12, Y_min=-5.12, Y_max=5.12):
    A = 10
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
        return (
         X, Y, Z, 100, 'Rastrigin function-3D')
    Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
    if objMin:
        return Z
    return -Z

def Schwefel(X=None, Y=None, objMin=True, is2Show=False, X_min=-500, X_max=500, Y_min=-500, Y_max=500):
    # https://www.sfu.ca/~ssurjano/schwef.html
    # http://benchmarkfcns.xyz/benchmarkfcns/schwefelfcn.html
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = 418.9829 * 2 - X * sin(sqrt(abs(X))) - Y * sin(sqrt(abs(Y)))
    return (
         X, Y, Z, 1800, 'Schwefel function-3D')
    
    Z = 418.9829 * 2 - sum(X * sin(sqrt(abs(X))) + Y * sin(sqrt(abs(Y))))
    if objMin:
        return Z
    return -Z

def Ackley(X=None, Y=None, objMin=True, is2Show=False, X_min=-5, X_max=5, Y_min=-5, Y_max=5):
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X ** 2 + Y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))) + np.e + 20
        return (
         X, Y, Z, 15, 'Ackley function')
    Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X ** 2 + Y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))) + np.e + 20
    if objMin:
        return Z
    return -Z


def Sphere(X=None, Y=None, objMin=True, is2Show=False, X_min=-3, X_max=3, Y_min=-3, Y_max=3):
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = X ** 2 + Y ** 2
        return (
         X, Y, Z, 20, 'Sphere function')
    Z = X ** 2 + Y ** 2
    if objMin:
        return Z
    return -Z


def Beale(X=None, Y=None, objMin=True, is2Show=False, X_min=-4.5, X_max=4.5, Y_min=-4.5, Y_max=4.5):
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = np.power(1.5 - X + X * Y, 2) + np.power(2.25 - X + X * Y ** 2, 2) + np.power(2.625 - X + X * Y ** 3, 2)
        return (
         X, Y, Z, 150000, 'Beale function')
    Z = np.power(1.5 - X + X * Y, 2) + np.power(2.25 - X + X * Y ** 2, 2) + np.power(2.625 - X + X * Y ** 3, 2)
    if objMin:
        return Z
    return -Z


def Booth(X=None, Y=None, objMin=True, is2Show=False, X_min=-10, X_max=10, Y_min=-10, Y_max=10):
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = np.power(X + 2 * Y - 7, 2) + np.power(2 * X + Y - 5, 2)
        return (
         X, Y, Z, 2500, 'Booth function')
    Z = np.power(X + 2 * Y - 7, 2) + np.power(2 * X + Y - 5, 2)
    if objMin:
        return Z
    return -Z


def Bukin(X=None, Y=None, objMin=True, is2Show=False, X_min=-15, X_max=-5, Y_min=-3, Y_max=3):
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = 100 * np.sqrt(np.abs(Y - 0.01 * X ** 2)) + 0.01 * np.abs(X + 10)
        return (
         X, Y, Z, 200, 'Bukin function')
    Z = 100 * np.sqrt(np.abs(Y - 0.01 * X ** 2)) + 0.01 * np.abs(X + 10)
    if objMin:
        return Z
    return -Z


def three_humpCamel(X=None, Y=None, objMin=True, is2Show=False, X_min=-5, X_max=5, Y_min=-5, Y_max=5):
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = 2 * X ** 2 - 1.05 * X ** 4 + 0.16666666666666666 * X ** 6 + X * Y + Y * 2
        return (
         X, Y, Z, 2000, 'three-hump camel function')
    Z = 2 * X ** 2 - 1.05 * X ** 4 + 0.16666666666666666 * X ** 6 + X * Y + Y * 2
    if objMin:
        return Z
    return -Z


def Holder_table(X=None, Y=None, objMin=True, is2Show=False, X_min=-10, X_max=10, Y_min=-10, Y_max=10):
    """
    global minma = -19.20850256788675
    x= ±9.664590028909654
    """
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = -np.abs(np.sin(X) * np.cos(Y) * np.exp(np.abs(1 - np.sqrt(X ** 2 + Y ** 2) / np.pi)))
        return (
         X, Y, Z, 0, 'Hölder table function', -20)
    Z = -np.abs(np.sin(X) * np.cos(Y) * np.exp(np.abs(1 - np.sqrt(X ** 2 + Y ** 2) / np.pi)))
    if objMin:
        return abs(Z - -19.20850256788675)
    return -Z


def Eggholder(X=None, Y=None, objMin=True, is2Show=False, X_min=-512, X_max=512, Y_min=-512, Y_max=512):
    """
    global minmum = −959.640662720850742
    x∗1,x∗2)=(512,404.231805123817)
    """
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
        Z = -(Y + 47) * np.sin(np.sqrt(np.abs(X / 2 + (Y + 47)))) - X * np.sin(np.sqrt(np.abs(X / 2 - (Y + 47))))
        return (
         X, Y, Z, 1000, 'Eggholder', -1000)
    Z = -(Y + 47) * np.sin(np.sqrt(np.abs(X / 2 + (Y + 47)))) - X * np.sin(np.sqrt(np.abs(X / 2 - (Y + 47))))
    if objMin:
        return abs(Z - -959.6406627208507)
    return -Z


def get_regret(fitness_now, objFcn, **kwarg):
    """
    return the regret
    """
    return objFcn(**kwarg) - fitness_now


if __name__ == '__main__':
    z_min = None
    X, Y, Z, z_max, title = Schwefel(is2Show=True) # Schwefel, Rastrigin
    draw_pic(X, Y, Z, z_max, title, z_min)
    
    
    