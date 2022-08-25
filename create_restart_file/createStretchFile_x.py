#!/usr/share/python
#
# This program creates the zetastr.i file to be used as input for Hypgen.
# All i-nodes will recieve the same stretching.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import sys
import operator

def smoothstep(yi,ye,m):
  """
  https://en.wikipedia.org/wiki/Smoothstep#Variations
  """

  x = np.linspace(0.0,1.0,m-2)
  y = 6.0*x**5 - 15.0*x**4 + 10*x**3 
  
  dy = ye-yi
  y = y*dy
  y = y + yi
  
  y1 = np.ones(1)*yi
  y2 = np.ones(1)*ye
  
  y = np.concatenate((y1,y,y2))
  
  return y

if __name__ == "__main__":
    plt.close('all')


    ############################
    # Input parameters (Malha 1):

#####    ny = 76  # Total number of elements in normal direction
#####    amp1 = 0.0075 
#####    dy0      = 0.00035       # Minimum normal distance
#####    print(dy0)
#####    ny = 66  # Total number of elements in normal direction
#####    amp1 = 0.01267402938

#####    nflat1 = 100
#####    nramp1 = 25
#####    amp1 = 0.04969214181
#####    dy0      = 0.000025       # Minimum normal distance
#####    print(dy0)

#####    ny = 214
  
#    nflat1 = 100
#    nramp1 = 25
#    amp1 = 0.024878062722
#    dy0      = 0.000025       # Minimum normal distance
#    print(dy0)
#    ny = 286  
  
#    nflat1 = 100
#    nramp1 = 25
#    amp1 = 0.0248307498
#    dy0      = 0.000050       # Minimum normal distance
#    print(dy0)
#    ny = 257
  
    nflat1 = 100
    nramp1 = 25
    amp1 = 0.02505613469
    dy0      = 0.000100       # Minimum normal distance
    print(dy0)
    ny = 226  
  
#    nflat1 = 145
#    nramp1 = 25
#    amp1 = 0.01
#    dy0      = 0.000100       # Minimum normal distance
#    print(dy0)

#    ny = 200
  
#    nflat1 = 100
#    nramp1 = 25
#    amp1 = 0.01034
#    dy0      = 0.000200       # Minimum normal distance
#    print(dy0)

#    ny = 125  
  
#####    nflat1 = 150
#####    nramp1 = 50
#####    amp1 = 0.0393525447275
#####    dy0      = 0.000010       # Minimum normal distance
#####    print(dy0)

#####    ny = 285
  
    ############################

    # Initialize variables
    y = np.zeros(ny)    
    dy = np.zeros(ny)    
    strfunc = np.zeros(ny)    

    dy[0] = dy0
    y[0] = 0.0
    y[1] = y[0] + dy[0]
    strfunc[0] = 1.0

    for i in range(1, nflat1):
        dy[i] = dy0
        strfunc[i] = 1.0
        y[i] = y[i-1] + dy[i]
        print(dy[i],strfunc[i],y[i])

    aux = 1.0+smoothstep(0.0,amp1,nramp1)
    for k in range(0,nramp1):
        i = i + 1
        dy[i] = y[i-1] - y[i-2]
        strfunc[i] = aux[k]
        y[i] = y[i-1] + dy[i]*strfunc[i]  
        print(dy[i],strfunc[i],y[i])  
    
    while i < ny-1:
        i = i + 1
        dy[i] = y[i-1] - y[i-2]
        strfunc[i] = 1.0 + amp1
        y[i] = y[i-1] + dy[i]*strfunc[i]
        print(dy[i],strfunc[i],y[i])

#    min_index, min_value = min(enumerate(np.absolute(y-0.025)), key=operator.itemgetter(1))
#    print(min_index)
#    print(dy[min_index])
        
#    min_index, min_value = min(enumerate(np.absolute(y-1.0)), key=operator.itemgetter(1))
#    print(min_index)
#    print(dy[min_index])
#    print('max(y) = ', max(y))


    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].set_title('Stretching function')
    axarr[0].plot(strfunc[:ny])
    axarr[1].set_title('dy')
    axarr[1].semilogy(dy,'-o')
    axarr[2].set_title('y location') 
#    axarr[2].semilogy(y,'-')
    axarr[2].plot(y,'-')





    print(y[nflat1])



    plt.close()

    xx = np.hstack((-y[::-1],y[1:]))
    #xx = xx + 0.015
   
#    plt.plot(xx,xx*0.0,marker='o')
#    plt.show()

    np.savetxt('x.dat',xx)

    print('')
    print('Stretch File Created!!!')
    print('')
