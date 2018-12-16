# implementation of gradient descent form scratch
# in this we start at any position and move into any random direction in negative direction of the graident function 

import numpy as np
import matplotlib.pyplot as plt

# creating function
function = lambda x: (x ** 3)-(3 *(x ** 2))+7

# getting 1000 evenly spaced number between -1 and 3 (for getting sttep curve)

x = np.linspace(-1,3,500)

plt.plot(x, function(x))
plt.show()

def deriv(x):
    x_deriv = 3* (x**2) - (6 * (x))
    return x_deriv


def step(x_new, x_prev, precision, l_r):
    
    # create empty lists where the updated values of x and y wil be appended during each iteration
    x_list, y_list = [x_new], [function(x_new)]
    # keep looping until our desired precision
    while abs(x_new - x_prev) > precision:
        
        # change the value of x
        x_prev = x_new
        
        # get the derivation of the old value of x
        d_x = - deriv(x_prev)
        
        # get your new value of x by adding the previous, the multiplication of the derivative and the learning rate
        x_new = x_prev + (l_r * d_x)
        
        # append the new value of x to a list of all x-s for later visualization of path
        x_list.append(x_new)
        
        # append the new value of y to a list of all y-s for later visualization of path
        y_list.append(function(x_new))

    print ("Local minimum occurs at: "+ str(x_new))
    print ("Number of steps: " + str(len(x_list)))
    
    
    plt.subplot(1,2,2)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,function(x), c="r")
    plt.title("Gradient descent")
    plt.show()

    plt.subplot(1,2,1)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,function(x), c="r")
    plt.xlim([1.0,2.1])
    plt.title("Zoomed in Gradient descent to Key Area")
    plt.show()

#Implement gradient descent (all the arguments are arbitrarily chosen)

step(0.5, 0, 0.001, 0.05)