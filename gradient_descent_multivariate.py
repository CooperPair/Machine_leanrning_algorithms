from sympy import *
import numpy as np

x = symbol('x')
y = symbol('y')
z = symbol('z')

f = x**2 + y**2 + z**2

# partial derivative wrt to x, y, z
fpx = f.diff(x)
fpy = f.diff(y)
fpz = f.diff(z)

grad = [fpx, fpy, fpz]

