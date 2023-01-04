import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt

def fid(m0, T2, f, t):
    return m0 * np.exp(-t / T2) * np.exp(1j *2 * np.pi * f * t)

t = np.linspace(0,10,200)
f = fid(1,2,3,t)
plt.plot(f.real, f.imag)
plt.plot(t, f.real)
print()