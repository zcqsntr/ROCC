import numpy as np
import matplotlib.pyplot as plt
import matplotlib
A = np.array([[-0.01, -0.06],
            [-0.01, -0.01]])

'''keeping species alive'''
#C0 = np.array([[4.4444444444], [2.222222222]]) # noise
#C0 = np.array([[10.],[1./3.]]) # normal
#C0 = np.array([[10.],[5.5555555555]]) # recurrent

#SS = np.array([[1.05879964],[1.94953122]]) # noise
#SS = np.array([[5.86086775], [2.98599383]]) # normal
#SS = np.array([[5.27121812],[5.117429]]) # reccurrent

'''finer control 4.5 > N > 3.5'''
C0 = np.array([[5.555555556], [4.44444444]])
SS = np.array([[4.37927702], [4.03026668]])

# calculates all system parameters at steady state
q = 1.5
R = np.array(q - np.matmul(A,SS))
print('R: ', R)
Rmax = np.array([[7.],[10.]])
Km = np.array([[1.],[1.]])
C = R*Km/(Rmax-R)
print('C: ', C)
dC = q*(C0-C) - R*SS
print('dC: ' ,dC)


def sdot(N, t):
    N = np.array(N)
    dN = N*(R + np.matmul(A,N) - q)
    return dN

# range of the vector field
x_min = 0
x_max = 10
y_min = 0
y_max = 10

# number of points to plot on each axis
nx = 50
ny = 50

# make vector field
X,Y = np.meshgrid(np.linspace(x_min,x_max,nx), np.linspace(y_min,y_max,ny))
U,V = np.zeros(X.shape), np.zeros(Y.shape)
for i in range(nx):
    for j in range(ny):
        N = np.array([[X[i,j]], [Y[i,j]]])
        dN = sdot(N,0)
        U[i,j] = dN[0]
        V[i,j] = dN[1]

matplotlib.rcParams.update({'font.size': 22})


# plot vector field
plt.figure(figsize = (16.0,16.0))

plt.quiver(X, Y, U, V, linewidth=.5)

plt.xlabel("N1")
plt.ylabel("N2")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.plot(SS[0], SS[1], 'ro') # plot the steady state on the vector field

plt.savefig('vector_field.png')
