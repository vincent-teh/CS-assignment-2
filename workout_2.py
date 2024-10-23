from typing import Callable
from wsgiref.headers import tspecials
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate


def HatFunctions(x, j, xe):
    # x: centers x_0,x_1, ..., x_{N}
    # j: the index of \phi_j
    # xe: evaluation points
    # phi: a vector with the same size as xe; function values phi_j(xe)
    N = np.size(x) - 1
    m = np.size(xe)
    phi = np.zeros(m)
    if j > 0 and j < N:
        ind1 = np.where((xe > x[j-1]) & (xe <= x[j]))[0]
        ind2 = np.where((xe > x[j]) & (xe < x[j+1]))[0]
        phi[ind1] = (xe[ind1] - x[j-1])/(x[j]-x[j-1])
        phi[ind2] = (x[j+1] - xe[ind2])/(x[j+1]-x[j])
    elif j == 0:
        ind = np.where(xe < x[1])[0]
        phi[ind] = (x[1] - xe[ind])/(x[1]-x[0])
    elif j == N:
        ind = np.where(xe > x[N-1])[0]
        phi[ind] = (xe[ind] - x[N-1])/(x[N]-x[N-1])
    else:
        raise ValueError('Value j must be between 0 and length(x)-1')

    return phi


def StiffMatAssembler(x):
#
# Returns the assembled stiffness matrix A
# Input is a vector x of node coords
#
    N = np.size(x) - 1        # number of elements
    n = N-1;                  # dimension V_h^0 = size(x)-2
    S = np.zeros([n, n])      # initialize stiffnes matrix to zero
    for i in range (0,n-1):     # loop over elements
        h = x[i+1] - x[i]     # element length
        # assemble element stiffness
        S[i:i+2,i:i+2] = S[i:i+2,i:i+2] + np.array([[1, -1],[-1, 1]])/h

    h1 = x[1]-x[0]; h2 = x[2]-x[1]
    hn = x[n]-x[n-1]; hn1 = x[n+1]-x[n]
    S[0,0] = 1/h1+1/h2            # adjust for left BC
    S[n-1,n-1] = 1/hn+1/hn1        # adjust for right BC
    return S


def LoadVecAssembler(x, f):
#
# Returns the assembled load vector b
# Input is a vector x of node coords
#
    N = np.size(x) - 1        # number of elements
    n = N-1
    b = np.zeros(n)         # initialize load vector to zero
    for i in range (0,n-1):     # loop over elements
        h = x[i+1] - x[i]     # element length
        b[i:i+2] = b[i:i+2] + np.array([f(x[i]), f(x[i+1])])*h/2
    return b

def MassMatAssembler(x):
    # Number of nodes
    num_elements = np.size(x) - 1
    element_length = x[1] - x[0]
    num_nodes = num_elements - 1

    # Initialize the mass matrix
    M = np.zeros((num_nodes, num_nodes))
    element_mass_matrix = (element_length / 6) * np.array([[4, 1],
                                                           [1, 4]])

    for i in range(num_nodes - 1):
        M[i:i+2, i:i+2] += element_mass_matrix

    return M


def a_fn(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def calc_FEM(t: float, ut: np.ndarray):
    a, b = 0, np.pi   # interval [a,b]
    N = 100    # number of intervals
    x = np.linspace(a,b, N) # node coords

    S = StiffMatAssembler(x)
    M = MassMatAssembler(x)
    return np.linalg.solve(M, S@ut)


def main() -> None:
    a, b = 0, np.pi   # interval [a,b]
    N = 100    # number of intervals
    x = np.linspace(a,b, N-2) # node coords
    u0 = np.sin(x)

    sol = scipy.integrate.solve_ivp(calc_FEM, t_span=[a, b], y0=u0, t_eval=np.linspace(a, b, 1000))
    print(sol)
    plt.plot(sol.y[:][0])
    plt.show()


    # xi = calc_FEM()

    # # evaluate at eval points
    # xe = np.linspace(a, b ,100) # evaluation points
    # PHI = np.zeros([np.size(xe),np.size(x)]) # shape functions
    # for j in range(0, np.size(x)):
    #     PHI[:,j] = HatFunctions(x,j,xe)

    # uh = np.zeros(np.size(xe))
    # for j in range(1,np.size(x)-1):
    #     uh = uh + xi[j-1]*PHI[:,j]

    # plt.figure(figsize = (6, 4))
    # plt.plot(xe,uh,linestyle = '-', color = 'blue')
    # plt.show()


if __name__ == "__main__":
    main()