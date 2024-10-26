from logging import warning
import numpy as np
import scipy.integrate


class FEM:
    def __init__(self, x: np.ndarray) -> None:
        self.x = x
        self.S = self.StiffMatAssembler(x)
        self.M = self.MassMatAssembler(x)
        self.x_start = x[0]
        self.x_end = x[-1]

    def update(self, t, ut) -> np.ndarray:
        return np.linalg.solve(self.M, -self.S @ ut)

    def solve(
        self,
        u0,
        t_start: float=0,
        t_final: float=1,
        t_steps: int=10,
        method: str | None = None,
    ):
        if t_start < 0:
            raise ValueError("t_start must be positive.")
        if t_final < t_start:
            raise ValueError("t_final must be larger than t_start.")
        if t_steps < 0:
            raise ValueError("t_steps must be an positive integer.")
        if method is None:
            method = "BDF"

        self.sol = scipy.integrate.solve_ivp(
            self.update,
            t_span=[t_start, t_final],
            y0=u0[1:-1],
            t_eval=np.linspace(t_start, t_final, t_steps),
            method=method,
        ).y
        return self.sol


    @property
    def sol_norm(self):
        if self.sol is None:
            return 0
        return self.sol[:, -1].T @ -self.S @ self.sol[:, -1]


    @staticmethod
    def MassMatAssembler(x: np.ndarray) -> np.ndarray:
        """
        Determines the number of elements the mass matrix should have based on the size of X
        (N - 2). Generate the output tridiagonal matrix with base vector of [[2,1], [1,2]].

        Args:
            x (Arraylike): size of the ut

        Returns:
            np.ndarray: nxn matrix
        """
        num_elements = np.size(x) - 1
        element_length = x[1] - x[0]
        num_nodes = num_elements - 1

        # Initialize the mass matrix
        M = np.zeros((num_nodes, num_nodes))
        element_mass_matrix = (element_length / 6) * np.array([[2, 1], [1, 2]])
        for i in range(num_nodes - 1):
            M[i : i + 2, i : i + 2] += element_mass_matrix
        return M

    @staticmethod
    def StiffMatAssembler(x: np.ndarray) -> np.ndarray:
        """
        Generate the stiffness matrix of size nxn where n is the size of the FEM mesh, N-2
        assuming Drichlet condition. Also correspond to the phi'phi' matrix.

        Args:
            x (np.ndarray): input x coordinate.

        Returns:
            np.ndarray: nxn stiffness matrix.
        """
        N = np.size(x) - 1  # number of elements
        n = N - 1
        # dimension V_h^0 = size(x)-2
        S = np.zeros([n, n])  # initialize stiffnes matrix to zero
        for i in range(0, n - 1):  # loop over elements
            h = x[i + 1] - x[i]  # element length
            # assemble element stiffness
            S[i : i + 2, i : i + 2] = (
                S[i : i + 2, i : i + 2] + np.array([[1, -1], [-1, 1]]) / h
            )

        h1 = x[1] - x[0]
        h2 = x[2] - x[1]
        hn = x[n] - x[n - 1]
        hn1 = x[n + 1] - x[n]
        S[0, 0] = 1 / h1 + 1 / h2  # adjust for left BC
        S[n - 1, n - 1] = 1 / hn + 1 / hn1  # adjust for right BC
        return S


class ExplicitEuler(FEM):
    def solve(
        self,
        u0,
        t_start: float = 0,
        t_final: float = 1,
        t_steps: int = 10,
        method: str | None = None,
    ):
        u = u0
        h = (t_final - t_start) / t_steps
        max_eig = np.linalg.eigvals(np.linalg.inv(self.M) @ (h * -self.S)).max()
        if max_eig > 0:
            warning(f"Explicit Euler is unstable. Max eigenvalue: {max_eig}")
        sol = np.zeros([len(u), t_steps])
        for index, _ in enumerate(np.linspace(t_start, t_final, t_steps)):
            u = np.linalg.solve(self.M, (self.M - h * self.S) @ u)
            sol[:, index] = u
        self.sol = sol

        return sol


class ImplicitEuler(FEM):
    def solve(
        self,
        u0,
        t_start: float = 0,
        t_final: float = 1,
        t_steps: int = 10,
        method: str | None = None,
    ):
        u = u0
        sol = np.zeros([len(u), t_steps])
        h = t_start + (t_final - t_start) / t_steps
        for index in range(t_steps):
            u = np.linalg.solve((self.M + h * self.S), (self.M) @ u)
            sol[:, index] = u
        self.sol = sol

        return sol


class TrapezoidalEuler(FEM):
    def solve(
        self,
        u0,
        t_start: float = 0,
        t_final: float = 1,
        t_steps: int = 10,
        method: str | None = None,
    ):
        u = u0
        sol = np.zeros([len(u), t_steps])
        h = t_start + (t_final - t_start) / t_steps
        h = h / 2
        for index in range(t_steps):
            u = np.linalg.solve((self.M + h * self.S), (self.M - h * self.S) @ u)
            sol[:, index] = u
        self.sol = sol

        return sol
