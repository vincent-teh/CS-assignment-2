from math import log
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate


PATH = os.path.join("report", "figures")


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
        mesh_size: int,
        t_start=0,
        t_final=1,
        t_steps=10,
        method: str | None = None,
    ):
        if not isinstance(mesh_size, int) or mesh_size < 3:
            raise ValueError("Mesh size must be larger than 3.")
        if t_start < 0:
            raise ValueError("t_start must be positive.")
        if t_final < t_start:
            raise ValueError("t_final must be larger than t_start.")
        if not isinstance(t_steps, int) or t_steps < 0:
            raise ValueError("t_steps must be an positive integer.")
        if method is None:
            method = "BDF"

        x = np.linspace(self.x_start, self.x_end, mesh_size)
        u0 = np.sin(x)
        self.sol = scipy.integrate.solve_ivp(
            self.update,
            t_span=[0, 1],
            y0=u0[1:-1],
            t_eval=np.linspace(t_start, t_final, t_steps),
            method=method,
        )
        return self.sol

    @property
    def sol_norm(self):
        if self.sol is None:
            return 0
        return self.sol.y[:, -1].T @ self.S @ self.sol.y[:, -1]

    @staticmethod
    def MassMatAssembler(x: np.ndarray) -> np.ndarray:
        """
        Determines the number of elements the mass matrix should have based on the size of X (N - 2). Generate the output tridiagonal matrix with base vector of [[2,1], [1,2]].

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
        Generate the stiffness matrix of size nxn where n is the size of the FEM mesh, N-2 assuming Drichlet condition. Also correspond to the phi'phi' matrix.

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


def main() -> None:
    x_start, x_end = 0, np.pi
    mesh_size = 50
    x = np.linspace(x_start, x_end, mesh_size)
    fem = FEM(x)
    t_steps = 10
    t_start = 0
    t_final = 1

    exact_fn = lambda t, x: np.exp(-t) * np.sin(x)

    sol = fem.solve(mesh_size=mesh_size)

    normalize_sol = np.zeros([mesh_size, t_steps])
    normalize_sol[1:-1] = sol.y

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(10):
        ax1.plot(x, normalize_sol[:, i])
    ax1.legend(["Time: " + str(i) for i in range(10)])
    ax1.set_title("FEM Solution")
    ax1.set_xlabel("X")
    ax1.set_ylabel("u(t)")
    ax1.set_ylim(0, None)

    for i in np.linspace(t_start, t_final, t_steps):
        ax2.plot(x, exact_fn(i, x))
    ax2.legend(["Time: " + str(i) for i in range(10)])
    ax2.set_title("Exact Solution")
    ax2.set_xlabel("X")
    ax2.set_ylabel("u(t)")
    ax2.set_ylim(0, None)

    plt.tight_layout()
    plt.show()

    exact_sol_norm = scipy.integrate.trapezoid(exact_fn(t_final, x) ** 2, x)
    print(f"Exact solution norm: {exact_sol_norm: .4f}")

    print(f"===============Error Analysis with Varying h===============")
    errors: list[float] = []
    mesh_sizes = (5, 10, 20, 30, 50, 60, 100)
    for mesh_size in mesh_sizes:
        x = np.linspace(x_start, x_end, mesh_size)
        fem = FEM(x)
        sol = fem.solve(mesh_size=mesh_size, t_steps=10)
        # print(f"FEM norm: {fem.sol_norm: .4f} ")
        errors.append(abs(exact_sol_norm - fem.sol_norm))
        print(f"Error: {errors[-1]:.4f}")

    plt.plot(mesh_sizes, errors)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of mesh in log scale.")
    plt.ylabel("Measured error in log scale.")
    plt.title("Error vs Mesh Size Curve with k=10")
    plt.savefig(os.path.join(PATH, "w2-mesh-error.eps"), format="eps")
    plt.show()

    print(f"===============Error Analysis with Varying k===============")
    t_steps = (10, 20, 30, 50, 100)
    mesh_size = 10
    errors: list[float] = []
    for t_step in t_steps:
        x = np.linspace(x_start, x_end, mesh_size)
        fem = FEM(x)
        sol = fem.solve(mesh_size, t_steps=t_step, method="Radau")
        errors.append(abs(exact_sol_norm - fem.sol_norm))
        print(f"Error: {errors[-1]:.4f}")
    plt.plot(t_steps, errors)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of steps in log scale.")
    plt.ylabel("Measured error in log scale.")
    plt.title("Error vs Time Steps Curve with h=10")
    plt.savefig(os.path.join(PATH, "w2-time-step-error.eps"), format="eps")
    plt.show()


if __name__ == "__main__":
    main()
