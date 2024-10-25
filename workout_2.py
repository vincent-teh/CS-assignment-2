from logging import warning
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.integrate

from myFEM import FEM, ExplicitEuler, ImplicitEuler, TrapezoidalEuler


PATH = os.path.join("report", "figures")


def plot_sol_over_time(FEM_: object, fig_name: str | None = None):
    x_start, x_end = 0, np.pi
    mesh_size = 7
    x = np.linspace(x_start, x_end, mesh_size)
    t_steps = 100
    t_start = 0
    t_final = 1

    exact_fn = lambda t, x: np.exp(-t) * np.sin(x)
    fem = FEM_(x)

    sol = fem.solve(u0=np.sin(x)[1:-1], t_steps=t_steps)

    normalize_sol = np.zeros([mesh_size, t_steps])
    normalize_sol[1:-1] = sol

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    T = np.linspace(t_start, t_final, t_steps)

    plot_step = 5

    for i, _ in enumerate(T):
        if i % (t_steps // plot_step):
            continue
        ax1.plot(x, normalize_sol[:, i])
    ax1.plot(x, normalize_sol[:, -1])
    ax1.legend(["Time: " + str(i) for i in range(plot_step)] + ["Final"])
    ax1.set_title("FEM Solution")
    ax1.set_xlabel("X")
    ax1.set_ylabel("u(t)")
    ax1.set_ylim(0, None)

    for i, t in enumerate(T):
        if i % (t_steps // plot_step):
            continue
        ax2.plot(x, exact_fn(t, x))
    ax2.plot(x, exact_fn(t_final, x))
    ax2.legend(["Time: " + str(i) for i in range(plot_step)] + ["Final"])
    ax2.set_title("Exact Solution")
    ax2.set_xlabel("X")
    ax2.set_ylabel("u(t)")
    ax2.set_ylim(0, None)

    plt.tight_layout()
    fig.suptitle(fig_name, fontsize=16)
    if fig_name is not None:
        plt.savefig(os.path.join(PATH, fig_name + ".eps"), format="eps")
    plt.show()


def plot_error_analysis(FEM_: object, fig_name: str|None = None):
    x_start = 0
    x_end = np.pi
    mesh_size = 7
    t_steps = 5
    t_final = 1

    x = np.linspace(x_start, x_end, 1000)

    exact_fn_prime = lambda t, x: np.exp(-t) * np.cos(x)
    exact_sol_norm = scipy.integrate.trapezoid(exact_fn_prime(t_final, x) ** 2, x)
    print(f"Exact solution norm: {exact_sol_norm: .4f}")

    print(f"===============Error Analysis with Step Count {t_steps}===============")
    errors: list[float] = []
    mesh_sizes = (4, 5, 7, 10, 15)
    u0 = np.sin(x[1:-1])
    for mesh_size in mesh_sizes:
        x = np.linspace(x_start, x_end, mesh_size)
        u0 = np.sin(x)
        fem = FEM_(x)
        sol = fem.solve(u0[1: -1], t_steps=t_steps)
        errors.append((abs(exact_sol_norm - fem.sol_norm)))
        print(f"Error: {errors[-1]:.4f}")

    mesh_sizes = 1 / np.array(mesh_sizes)
    plt.plot(mesh_sizes, errors)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Size of the mesh in log scale.")
    plt.ylabel("Measured error in log scale.")
    plt.title(f"Error vs Mesh Size Curve with k={t_steps}")
    if fig_name is not None:
        plt.savefig(os.path.join(PATH, fig_name + "-mesh-error.eps"), format="eps")
    plt.show()

    mesh_size = 10
    print(f"===============Error Analysis with Mesh Size {mesh_size}===============")
    t_steps = (5, 10, 20, 30, 50, 100)
    errors: list[float] = []
    for t_step in t_steps:
        x = np.linspace(x_start, x_end, mesh_size)
        u0 = np.sin(x)
        fem = FEM_(x)
        sol = fem.solve(u0[1: -1], t_steps=t_step)
        errors.append(abs(exact_sol_norm - fem.sol_norm))
        print(f"Error: {errors[-1]:.4f}")
    plt.plot(t_steps, errors)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of steps in log scale.")
    plt.ylabel("Measured error in log scale.")
    plt.title(f"Error vs Time Steps Curve with h={mesh_size}")
    if fig_name is not None:
        plt.savefig(os.path.join(PATH, fig_name + "-steps-error.eps"), format="eps")
    plt.show()


def main() -> None:
    # plot_sol_over_time(ExplicitEuler, "Explicit_Euler")
    # plot_sol_over_time(ImplicitEuler, "Implicit_Euler")
    # plot_sol_over_time(TrapezoidalEuler, "Trapezoidal")

    # plot_error_analysis(ExplicitEuler, "Explicit_Euler")
    # plot_error_analysis(ImplicitEuler, "Implicit_Euler")
    plot_error_analysis(TrapezoidalEuler, "Trapezoidal")


if __name__ == "__main__":
    main()
