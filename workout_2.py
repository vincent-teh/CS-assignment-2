from logging import warning
from typing import Type
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


def calc_loop_error(
    FEM_: Type[FEM],
    x_start,
    x_end,
    exact_sol_norm,
    mesh_sizes: int | list[int],
    step_sizes: int | list[int],
) -> list[float]:
    if not isinstance(mesh_sizes, list):
        if not isinstance(step_sizes, list):
            x = np.linspace(x_start, x_end, mesh_size)
            u0 = np.sin(x)
            return FEM_(x).solve(u0[1:-1], t_steps=step_sizes)
        mesh_sizes = [mesh_sizes for _ in range(len(step_sizes))]
    if isinstance(step_sizes, list):
        if len(mesh_sizes) != len(step_sizes):
            raise ValueError(
                f"Array size of mesh {len(mesh_sizes)} & step sizes {len(step_sizes)} not match."
            )
    else:
        step_sizes = [step_sizes for _ in range(len(mesh_sizes))]

    errors = []
    for mesh_size, step_size in zip(mesh_sizes, step_sizes):
        x = np.linspace(x_start, x_end, mesh_size)
        u0 = np.sin(x)
        fem = FEM_(x)
        sol = fem.solve(u0[1:-1], t_steps=step_size)
        errors.append((abs(exact_sol_norm - fem.sol_norm)))
        print(f"Error: {errors[-1]:.4f}")

    return errors


def plot_error_mesh_analysis(FEM_: Type[FEM], t_steps: int = 10):
    x_start = 0
    x_end = np.pi
    t_final = 1

    x = np.linspace(x_start, x_end, 1000)
    u0 = np.sin(x)

    exact_fn_prime = lambda t, x: np.exp(-t) * np.cos(x)
    exact_sol_norm = scipy.integrate.trapezoid(exact_fn_prime(t_final, x) ** 2, x)
    print(f"Exact solution norm: {exact_sol_norm: .4f}")

    print(f"===============Error Analysis with Step Count {t_steps}===============")
    mesh_sizes = [5, 7, 10, 20]
    errors = calc_loop_error(
        FEM_, x_start, x_end, exact_sol_norm, mesh_sizes, step_sizes=t_steps
    )
    mesh_sizes = 1 / np.array(mesh_sizes)
    # plt.figure(figsize=(10.5, 5))
    plt.plot(mesh_sizes, errors, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Size of the mesh in log scale.")
    plt.ylabel("Measured error in log scale.")
    plt.title(f"Error vs Mesh Size Curve with k={t_steps}")


def plot_error_time_analysis(FEM_: Type[FEM]):
    x_start = 0
    x_end = np.pi
    t_steps = 5
    t_final = 1
    mesh_size = 7

    x = np.linspace(x_start, x_end, 1000)
    u0 = np.sin(x)

    exact_fn_prime = lambda t, x: np.exp(-t) * np.cos(x)
    exact_sol_norm = scipy.integrate.trapezoid(exact_fn_prime(t_final, x) ** 2, x)
    print(f"Exact solution norm: {exact_sol_norm: .4f}")

    print(f"===============Error Analysis with Step Count {t_steps}===============")
    step_sizes = [5, 10, 20, 50, 100]
    errors = calc_loop_error(
        FEM_, x_start, x_end, exact_sol_norm, mesh_size, step_sizes
    )
    plt.plot(step_sizes, errors, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of steps in log scale.")
    plt.ylabel("Measured error in log scale.")
    plt.title(f"Error vs Time Steps Curve with h={mesh_size}")


def plot_error_both_analysis(FEM_: Type[FEM]):
    x_start = 0
    x_end = np.pi
    t_final = 1
    mesh_size = 7

    x = np.linspace(x_start, x_end, 1000)
    u0 = np.sin(x)

    exact_fn_prime = lambda t, x: np.exp(-t) * np.cos(x)
    exact_sol_norm = scipy.integrate.trapezoid(exact_fn_prime(t_final, x) ** 2, x)
    print(f"Exact solution norm: {exact_sol_norm: .4f}")

    print(f"===============Error Improvement===============")
    mesh_sizes = [5, 6, 7, 8, 100]
    step_sizes = [5, 10, 20, 40, 100000]
    errors = calc_loop_error(
        FEM_, x_start, x_end, exact_sol_norm, mesh_sizes, step_sizes
    )
    plt.plot(step_sizes, errors, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of steps in log scale.")
    plt.ylabel("Measured error in log scale.")
    plt.title(f"Increase both no of meshes and steps.")


def main() -> None:
    methods = {
        "Explicit_Euler": ExplicitEuler,
        "Implicit_Euler": ImplicitEuler,
        "Trapezoidal": TrapezoidalEuler,
    }
    for name, fem in methods.items():
        plot_sol_over_time(fem, name)

    for name, fem in methods.items():
        plot_error_mesh_analysis(fem)
    plt.legend(methods.keys())
    plt.savefig(os.path.join(PATH, "mesh-analysis-total.eps"), format="eps")
    plt.show()

    for name, fem in methods.items():
        plot_error_mesh_analysis(fem, t_steps=20)
    plt.legend(methods.keys())
    plt.savefig(os.path.join(PATH, "mesh-analysis-total-converge.eps"), format="eps")
    plt.show()

    for name, fem in methods.items():
        plot_error_time_analysis(fem)
    plt.legend(methods.keys())
    plt.savefig(os.path.join(PATH, "step-analysis-total.eps"), format="eps")
    plt.show()

    plot_error_both_analysis(TrapezoidalEuler)
    plt.savefig(os.path.join(PATH, "both-mesh-step.eps"), format="eps")
    plt.show()

if __name__ == "__main__":
    main()
