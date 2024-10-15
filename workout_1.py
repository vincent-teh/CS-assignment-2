import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.stats
import os

import q1

from tqdm import tqdm

np.random.seed(42)


def solve_my_ivp(t_final):
    initial = q1.MyOdeParam(a1=np.random.normal(np.pi/4, 0.02),
                         a2=np.random.normal(np.pi/4, 0.02),
                         b1=0,
                         b2=0)

    return scipy.integrate.solve_ivp(
        q1.ode, t_span=[0, t_final], y0=initial.to_array(), method="RK45")


def main() -> None:
    T_FINAL = 10
    N_TRIAL = 1000
    FIG_PATH = os.path.join("report", "figures")

    a1t = np.zeros(N_TRIAL)
    a2t = np.zeros(N_TRIAL)

    for i in tqdm(range(N_TRIAL)):
        sol = solve_my_ivp(T_FINAL)
        a1t[i] = sol.y[0][-1]
        a2t[i] = sol.y[1][-1]
        if i < 3:
            q1.plot_pendulum(sol.y, f"Pendulum_Motions_{i+1}", FIG_PATH)

    q1.plot_hist_pendulum(a1t, "A", FIG_PATH)
    q1.plot_hist_pendulum(a2t, "B", FIG_PATH)

    print(f"mean of A1(T):   {a1t.mean():.4f}")
    print(f"std dev of A1(T): {a1t.std():.4f}")
    print(f"mean of A2(T):   {a2t.mean():.4f}")
    print(f"std dev of A2(T): {a2t.std():.4f}")
    return


if __name__ == "__main__":
    main()
