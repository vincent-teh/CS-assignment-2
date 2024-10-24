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
        q1.ode, t_span=[0, t_final], y0=initial.to_array(), method="RK45", t_eval=np.linspace(0, t_final, 1000))


def calc_prob_interval(lower_bound, upper_bound, mean, std_dev):
    cdf_lower = scipy.stats.norm.cdf(lower_bound, loc=mean, scale=std_dev)
    cdf_upper = scipy.stats.norm.cdf(upper_bound, loc=mean, scale=std_dev)

    # The probability of the interval is the difference between the two CDF values
    return cdf_upper - cdf_lower

def main() -> None:
    T_FINAL = 10
    N_TRIAL = 1000
    FIG_PATH = os.path.join("report", "figures")

    a1t = np.zeros(N_TRIAL)
    a2t = np.zeros(N_TRIAL)

    printing = input("Press Y if you want to perform plotting:")

    for i in tqdm(range(N_TRIAL)):
        sol = solve_my_ivp(T_FINAL)
        a1t[i] = sol.y[0][-1]
        a2t[i] = sol.y[1][-1]
        if printing == "Y" and i < 3:
            q1.plot_pendulum(sol.y, f"Pendulum_Motions_{i+1}", FIG_PATH)

    if printing == "Y":
        q1.plot_hist_pendulum(a1t, "A", FIG_PATH)
        q1.plot_hist_pendulum(a2t, "B", FIG_PATH)

    print(f"mean of A1(T):   {a1t.mean():.4f}")
    print(f"std dev of A1(T): {a1t.std():.4f}")
    print(f"mean of A2(T):   {a2t.mean():.4f}")
    print(f"std dev of A2(T): {a2t.std():.4f}")
    print(f"Probability of A1(T): {calc_prob_interval(0, np.pi/4, a1t.mean(), a1t.std())}")
    print(f"Probability of A2(T): {calc_prob_interval(0, np.pi/4, a2t.mean(), a2t.std())}")
    print(f"Probability of A1(T) neg interval: {calc_prob_interval(-np.pi/4, 0, a1t.mean(), a1t.std()):.4f}")
    print(f"Probability of A2(T) neg interval: {calc_prob_interval(-np.pi/4, 0, a2t.mean(), a2t.std()):.4f}")
    return


if __name__ == "__main__":
    main()
