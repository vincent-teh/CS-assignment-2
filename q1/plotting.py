import matplotlib.pyplot as plt
import numpy as np
import os


def plot_pendulum(y: np.ndarray, title: str, path: str) -> None:
    x = np.linspace(0, 10, len(y[0]))
    plt.plot(x, y[0])
    plt.plot(x, y[1])
    plt.legend(["Pendulum A", "Pendulum B"])
    plt.title(title)
    plt.xlabel("Time, s")
    plt.ylabel("Position of the Pendulum, Theta")
    plt.savefig(os.path.join(path, title+".eps"), format='eps')
    plt.show()


def plot_hist_pendulum(a_t: np.ndarray, title: str, path: str) -> None:
    plt.hist(a_t)
    plt.title("Final Position of Pendulum " + title)
    plt.xlabel("Final Position of the Pendulum, Theta")
    plt.ylabel("Count, n")
    plt.savefig(os.path.join(path, "Pendulum-" + title + ".eps"), format='eps')
    plt.show()
