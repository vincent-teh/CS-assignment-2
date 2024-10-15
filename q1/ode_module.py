from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class MyOdeParam:
    a1: np.float64
    a2: np.float64
    b1: np.float64
    b2: np.float64

    def to_array(self):
        return list(asdict(self).values())

    @staticmethod
    def from_array(x: list[float]) -> MyOdeParam:
        return MyOdeParam(*x)


def ode(t: float, x: list[float]) -> list[float]:
    alpha = np.random.uniform(9.8, 10.2)
    v = MyOdeParam.from_array(x)

    da1 = v.b1
    da2 = v.b2
    db1 = -(np.sin(v.a1) + alpha * (v.a1 - v.a2))
    db2 = -np.sin(v.a1) + alpha * (v.a1 - v.a2)
    return [da1, da2, db1, db2]
