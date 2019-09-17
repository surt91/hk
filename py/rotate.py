from math import cos, sin, sqrt

import numpy as np


def build_rotmat(ux, uy, uz, theta):
    return np.array([
        [
            cos(theta)+ux**2*(1-cos(theta)),
            ux*uy*(1-cos(theta))-uz*sin(theta),
            ux*uz*(1-cos(theta))+uy*sin(theta)
        ],
        [
            uy*ux*(1-cos(theta))+uz*sin(theta),
            cos(theta)+uy**2*(1-cos(theta)),
            uy*uz*(1-cos(theta))-ux*sin(theta)
        ],
        [
            uz*ux*(1-cos(theta))-uy*sin(theta),
            uz*uy*(1-cos(theta))+ux*sin(theta),
            cos(theta)+uz**2*(1-cos(theta))
        ]
    ])


r = build_rotmat(-1/sqrt(2), 1/sqrt(2), 0, -0.955316618124509278163857102515757754243414695010005490959)
r2 = build_rotmat(0, 0, 1, -0.2617994)


def rotate_to_xy_plane(v):
    u = np.matmul(r, v)
    return np.matmul(r2, u)


if __name__ == "__main__":
    print("[0, 0, 0] ->", rotate_to_xy_plane(np.array([1, 0, 0])))
    print("[-1, 1, 0] ->", rotate_to_xy_plane(np.array([-1, 1, 0])))
    print("[-1, 0, 1] ->", rotate_to_xy_plane(np.array([-1, 0, 1])))
