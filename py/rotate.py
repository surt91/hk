from math import cos, sin, sqrt

import numpy as np

# -atan(sqrt(2))
theta = -0.955316618124509278163857102515757754243414695010005490959
ux = -1/sqrt(2)
uy = 1/sqrt(2)
uz = 0
r = np.array([
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

# TODO: also rotate it around the z axis and translate to somewhere sensible

def rotate_to_xy_plane(v):
    return np.matmul(r, v)
