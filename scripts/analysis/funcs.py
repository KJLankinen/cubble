import numpy as np


def vorticity(VX, VY, delta_x, delta_y, acc=2):

    VX = VX[1:-1, 1: -1].copy()
    VY = VY[1:-1, 1: -1].copy()
    grid_w = np.empty(VX.shape)
    dim_x = grid_w.shape[0]
    dim_y = grid_w.shape[1]

    grid_vx_extend = np.column_stack((VX, VX))
    grid_vy_extend = np.row_stack((VY, VY))

    if acc == 2:
        calc_vorticity_2nd_accuracy(grid_w, grid_vx_extend, grid_vy_extend, delta_x, delta_y)
    elif acc == 4:
        calc_vorticity_4th_accuracy(grid_w, grid_vx_extend, grid_vy_extend, delta_x, delta_y)

    #grid_w = np.abs(grid_w)
    return grid_w


def calc_vorticity_2nd_accuracy(vort, vx, vy, delta_x, delta_y):
    for i in range(vort.shape[0]):
        for j in range(vort.shape[1]):
            vort[i, j] = 0.5 * (
                    (vy[i + 1, j] - vy[i - 1, j]) / delta_x
                    - (vx[i, j + 1] - vx[i, j - 1]) / delta_y
            )


def calc_vorticity_4th_accuracy(vort, vx, vy, delta_x, delta_y):
    for i in range(vort.shape[0]):
        for j in range(vort.shape[1]):
            vort[i, j] = (
                (
                        (1/12) * (vy[i - 2, j] - vy[i + 2, j])
                        - (2/3) * (- vy[i - 1, j] + vy[i + 1, j])
                ) / delta_x

                -

                (
                        (1/12) * (vx[i, j - 2] - vx[i, j + 2])
                        - (2/3) * (- vx[i, j - 1] + vx[i, j + 1])
                ) / delta_y
            )
