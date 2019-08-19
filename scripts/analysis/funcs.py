import numpy as np
import uuid
from shutil import copyfile


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


def get_variables(df):
    return df["x"], df["y"], df["z"], df["a"], df["vx"], df["vy"], df["vz"], df["path"], df["dist"]


def save_result(*, fig, save_folder, name, script_path):
    """
    Saves the matplotlib figure and the script that was used to generate it

    Args:
        fig (matplotlib figure object): figure to be saved
        save_folder (Path object): Path to folder in which figure will be saved
        name (str): Name of the plot file
        script_path (Path object): Path to the script that generated figure

    """

    uid = str(uuid.uuid4())[:8]  # generate unique ID

    script_copy_save_folder = save_folder / "scripts_used_for_generation"  # folder in which script will be copied
    script_copy_path = script_copy_save_folder / f"{uid}-{script_path.name}"  # Copied script file name and path

    if not save_folder.exists():  # Create folder for saving if does not yet exist
        save_folder.mkdir(exist_ok=False)
    if not script_copy_save_folder.exists():
        script_copy_save_folder.mkdir(exist_ok=False)

    name += f"-{uid}.pdf"  # name of figure file
    fig.savefig(str((save_folder / name).resolve()))  # save the figure as pdf
    copyfile(str(script_path.resolve()), str(script_copy_path.resolve()))  # Copy the script used to generate the plot
