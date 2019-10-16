import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def main():
    root_dir = sys.argv[1]

    fig, axes = plt.subplots(2, 2, sharex=True)

    for i, phi in enumerate(["phi_851", "phi_876"]):
        data = np.loadtxt(os.path.join(root_dir, phi, "snapshot.csv.10"), delimiter=',', skiprows=1)

        ed = data[:,11] / (data[:,3]**2)
        eds = np.sort(ed)
        
        q1 = eds[int(len(eds) / 4)]
        q2 = eds[int(2 * len(eds) / 4)]
        q3 = eds[int(3 * len(eds) / 4)]
        
        r1 = data[:,3][ed < q1]
        r2 = data[:,3][((ed > q1).astype(int) * (ed < q2).astype(int)).astype(bool)]
        r3 = data[:,3][((ed > q2).astype(int) * (ed < q3).astype(int)).astype(bool)]
        r4 = data[:,3][ed > q3]
        
        vr1 = data[:,8][ed < q1]
        vr2 = data[:,8][((ed > q1).astype(int) * (ed < q2).astype(int)).astype(bool)]
        vr3 = data[:,8][((ed > q2).astype(int) * (ed < q3).astype(int)).astype(bool)]
        vr4 = data[:,8][ed > q3]

        ed1 = ed[ed < q1]
        ed2 = ed[((ed > q1).astype(int) * (ed < q2).astype(int)).astype(bool)]
        ed3 = ed[((ed > q2).astype(int) * (ed < q3).astype(int)).astype(bool)]
        ed4 = ed[ed > q3]

        axes[i, 0].semilogy(r4, np.abs(vr4), '.', markersize=2.0)
        axes[i, 0].semilogy(r3, np.abs(vr3), '.', markersize=2.0)
        axes[i, 0].semilogy(r2, np.abs(vr2), '.', markersize=2.0)
        axes[i, 0].semilogy(r1, np.abs(vr1), '.', markersize=2.0)

        axes[i, 1].semilogy(r4, ed4, '.', markersize=2.0, label='Group 4')
        axes[i, 1].semilogy(r3, ed3, '.', markersize=2.0, label='Group 3')
        axes[i, 1].semilogy(r2, ed2, '.', markersize=2.0, label='Group 2')
        axes[i, 1].semilogy(r1, ed1, '.', markersize=2.0, label='Group 1')

    axes[1, 1].xaxis.label.set_fontsize(20)
    axes[1, 1].xaxis.set_label_coords(-0.1, -0.1)
    axes[1, 1].yaxis.label.set_fontsize(20)
    axes[1, 1].yaxis.set_label_coords(-0.12, 1.0)   
    
    axes[1, 0].yaxis.label.set_fontsize(20)
    axes[1, 0].yaxis.set_label_coords(-0.15, 1.0)

    axes[0, 0].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
    #axes[0, 0].tick_params(axis='y', which='both', labelsize=13, direction='in', pad=-10)
    axes[0, 1].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
    #axes[0, 1].tick_params(axis='y', which='both', right=True, left=False, labelleft=False, labelsize=13, direction='in', pad=-10)

    #axes[1, 0].tick_params(axis='x', which='both', labelsize=13, direction='in', pad=-20)
    #axes[1, 0].tick_params(axis='y', which='both', labelsize=13, direction='in', pad=-10)
    #axes[1, 1].tick_params(axis='x', which='both', labelsize=13, direction='in', pad=-20)
    #axes[1, 1].tick_params(axis='y', which='both', right=True, labelsize=13, direction='in', pad=-40)

    #for i in [0, 1]:
    #    for label in axes[i, 0].get_yticklabels():
    #        label.set_horizontalalignment('left')

    axes[1, 0].set_ylabel(r"$\vert v_R \vert$", rotation=0)

    axes[1, 1].set_xlabel(r"$R$")
    axes[1, 1].set_ylabel(r"$\frac{E}{R^2}$", rotation=0)

    axes[0, 1].legend(fontsize=15, loc='lower right', markerscale=5.0)
    plt.show()

if __name__ == "__main__":
    main()
