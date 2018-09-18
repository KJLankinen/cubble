import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

##########
# Fitting the data -- Least Squares Method
##########

# Power-law fitting is best done by first converting
# to a linear equation and then fitting to a straight line.
# Note that the `logyerr` term here is ignoring a constant prefactor.
#
#  y = a * x^b
#  log(y) = log(a) + b*log(x)
#

def fit_to_data(xdata, ydata):
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    yerr = np.ones(ydata.shape)
    logyerr = yerr / ydata
    
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    powerlaw = lambda x, amp, index: amp * (x**index)
    
    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy, logyerr), full_output=1)
    
    pfinal = out[0]
    covar = out[1]
    print pfinal
    print covar
    
    index = pfinal[1]
    amp = 10.0**pfinal[0]
    
    indexErr = np.sqrt( covar[1][1] )
    ampErr = np.sqrt( covar[0][0] ) * amp
    
    ##########
    # Plotting data
    ##########
    
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(xdata, powerlaw(xdata, amp, index))     # Fit
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    plt.text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
    plt.text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
    plt.title('Best Fit Power Law')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(1, 11)

    plt.subplot(2, 1, 2)
    plt.loglog(xdata, powerlaw(xdata, amp, index))
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    plt.xlabel('X (log scale)')
    plt.ylabel('Y (log scale)')
    plt.xlim(1.0, 11)

    plt.show()

def asd(data_loc, ax, plot_label):
    data = np.loadtxt(data_loc)
    x = data[:, 0]
    y = data[:, 1]
    
    ax.loglog(x, y, '.', linewidth=2.5, label=plot_label)
    
def main():
    alpha = 0.5
    xlim = 50000

    fig = plt.figure()
    ax = fig.add_subplot(111) # nrows, ncols, index
    
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.set_xlim(1, xlim)
    ax.set_ylim(0.65, 10)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\frac{R}{<R_{in}>}$")
    ax.grid(1)

    ax.loglog(np.linspace(15, 20000, 10000), pow(np.linspace(15, 20000, 10000) / 15.0, alpha), 'k--', linewidth=2.0, label=r"$a\tau^b, b=$" + str(alpha))
    
    for i in range(17):
        path = "multiple_runs_data/" + str(i) + "/data/collected_data.dat"
        label_str = r"$\phi=$" + str(0.900 - i * 0.025) + r", $\kappa=$" + str(0.01)
        asd(path, ax, label_str)

    ax.legend(loc=2, prop={'size' : 20})

    plt.show()

if __name__ == "__main__":
    main()
