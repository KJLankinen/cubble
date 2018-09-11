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

def asd(x, y):

    alpha = 0.498
    tau = 24.0
    plot_label_str = r"$(\frac{t^*}{\tau_c})^{\alpha}, \alpha = $" + str(alpha) + r"$, \tau_c = $" + str(tau)
    line_y = pow(x / tau, alpha)

    fig = plt.figure()
    ax = fig.add_subplot(111) # nrows, ncols, index
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
        
    ax.semilogx(x, y, 'r+', linewidth=2.0, label="data")
    ax.semilogx(x[30:], line_y[30:], 'k--', linewidth=2.0, label=plot_label_str)

    ax.set_xlim(1, 1000)
    
    ax.set_xlabel(r"$t^*$")
    ax.set_ylabel(r"$\frac{R}{<R_{in}>}$")
    #ax.patch.set_facecolor((0.3, 0.3, 1.0))
    
    ax.grid(1)
    ax.legend(loc=2, prop={'size' : 20})
    
    plt.show()
    
def main():
    data = np.loadtxt("data/collected_data.dat")
    x = data[:, 0]
    y = data[:, 1]

    #fit_to_data(x, y)
    asd(x, y)

if __name__ == "__main__":
    main()
