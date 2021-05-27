import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as pdates
import headers.linear_algebra_pack as lag
import headers.auxiliary_functions_pack as aux


def plot_matrix(matrix, title=""):
    plt.imshow(matrix)
    plt.title(title)
    plt.xticks([])  # nothing to show, only the matrix plots themselves => no specific axes ticks for matrices
    plt.yticks([])


def plot_contribution(spectrum):
    # spectrum_copy = np.insert(spectrum, 0, 0, axis=0)
    sum_spec = spectrum.sum()
    plt.plot(spectrum / sum_spec * 100, 'red', label='$\mathcal{R}(i)$', lw=2.5)
    plt.plot(spectrum.cumsum() / sum_spec * 100, 'firebrick', label='$\mathcal{C}(i)$', lw=2.5)
    plt.style.use('seaborn-talk')
    plt.title("Relative [$\mathcal{R}(i)$] and Cumulative [$\mathcal{C}(i)$] contributions [in %] of $\mathcal{W}_i$ to $\mathbf{X}$")
    plt.xlabel("$i$")
    plt.ylabel("Contribution, %")
    plt.xlim(0, 15)
    plt.xticks(range(1, 15))
    plt.legend()
    plt.show()


def full_svd_process_and_plot_snp500(date, np_time_series, string_period, num_of_ts='', plot=False):
    l, k, n = aux.get_LKN(np_time_series)
    x = np.arange(n)
    _X = lag.trajectory_matrix(np_time_series, l)
    u, v, s = lag.specific_singular_value_decomposition(_X)
    d = lag.rank(_X)
    a_span = lag.singular_elementary_decomposition(u, v, s, d)

    if plot:
        ''' 1st: plotting the time series itself '''
        figure, ax = plt.subplots()
        ax.plot(date.astype('datetime64',copy=False), np_time_series, lw=2.5)
        plt.legend([r"$\mathcal{F}_{%s}(x)$" % num_of_ts])
        plt.xlabel("$Dates$")
        plt.ylabel(r"$\mathcal{F}_{%s}(x)$" % num_of_ts)
        plt.title(r"Time series of S&P500* index ($\mathcal{F}_{%s}(x)$), period: %s (%s months)" % (num_of_ts, string_period, n))

        five_yr = pdates.YearLocator(5)
        ax.xaxis.set_major_locator(five_yr)

        one_yr = pdates.YearLocator(1)
        ax.xaxis.set_minor_locator(one_yr)

        ax.xaxis.set_major_formatter(pdates.DateFormatter('%Y-%m'))
        datemin = np.datetime64(date[0], 'M') - np.timedelta64(6, 'M')
        datemax = np.datetime64(date[len(date) - 1], 'M') + np.timedelta64(6, 'M')

        ax.set_xlim(datemin, datemax)
        ax.format_xdata = pdates.DateFormatter('%Y-%m')
        figure.autofmt_xdate()

        plt.show()

        ''' 2nd: plotting the trajectory matrix of $\mathcal{F}(x)$ '''
        figure = plt.matshow(_X)
        plt.xlabel(r"Column Vectors of $\mathbf{X}^{[%s]}$" % num_of_ts)
        plt.ylabel(r"Row Vectors of $\mathbf{X}^{[%s]}$" % num_of_ts)
        plt.colorbar(figure.colorbar, fraction=0.05)
        figure.colorbar.set_label(r"$\mathcal{F}_{%s}(x)$" % num_of_ts)
        plt.title(r"The Trajectory Matrix of $\mathcal{F}_{%s}(x)$" % num_of_ts)
        plt.show()

        ''' 3rd: plotting first 15 elementary matrices obdained by SVD of $\mathbf{X}$ '''
        if 15 <= d:
            for i in range(15):
                plt.subplot(3, 5, i + 1)
                title = r"$\mathcal{W}_{%s}^{[%s]}$" % (str(i), num_of_ts)
                plot_matrix(a_span[i], title)
            plt.tight_layout()
            plt.show()
        else:
            print("\n\nSorry, unable to print first 15 elementary matrices of X due to internal length-value error.\n\n")

        ''' 4th: plotting both relative and cumulative contribution of $$'''
        plot_contribution(s)

        ''' 5th: plotting the $w$-correlation matrix for the reconstructed diagonally averaged time series' components '''
        w_corr = lag.w_correlation_matrix(np_time_series, a_span)
        figure = plt.imshow(w_corr, cmap=plt.get_cmap('binary'))
        plt.xlabel(r"$\tilde{\mu}^{(i)}$")
        plt.ylabel(r"$\tilde{\mu}^{(j)}$")
        plt.colorbar(figure.colorbar, fraction=0.05)
        figure.colorbar.set_label(r"$\mathbb{W}^{[%s]}_{i,j}$" % num_of_ts)
        plt.clim(0, 1)
        plt.title(r"$w$-correlation matrix for projected groupings of $\tilde{\mu}^{(i,j)}$ (%s)" % string_period)
        plt.show()
    return a_span, x


def plot_ts_and_components(_range, list_of_specials, list_noisy, str_period, num_of_ts):
    figure, ax = plt.subplots()

    plt.xlabel("$Dates$")
    plt.ylabel(r"$\mathcal{F}_{%s}(x)$" % num_of_ts)
    plt.title(r"Time series of S&P500* index ($\mathcal{F}_{%s}(x)$) and its"
              r" components, period: %s (%s months)" % (num_of_ts, str_period, len(_range)))

    for plot_components, names in list_of_specials:
        ax.plot(_range.astype('datetime64',copy=False), plot_components, label=(names % num_of_ts), lw=2.5)
    ax.plot(_range.astype('datetime64',copy=False), list_noisy[0][0], color='red',
              label=(list_noisy[0][1] % num_of_ts), lw=2.5)
    ax.plot(_range.astype('datetime64',copy=False), list_noisy[1][0], color='black',
              label=(list_noisy[1][1] % num_of_ts), lw=2.5)

    plt.legend(loc=(0.98, 0.25), fancybox=True, shadow=True)
    five_yr = pdates.YearLocator(5)
    ax.xaxis.set_major_locator(five_yr)

    one_yr = pdates.YearLocator(1)
    ax.xaxis.set_minor_locator(one_yr)

    ax.xaxis.set_major_formatter(pdates.DateFormatter('%Y-%m'))
    datemin = np.datetime64(_range[0], 'M') - np.timedelta64(6, 'M')
    datemax = np.datetime64(_range[len(_range) - 1], 'M') + np.timedelta64(6, 'M')

    ax.set_xlim(datemin, datemax)
    ax.format_xdata = pdates.DateFormatter('%Y-%m')
    figure.autofmt_xdate()
    plt.show()


def plot_full_snp500(_range, list_of_specials_01, y_scale="linear"):
    figure, ax = plt.subplots()

    plt.xlabel("$Dates$")
    plt.ylabel(r"$\mathcal{F}(x)$")
    plt.title(r"Time series of S&P500* index ($\mathcal{F}(x)$) and its "
              r"roughly estimated elements, full period [1926-2018*]")
    plt.yscale(y_scale)

    flag = False
    for plot_components, names, colored, indicator in list_of_specials_01:
        if indicator:
            if colored != '':
                ax.plot(_range.astype('datetime64',copy=False), plot_components, color=colored, label=names, lw=2.5)
            else:
                ax.plot(_range.astype('datetime64',copy=False), plot_components, label=names, lw=2.5)
        flag |= indicator

    if flag:
        plt.legend(loc=(0, 0.25), fancybox=True, shadow=True)

        five_yr = pdates.YearLocator(5)
        ax.xaxis.set_major_locator(five_yr)

        one_yr = pdates.YearLocator(1)
        ax.xaxis.set_minor_locator(one_yr)

        ax.xaxis.set_major_formatter(pdates.DateFormatter('%Y-%m'))
        datemin = np.datetime64(_range[0], 'M') - np.timedelta64(6, 'M')
        datemax = np.datetime64(_range[len(_range) - 1], 'M') + np.timedelta64(6, 'M')

        ax.set_xlim(datemin, datemax)
        ax.format_xdata = pdates.DateFormatter('%Y-%m')
        figure.autofmt_xdate()

        plt.show()
    else:
        print("\n\nSorry, nothing to print.\n\n")


### Checks ###


def main():
    # TODO: play - use some data with my functions, with accordance to their definition.
    pass


if __name__ == "__main__":
    main()
