import headers.linear_algebra_pack as lag
import headers.pyplot_vision_pack as pvp
import headers.auxiliary_functions_pack as aux
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
from math import *


def main():
    plt.rcParams['figure.figsize'] = (12,8)
    plt.rcParams['font.size'] = 16
    plt.rcParams['image.cmap'] = 'gist_heat'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.prop_cycle'] = cycler(color=plt.get_cmap('tab20').colors)

    test_sample_size = 400

    x = np.arange(test_sample_size)
    trend = (np.sqrt(x + 4 * np.sqrt(x + 4)) - 10) ** 2
    period_1 = 2 * (np.sin(0.1 * np.pi * x))**2
    period_2 = 4 * (np.cos((2 * np.pi * (2 - x))/15)) ** 2

    np.random.seed(1234)
    white_noise = 4 * np.sqrt(np.random.rand(test_sample_size)) - 2.
    f = trend + period_1 + period_2 + white_noise
    l, k, n = aux.get_LKN(f)

    x = np.arange(n)
    plt.plot(x, f, lw=2.5)
    plt.plot(x, trend, 'r', lw=2.5)
    plt.plot(x, period_1, lw=2.5)
    plt.plot(x, period_2, lw=2.5)
    plt.plot(x, white_noise, 'black', alpha=0.4, lw=2.5)
    plt.legend(["Test Time Series ($f$)", "Trend", "$1^{st}$ Periodic Component", "$2^{nd}$ Periodic Component", "White Noise"])
    plt.xlabel("$x$")
    plt.ylabel(r"$f^{[\mathrm{test}]}(x)$")
    plt.title(r"The Time Series ($f^{[\mathrm{test}]}(x)$) and its Components")
    plt.show()

    _x = lag.trajectory_matrix(f, l)

    u, v, s = lag.specific_singular_value_decomposition(_x)
    a_span = lag.singular_elementary_decomposition(u, v, s, lag.rank(_x))

    d = len(a_span)
    ax = plt.matshow(_x)
    plt.xlabel(r"Column Vectors of $\mathbf{X}^{[%s]}$" % r"\mathrm{test}")
    plt.ylabel(r"Row Vectors of $\mathbf{X}^{[%s]}$" % r"\mathrm{test}")
    plt.colorbar(ax.colorbar, fraction=0.05)
    ax.colorbar.set_label("$f^{[\mathrm{test}]}(x)$")
    plt.title("The Trajectory Matrix of $f^{[\mathrm{test}]}(x)$")
    plt.show()

    pvp.plot_contribution(s)

    for i in range(15):
        plt.subplot(3, 5, i + 1)
        title = r"$\mathcal{W}_{" + str(i + 1) + r"}^{[\mathrm{test}]}$"
        pvp.plot_matrix(a_span[i], title)
    plt.tight_layout()
    plt.show()

    for i in range(15):
        plt.subplot(3, 5, i + 1)
        title = r"$\tilde{\mathcal{W}}_{" + str(i + 1) + r"}^{[\mathrm{test}]}$"
        pvp.plot_matrix(lag.book_diagonal_averaging(a_span[i]), title)
    plt.tight_layout()
    plt.show()

    correlation_matrix = lag.w_correlation_matrix(f, a_span)

    figure = plt.imshow(correlation_matrix, cmap=plt.get_cmap('binary'))
    plt.xlabel(r"$\tilde{\mu}^{(j)}$")
    plt.ylabel(r"$\tilde{\mu}^{(j)}$")
    plt.colorbar(figure.colorbar, fraction=0.05)
    plt.clim(0,1)
    figure.colorbar.set_label(r"$\mathbb{W}^{[\mathrm{test}]}_{i,j}$")
    plt.title(r"$w$-correlation matrix for projected groupings of $\tilde{\mu}^{(i,j)}$ of test time series")
    plt.show()

    f_predicted_tr = lag.fast_diagonal_averaging(a_span[[0, 1, 2]].sum(axis=0))
    f_predicted_p2 = lag.fast_diagonal_averaging(a_span[[3, 4]].sum(axis=0))
    f_predicted_p1 = lag.fast_diagonal_averaging(a_span[[5, 6]].sum(axis=0))
    f_predicted_noise = lag.fast_diagonal_averaging(a_span[7:].sum(axis=0))

    a, b, c, d = 0, 0, 0, 0
    text_1, text_2, text_3, text_4 = '', '', '', ''
    for i in range(2):
        if i == 1:
            a, b, c, d = -4, 1, 2, 1
            text_1, text_2, text_3, text_4 = '- 4', '+ 1', '+ 2', '+ 1'
        components = [(trend + a, "Trend", r"$f^{(\mathrm{trend})}$",
                       f_predicted_tr + a, r"$\tilde{F}^{(\mathrm{trend})} %s$" % text_1),
                      (period_1, "$1^{st}$ Periodic Component", r"$f^{(\mathrm{period})_\alpha}$",
                       f_predicted_p1 + b, r"$\tilde{F}^{(\mathrm{period})_\alpha} %s$" % text_2),
                      (period_2, "$2^{nd}$ Periodic Component", r"$f^{(\mathrm{period})_\beta}$",
                       f_predicted_p2 + c, r"$\tilde{F}^{(\mathrm{period})_\beta} %s$" % text_3),
                      (white_noise, "White Noise", r"$\eta(x)$",
                       f_predicted_noise + d, r"$\tilde{\eta}(x) %s$" % text_4)]

        figure = plt.figure()
        position = 1
        for known_component, name, label_1, estimated_component, label_2 in components:
            element = figure.add_subplot(2, 2, position)
            element.plot(x, known_component, label=label_1, color='steelblue', linestyle='--', lw=2.5)
            element.plot(x, estimated_component, label=label_2, color='darkred', alpha=0.5)
            element.set_title(name, fontsize=16)
            element.set_xticks([50, 100, 150, 200, 250, 300, 350])
            position += 1
            plt.legend()
        figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
