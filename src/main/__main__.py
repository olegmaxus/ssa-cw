import headers.linear_algebra_pack as lag
import headers.auxiliary_functions_pack as aux
import headers.pyplot_vision_pack as pvp
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~ MATPLOTLIB GLOBAL PRESETTING

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16
plt.rcParams['image.cmap'] = 'gist_heat'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.prop_cycle'] = cycler(color=plt.get_cmap('tab20').colors)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~ DATA EXTRACTION AND ALIGNMENT

F = aux.read_graph_data_snp500(sys.argv[1])
F_1 = F[:300]
F_2 = F[300:600]
F_3 = F[600:900]
F_4 = F[900:]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~ REQUESTING TO PLOT PARTIALS

# asking to plot the parts of the huge TS and their mip SSA-derived components.
string_yn = str(input("Hello!\nPlease, specify:\nShould the program plot the period-derived "
                      "components of the initial time series?\nPlease, print: + or -: "))
other_yn = str(input("\nPlease, specify:\nShould the program plot the SVD-derivations for "
                      "each component?\nPlease, print: + or -: "))

# ~~~~~~~~~~~~~~~~~~ #

if string_yn == "+":
    string_yn = True
elif string_yn == "-":
    string_yn = False
else:
    string_yn = True
    print("\nUnrecognizable input, assuming: positive answer.\n")

# ~~~~~~~~~~~~~~~~~~ #

if other_yn == "+":
    other_yn = True
elif other_yn == "-":
    other_yn = False
else:
    other_yn = False
    print("\nUnrecognizable input, assuming: negative answer.\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~ SVD AND ELEMENTARY GROUPINGS

'''
    The below groupings have been done with accordance
    to the distribution of highest correlations along 
    the w-correlation matrices for each unique time se-
    ries.
'''

''' Performing SVD for each time period's data set '''

x = np.arange(len(F))
elementary_span_1, x_1 = pvp.full_svd_process_and_plot_snp500(F_1, "1926 - 1951", 1, plot=other_yn)
elementary_span_2, x_2 = pvp.full_svd_process_and_plot_snp500(F_2, "1951 - 1976", 2, plot=other_yn)
elementary_span_3, x_3 = pvp.full_svd_process_and_plot_snp500(F_3, "1976 - 2001", 3, plot=other_yn)
elementary_span_4, x_4 = pvp.full_svd_process_and_plot_snp500(F_4, "2001 - 2018*", 4, plot=other_yn)

''' 1st: '''
f_predicted_1_tr = lag.fast_diagonal_averaging(elementary_span_1[0])
f_predicted_1_p1 = lag.fast_diagonal_averaging(elementary_span_1[[1, 2]].sum(axis=0))
f_predicted_1_p2 = lag.fast_diagonal_averaging(elementary_span_1[[3, 4, 5, 6, 9]].sum(axis=0))
f_predicted_1_p3 = lag.fast_diagonal_averaging(elementary_span_1[[7, 8]].sum(axis=0)) + \
                   lag.fast_diagonal_averaging(elementary_span_1[10:13].sum(axis=0))
f_predicted_1_noise = lag.fast_diagonal_averaging(elementary_span_1[13:].sum(axis=0))

f_1_overall_no_noise = f_predicted_1_tr + f_predicted_1_p1 + f_predicted_1_p2 + f_predicted_1_p3

collection_ts_1 = [(F_1, r"$\mathcal{F}_{%s}}(x)$"),
                   (f_predicted_1_tr, r"$\tilde{F}_{%s}^{(\mathrm{trend})}$"),
                   (f_predicted_1_p1, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\alpha}}$"),
                   (f_predicted_1_p2, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\beta}}$"),
                   (f_predicted_1_p3, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\gamma}}$")]

collection_noisy_1 = [(f_1_overall_no_noise, r"$\tilde{F}_{%s}(x)$"),
                      (f_predicted_1_noise, r"$\tilde{\eta}_{%s}(x)$")]
# plotting #
if string_yn:
    pvp.plot_ts_and_components(x_1, collection_ts_1, collection_noisy_1, "1926 - 1951", 1)

''' 2nd: '''
f_predicted_2_tr = lag.fast_diagonal_averaging(elementary_span_2[0])
f_predicted_2_p1 = lag.fast_diagonal_averaging(elementary_span_2[[1, 2, 3]].sum(axis=0))
f_predicted_2_p2 = lag.fast_diagonal_averaging(elementary_span_2[4:7].sum(axis=0))
f_predicted_2_p3 = lag.fast_diagonal_averaging(elementary_span_2[[7, 8]].sum(axis=0))
f_predicted_2_p4 = lag.fast_diagonal_averaging(elementary_span_2[[9, 10]].sum(axis=0))
f_predicted_2_p5 = lag.fast_diagonal_averaging(elementary_span_2[[11, 12]].sum(axis=0))
f_predicted_2_noise = lag.fast_diagonal_averaging(elementary_span_2[13:].sum(axis=0))

f_predicted_2_p23 = f_predicted_2_p2 + f_predicted_2_p3
f_predicted_2_p45 = f_predicted_2_p4 + f_predicted_2_p5

f_2_overall_no_noise = f_predicted_2_tr + f_predicted_2_p1 + f_predicted_2_p23 + f_predicted_2_p45

collection_ts_2 = [(F_2, r"$\mathcal{F}_{%s}}(x)$"),
                   (f_predicted_2_tr, r"$\tilde{F}_{%s}^{(\mathrm{trend})}$"),
                   (f_predicted_2_p1, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\alpha}}$"),
                   (f_predicted_2_p23, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\beta}}$"),
                   (f_predicted_2_p45, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\gamma}}$")]

collection_noisy_2 = [(f_2_overall_no_noise, r"$\tilde{F}_{%s}(x)$"),
                      (f_predicted_2_noise, r"$\tilde{\eta}_{%s}(x)$")]
# plotting #
if string_yn:
    pvp.plot_ts_and_components(x_2, collection_ts_2, collection_noisy_2, "1951 - 1976", 2)

''' 3rd: '''
f_predicted_3_tr = lag.fast_diagonal_averaging(elementary_span_3[[0, 1, 2]].sum(axis=0))
f_predicted_3_p1 = lag.fast_diagonal_averaging(elementary_span_3[[3, 4]].sum(axis=0))
f_predicted_3_p2 = lag.fast_diagonal_averaging(elementary_span_3[[5, 6]].sum(axis=0))
f_predicted_3_p3 = lag.fast_diagonal_averaging(elementary_span_3[[7, 8]].sum(axis=0))
f_predicted_3_noise = lag.fast_diagonal_averaging(elementary_span_3[9:].sum(axis=0))

f_3_overall_no_noise = f_predicted_3_tr + f_predicted_3_p1 + f_predicted_3_p2 + f_predicted_3_p3

collection_ts_3 = [(F_3, r"$\mathcal{F}_{%s}}(x)$"),
                   (f_predicted_3_tr, r"$\tilde{F}_{%s}^{(\mathrm{trend})}$"),
                   (f_predicted_3_p1, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\alpha}}$"),
                   (f_predicted_3_p2, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\beta}}$"),
                   (f_predicted_3_p3, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\gamma}}$")]

collection_noisy_3 = [(f_3_overall_no_noise, r"$\tilde{F}_{%s}(x)$"),
                      (f_predicted_3_noise, r"$\tilde{\eta}_{%s}(x)$")]
# plotting #
if string_yn:
    pvp.plot_ts_and_components(x_3, collection_ts_3, collection_noisy_3, "1976 - 2001", 3)

''' 4th: '''
f_predicted_4_tr = lag.fast_diagonal_averaging(elementary_span_4[[0]].sum(axis=0))
f_predicted_4_p1 = lag.fast_diagonal_averaging(elementary_span_4[[1, 2]].sum(axis=0))
f_predicted_4_p2 = lag.fast_diagonal_averaging(elementary_span_4[3:6].sum(axis=0))
f_predicted_4_p3 = lag.fast_diagonal_averaging(elementary_span_4[[6, 7]].sum(axis=0))
f_predicted_4_noise = lag.fast_diagonal_averaging(elementary_span_4[8:].sum(axis=0))

f_4_overall_no_noise = f_predicted_4_tr + f_predicted_4_p1 + f_predicted_4_p2 + f_predicted_4_p3

collection_ts_4 = [(F_4, r"$\mathcal{F}_{%s}}(x)$"),
                   (f_predicted_4_tr, r"$\tilde{F}_{%s}^{(\mathrm{trend})}$"),
                   (f_predicted_4_p1, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\alpha}}$"),
                   (f_predicted_4_p2, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\beta}}$"),
                   (f_predicted_4_p3, r"$\tilde{F}_{%s}^{(\mathrm{period})_{\gamma}}$")]

collection_noisy_4 = [(f_4_overall_no_noise, r"$\tilde{F}_{%s}(x)$"),
                      (f_predicted_4_noise, r"$\tilde{\eta}_{%s}(x)$")]
# plotting #
if string_yn:
    pvp.plot_ts_and_components(x_4, collection_ts_4, collection_noisy_4, "2001 - 2018*", 4)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~ GROUPING PERIOD PLOTS ONTO ROUGH FULL-RANGE ESTIMATION

''' 
    Plotting the partial time series' and their components
    has been done. Now, let's try to "reconstruct" full 
    time series and its trends, given the most important 
    periods (alpha, beta, gamma respectively) and trends.
    The noise would also be reconstructed accordingly.
'''

f_overall_rough_trend = np.concatenate((np.concatenate((np.concatenate((f_predicted_1_tr, f_predicted_2_tr), axis=0),
                                                        f_predicted_3_tr), axis=0), f_predicted_4_tr), axis=0)
f_overall_rough_period_1 = np.concatenate((np.concatenate((np.concatenate((f_predicted_1_p1, f_predicted_2_p1), axis=0),
                                                           f_predicted_3_p2), axis=0), f_predicted_4_p1), axis=0)
f_overall_rough_period_2 = np.concatenate((np.concatenate((np.concatenate((f_predicted_1_p2, f_predicted_2_p2), axis=0),
                                                           f_predicted_3_p1), axis=0), f_predicted_4_p3), axis=0)
f_overall_rough_period_3 = np.concatenate((np.concatenate((np.concatenate((f_predicted_1_p3, f_predicted_2_p3), axis=0),
                                                           f_predicted_3_p3), axis=0), f_predicted_4_p2), axis=0)
f_overall_rough_noise = np.concatenate((np.concatenate((np.concatenate((f_predicted_1_noise, f_predicted_2_noise), axis=0),
                                                        f_predicted_3_noise), axis=0), f_predicted_4_noise), axis=0)

f_overall_no_noise = np.concatenate((np.concatenate((np.concatenate((f_1_overall_no_noise, f_2_overall_no_noise), axis=0),
                                                      f_3_overall_no_noise), axis=0), f_4_overall_no_noise), axis=0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~ REQUESTING WHETHER TO PLOT TIME SERIES' COMPONENTS

print("\nPlease, specify:\nWhich components of the \"rough\" SSA partially-derived time series you want to be plotted?"
      "\nPlease, input + or -:\n")

true_false = ["Time Series itself: ",
              "Trend component: ",
              "1st periodic component: ",
              "2nd periodic component: ",
              "3rd periodic component: ",
              "Noise component: ",
              "Time Series, noise reduced: "]
for i in range(len(true_false)):
    answer = str(input(true_false[i]))
    if answer == "+":
        true_false[i] = True
    elif answer == "-":
        true_false[i] = False
    else:
        print("\nUnrecognizable input, assuming: negative answer.\n")
        true_false[i] = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~ PLOTTING FULL TIME SERIES AND COMPONENTS

estimated = [(F, r"$\mathcal{F}(x)$", '', true_false[0]),
             (f_overall_rough_trend,    r"$\tilde{F}_{\Sigma}^{(\mathrm{trend})}$", 'orange', true_false[1]),
             (f_overall_rough_period_1, r"$\tilde{F}_{\Sigma}^{(\mathrm{period})_"
                                        r"{(\alpha_1,\alpha_2,\beta_3,\alpha_4)}}$", '', true_false[2]),
             (f_overall_rough_period_2, r"$\tilde{F}_{\Sigma}^{(\mathrm{period})_"
                                        r"{(\beta_1,\beta_2,\alpha_3,\gamma_4)}}$", '', true_false[3]),
             (f_overall_rough_period_3, r"$\tilde{F}_{\Sigma}^{(\mathrm{period})_"
                                        r"{(\gamma_1,\gamma_2,\gamma_3,\beta_4)}}$", '', true_false[4]),
             (f_overall_rough_noise,    r"$\tilde{\eta}_{\Sigma}(x)$", 'black', true_false[5]),
             (f_overall_no_noise,       r"$\tilde{F}_{\Sigma}(x)$", 'red', true_false[6])]

pvp.plot_full_snp500(x, estimated)
