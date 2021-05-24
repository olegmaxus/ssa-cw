import numpy as np
import matplotlib.pyplot as plt


def plot_contribution(spectrum):
    #spectrum_copy = np.insert(spectrum, 0, 0, axis=0)
    sum_spec = spectrum.sum()
    plt.plot(spectrum / sum_spec * 100, 'red', label='$\mathcal{R}(i)$', lw=2.5)
    plt.plot(spectrum.cumsum() / sum_spec * 100, 'firebrick', label='$\mathcal{C}(i)$', lw=2.5)
    plt.style.use('seaborn-talk')
    plt.title("Relative [$\mathcal{R}(i)$] and Cumulative [$\mathcal{C}(i)$] contributions [in %] of $\mathcal{W}_i$ to $\mathbf{X}$")
    plt.xlabel("$i$")
    plt.ylabel("Contribution, %")
    plt.xlim(0, (len(spectrum) - 1) // 10 + 2)
    plt.xticks(range(0, (len(spectrum) - 1) // 10 + 2), range(1, (len(spectrum) - 1) // 10 + 3))
    plt.legend()
    plt.show()


def sum_disjoint_eigensubsets(nd_array, split_set, extra_key=np.array([])):

    return


def read_data(string):
    return string

def plot_matrix(m, title=""):
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

### Checks ###


if __name__ == "__main__":
    print('hello')
