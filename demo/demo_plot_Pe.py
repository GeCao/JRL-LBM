import os
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from scipy.interpolate import make_interp_spline


mus = {"water": 0.89e-3, "ethaline": 45.23e-3}
rhos = {"water": 1e3, "ethaline": 1.11614e3}
Ds = {"water": 7.85e-10, "ethaline": 0.22e-10}
viscs = {"water": 0.89e-6, "ethaline": mus["ethaline"] / rhos["ethaline"]}


def main():
    fluid = "water"
    is_negative = True
    posneg_prefix = "" if is_negative else "_positive"
    path = pathlib.Path(__file__).parent.absolute()
    
    visc = viscs[fluid]
    radius_obs = 12.5e-6
    D = Ds[fluid]
    
    prefix = "_45degree_g9_for_PeTest"
    Res = [1e-2, 7e-3, 5e-3, 1e-3, 7e-4, 5e-4, 3e-4]  # , 1e-4, 7e-5]
    Pes = [1e-3, 3e-3, 7e-3, 1e-2, 3e-2]  # , 7e-2, 1e-1]
    colors = ["red", "orange", "green", "black", "blue", "purple", "gray"]
    n_cases = len(Res) * len(Pes)
    
    dir_path = f"{path}/{prefix}/records{posneg_prefix}/"
    filenames = os.listdir(dir_path)
    filenames = [filename for filename in filenames if '.txt' in filename]
    filenames = sorted(filenames)
    xs = [[] for i in range(n_cases)]
    currents = [[] for i in range(n_cases)]
    for i, Re in enumerate(Res):
        for j, Pe in enumerate(Pes):
            l = i * len(Pes) + j
            
            # Try to find your target Re number
            filename = None
            for fn in filenames:
                if f"Re{Re}" in fn and f"Pe{int(Pe * 10000)}" in fn:
                    filename = fn
            
            if filename is None:
                print(f"Case {prefix}, no Re={Re}, Pe={Pe} example was find")
                exit(-1)
                
            fo = open(os.path.join(dir_path, filename), "r")
            fo.readline()
            data = fo.readline()[:-1].split(" ")
            while data:
                if len(data) == 0 or len(data[0]) == 0:
                    break

                xs[l].append(float(data[0]))
                currents[l].append(float(data[1]))
                data = fo.readline()[:-1].split(" ")

            fo.close()
            
            xs[l] = np.array(xs[l][:-1]) / 12.5
            currents[l] = np.flip(np.asarray(currents[l][:-1]))

    # Normalization
    norm_index = 10  # current.shape[-1] // 2
    p0 = 30 if is_negative else 40
    print(xs[0][p0])
    for l in range(len(xs)):
        if is_negative:
            denominator_i = np.mean(currents[l][p0:p0+norm_index])
            currents[l] = currents[l] / denominator_i
        else:
            denominator_i = np.mean(currents[l][p0:p0+norm_index])
            currents[l] = currents[l] / denominator_i
            # t0 = denominator_i  # lower
            # t1 = currents[l][1]  # higher
            # currents[l] = (currents[l] - t0) / (t1 - t0) * t1 / currents[-1][1] + 1

    plt.title(f"fluid = {fluid}")
    for l, x in enumerate(xs):
        i = l // len(Pes)
        j = l % len(Pes)
        Pe = Pes[j]
        Re = Res[i]
        color = colors[j]
        line = "-"
        if i == 0:
            plt.plot(x, currents[l], line, label="Pe={:.3f}".format(Pe), linewidth=1, color=color)
        else:
            plt.plot(x, currents[l], line, linewidth=1, color=color)
        # plt.scatter(x, currents[l], s=5)

    plt.xlabel("D / a")
    if is_negative:
        plt.xlim(0, 6)
        plt.ylim(0, 1)
    else:
        plt.xlim(0, 9)
        plt.ylim(0.9, 2.5)
    plt.legend(loc="upper right")
    plt.savefig(str(path) + f"/plot_Pe.png")


if __name__ == "__main__":
    main()
