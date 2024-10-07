import os
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from scipy.interpolate import make_interp_spline


mus = {"water": 0.89e-3, "water40": 6.06e-3, "ethaline": 45.23e-3}
rhos = {"water": 1e3, "water40": 1.07819e3, "ethaline": 1.11614e3}
Ds = {"water": 7.85e-10, "water40": 1.53e-10, "ethaline": 0.22e-10}
viscs = {"water": 0.89e-6, "water40": mus["water40"] / rhos["water40"], "ethaline": mus["ethaline"] / rhos["ethaline"]}


def main():
    path = pathlib.Path(__file__).parent.absolute()
    
    radius_obs = 12.5e-6
    
    filenames = [
        "_45degree_g9_water/records_positive/record_res64_Re7e-05_Pe793.txt",
        "_water40_g9/records/record_res128_Re7e-05_Pe25714.txt",
    ]
    labels = ["water + pos + low vel", r"$60\%$ET + neg + high vel"]
    Res = [7e-5, 7e-5]
    Pes = [0.0793, 2.5714]
    fluids = ["water", "water40"]
    print()
    vel_obss = []
    xs = [[] for i in range(len(Res))]
    currents = [[] for i in range(len(Res))]
    for l, Re in enumerate(Res):
        fluid = fluids[l]
        visc = viscs[fluid]
        D = Ds[fluid]
        vel_obs = Re * visc / radius_obs
        vel_obss.append(vel_obs)
        Pe = vel_obs * radius_obs / D
        filename = filenames[l]
        print(Pe)
            
        fo = open(os.path.join(path, filename), "r")
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
    p0 = 50
    print(xs[0][p0])
    for l in range(len(xs)):
        denominator_i = np.mean(currents[l][p0:p0+norm_index])
        currents[l] = currents[l] / denominator_i

    counter = 0
    plt.title(f"fluid = {fluid}")
    for l, x in enumerate(xs):
        vel_obs = vel_obss[l]
        Pe = Pes[l]
        Re = Res[l]
        line = "-"
        plt.plot(x, currents[l], line, label=labels[l], linewidth=2)
        plt.scatter(x, currents[l], s=5)
        
    
    # And Rg theoretical files:
    Rgs = [5]
    for i, Rg in enumerate(Rgs):
        theoretical_dir = "PF"  # "NF" if is_negative else "PF"
        Rg_file_path = os.path.join(path, f"{theoretical_dir}/Rg{Rg}.txt")
        fo = open(Rg_file_path, "r")
        fo.readline()
        data = fo.readline()[:-1].split(" ")
        Rg_x = []
        Rg_current = []
        while data:
            if len(data) == 0 or len(data[0]) == 0:
                break

            Rg_x.append(float(data[0]))
            Rg_current.append(float(data[1]))
            data = fo.readline()[:-1].split(" ")

        fo.close()
        Rg_x = np.asarray(Rg_x, dtype=np.float32)
        Rg_current = np.asarray(Rg_current, dtype=np.float32)
        Rg_current = Rg_current / Rg_current[60]
        line = "-."
        plt.plot(Rg_x, Rg_current, line, label=f"Rg {Rg}")
        # if is_negative:
        #     plt.plot(Rg_x, [1.0 for i in range(len(Rg_x))], ".", color="black")

    plt.xlabel("D / a")
    plt.xlim(0, 4.5)
    # plt.ylim(0, 2)
    plt.legend(loc="upper right")
    plt.savefig(str(path) + f"/plot_similarity.png")


if __name__ == "__main__":
    main()
