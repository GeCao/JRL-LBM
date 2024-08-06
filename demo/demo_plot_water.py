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
    fluid = "water"
    is_negative = True
    posneg_prefix = "" if is_negative else "_positive"
    path = pathlib.Path(__file__).parent.absolute()
    
    visc = viscs[fluid]
    radius_obs = 12.5e-6
    D = Ds[fluid]
    
    prefix = "_45degree_g9_water" if fluid == "water" else "_ethaline_g9"
    experiment_file = "100water_normalized_Pe0.3184713376.txt"
    experiment_vel = "21 um/s"
    Res = [1e-2, 7e-3, 5e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 7e-5]  # , 5e-5]  # 3e-3
    if not is_negative:
        Res = [1e-2, 7e-3, 5e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 7e-5]
        prefix = "_ppp_g9"
        experiment_file = "100water_normalized_Pe0.0079617834.txt"
        experiment_vel = "0.5 um/s"
    if fluid == "ethaline":
        Res = [1e-4] # [1e-5, 7e-6, 3e-6, 1e-6]  # [1e-4, 7e-5, 3e-5, 1e-5, 7e-6, 3e-6, 1e-6]
        prefix = "_ethaline_g9"
        experiment_file = "0water_normalized_Pe11.3636363636.txt"
        experiment_vel = "30 um/s"
    elif fluid == "water40":
        Res = [1e-4]
        prefix = "_water40_g9"
        experiment_file = "0water_normalized_Pe11.3636363636.txt"
        experiment_vel = "30 um/s"
    dir_path = f"{path}/{prefix}/records{posneg_prefix}/"
    filenames = os.listdir(dir_path)
    filenames = [filename for filename in filenames if '.txt' in filename]
    filenames = sorted(filenames)
    Pes = []
    vel_obss = []
    xs = [[] for i in range(len(Res))]
    currents = [[] for i in range(len(Res))]
    for l, Re in enumerate(Res):
        vel_obs = Re * visc / radius_obs
        vel_obss.append(vel_obs)
        Pe = vel_obs * radius_obs / D
        Pes.append(Pe)
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
    p0 = 35 if is_negative else 40
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

    counter = 0
    plt.title(f"fluid = {fluid}")
    for l, x in enumerate(xs):
        vel_obs = vel_obss[l]
        Pe = Pes[l]
        Re = Res[l]
        line = "--" if Pe == 0.01 else "-"
        plt.plot(x, currents[l], line, label="vel={:.2f}um/s".format(vel_obs * 1e6), linewidth=2)
        plt.scatter(x, currents[l], s=5)
        
        
    # Add experiments
    experiment_path = os.path.join(f"{path}/experiments{posneg_prefix}", experiment_file)
    fo = open(experiment_path, "r")
    fo.readline()
    data = fo.readline()[:-1].split("\t")
    exp_x = []
    exp_current = []
    while data:
        if len(data) == 0 or len(data[0]) == 0:
            break

        exp_x.append(float(data[0]))
        exp_current.append(float(data[1]))
        data = fo.readline()[:-1].split("\t")

    fo.close()
    exp_x = np.asarray(exp_x, dtype=np.float32) + 0.
    exp_current = np.asarray(exp_current, dtype=np.float32)
    if is_negative:
        exp_current = exp_current / exp_current[60]
    else:
        exp_current = exp_current / exp_current[60]
        # t0 = exp_current[0]  # lower
        # t1 = exp_current[-1]  # higher
        # exp_current = (exp_current - t0) / (t1 - t0) * t1 / exp_current[0] + 1
    line = "--"
    plt.plot(exp_x, exp_current, line, label=f"experiment ({experiment_vel})")
    
    # And Rg theoretical files:
    Rgs = [5]
    for i, Rg in enumerate(Rgs):
        theoretical_dir = "NF" if is_negative else "PF"
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
        if is_negative:
            plt.plot(Rg_x, [1.0 for i in range(len(Rg_x))], ".", color="black")

    plt.xlabel("D / a")
    if is_negative:
        plt.xlim(0, 6)
        plt.ylim(0, 2)
    else:
        plt.xlim(0, 9)
        plt.ylim(0.9, 2.5)
    plt.legend(loc="upper right")
    plt.savefig(str(path) + f"/plot_{fluid}{posneg_prefix}.png")


if __name__ == "__main__":
    main()
