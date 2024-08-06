import os
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from scipy.interpolate import make_interp_spline


def main():
    # Re = 5.5e-5  # 8.5e-5 # 3e-4  # 7e-5
    # Pe = 0.0623  # 0.0963 # 0.3401  # 0.0793
    # experiment_file = "100water_normalized_Pe0.0636942675.txt"  # "100water_normalized_Pe0.3184713376.txt"
    Re = 3e-4  # 8.5e-5 # 3e-4  # 7e-5
    Pe = 0.3401  # 0.0963 # 0.3401  # 0.0793
    experiment_file = "100water_normalized_Pe0.3184713376.txt"  # "100water_normalized_Pe0.0636942675.txt"
    experiment_offest = -0.1  # 0
    is_negative = True
    posneg_prefix = "" if is_negative else "_positive"
    path = pathlib.Path(__file__).parent.absolute()
    
    prefixs = ["_45degree_g9", "_45degree_g0", "_cylinder_g9", "_InfinitePlane_g9"]
    labels = [r"45$^{\circ}$, g=9.8", r"45$^{\circ}$, g=0", r"0$^{\circ}$, g=9.8", r"90$^{\circ}$, g=9.8"]
    xs = {}
    currents = {}
    for l, prefix in enumerate(prefixs):
        dir_path = f"{path}/{prefix}/records{posneg_prefix}/"
        filenames = os.listdir(dir_path)
        filenames = [filename for filename in filenames if '.txt' in filename]
        
        # Try to find your target Re number
        filename = None
        for fn in filenames:
            if f"Re{Re}" in fn and f"Pe{int(Pe * 10000)}" in fn:
                filename = fn
        
        if filename is None:
            print(f"Case {prefix}, no Re={Re} example was find")
            exit(-1)
            
        fo = open(os.path.join(dir_path, filename), "r")
        fo.readline()
        data = fo.readline()[:-1].split(" ")
        xs[prefix] = []
        currents[prefix] = []
        while data:
            if len(data) == 0 or len(data[0]) == 0:
                break

            xs[prefix].append(float(data[0]))
            currents[prefix].append(float(data[2]))
            data = fo.readline()[:-1].split(" ")

        fo.close()
        
        xs[prefix] = np.array(xs[prefix][:-1]) / 12.5
        currents[prefix] = np.flip(np.asarray(currents[prefix])[:-1])

    # Normalization
    norm_index = 10  # current.shape[-1] // 2
    p0 = 30 if is_negative else 60
    for prefix in xs:
        denominator_i = np.mean(currents[prefix][p0:p0+norm_index])
        currents[prefix] = currents[prefix] / denominator_i

    counter = 0
    plt.title(f"Re={Re}")
    for i, prefix in enumerate(xs):
        x_smooth = xs[prefix]
        y_smooth = currents[prefix]
        line = "--" if Pe == 0.01 else "-"
        plt.plot(x_smooth, y_smooth, line, label=labels[i])
        
        
    # Add experiments
    experiment_path = os.path.join(f"{path}/experiments", experiment_file)
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
    exp_x = np.asarray(exp_x, dtype=np.float32) + experiment_offest
    exp_current = np.asarray(exp_current, dtype=np.float32)
    exp_current = exp_current / exp_current[60]
    line = "--"
    plt.plot(exp_x, exp_current, line, label=f"experiment")
    
    # And Rg theoretical files:
    if is_negative:
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
            plt.plot(Rg_x, [1.0 for i in range(len(Rg_x))], ".", color="black")

    plt.xlabel("D / a")
    plt.xlim(0, 7)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(str(path) + f"/plot_validation.png")


if __name__ == "__main__":
    main()
