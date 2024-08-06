import os, sys
import math
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../")

mus = {"water": 0.89e-3, "water40": 6.06e-3, "ethaline": 45.23e-3}
rhos = {"water": 1e3, "water40": 1.07819e3, "ethaline": 1.11614e3}
Ds = {"water": 7.85e-10, "water40": 1.53e-10, "ethaline": 0.22e-10}
viscs = {"water": 0.89e-6, "water40": mus["water40"] / rhos["water40"], "ethaline": mus["ethaline"] / rhos["ethaline"]}
prefixs = {"water": "_45degree", "water40": "_ethaline", "ethaline": "_ethaline"}

# 4759 for log, 7052 for others

def main(fluid: str, is_negative: bool):
    demo_path = pathlib.Path(__file__).parent.absolute()
    negpos_prefix = "is_negative" if is_negative else "no-is_negative"
    visc = viscs[fluid]
    D = Ds[fluid]
    prefix = prefixs[fluid]
    radius_obs = 12.5e-6
    if fluid == "water":
        # 712 um/s - 3.56 um/s
        Res = [1e-2, 7e-3, 5e-3, 3e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 7e-5]  # , 5e-5]
        if not is_negative:
            # 35.6 - 4.98 um/s
            Res = [3e-3, 1e-3, 7e-4]  # [1e-2, 7e-3, 5e-3]  # [5e-4, 3e-4, 1e-4]  # , 7e-5]
    elif fluid == "ethaline":
        # 32,418 um/s - 162 um/s
        # Res = [1e-2, 7e-3, 5e-3, 3e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 7e-5, 5e-5]
        # 324 um/s - 2.27 um/s
        Res = [1e-4, 7e-5, 3e-5, 1e-5, 7e-6, 3e-6, 1e-6]
        Res = [1e-3]
    elif fluid == "water40":
        Res = [1e-4]
    else:
        raise NotImplementedError(f"Not implemented for fluid {fluid}")

    for Re in Res:
        vel_obs = Re * visc / radius_obs
        Pe = vel_obs * radius_obs / D
        print(Pe)
    
        # 1. We choose 45 degree (log), with gravity, this is the best case.
        os.system(
            f"python demo_2d_LBM_JRL_fluid{prefix}.py --Re {Re} --vel_obs_real {vel_obs} --gravity_strength_real {-9.8}"
        )
        os.system(
            f"python demo_2d_LBM_JRL_C{prefix}.py --Re {Re} --Pe {Pe} --vel_obs_real {vel_obs} --gravity_strength_real {-9.8} --{negpos_prefix}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )

    parser.add_argument(
        "--fluid", type=str, default="water", help=("water or ethaline")
    )

    parser.add_argument('--is_negative', dest='is_negative', action='store_true')
    parser.add_argument('--no-is_negative', dest='is_negative', action='store_false')
    parser.set_defaults(is_negative=True)

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
    # main_water()
