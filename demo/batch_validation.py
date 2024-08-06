import os, sys
import math
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../")

mus = {"water": 0.89e-3, "ethaline": 45.23e-3}
rhos = {"water": 1e3, "ethaline": 1.11614e3}
Ds = {"water": 7.85e-10, "ethaline": 0.22e-10}
viscs = {"water": 0.89e-6, "ethaline": mus["ethaline"] / rhos["ethaline"]}


def main(fluid: str, is_negative: bool):
    demo_path = pathlib.Path(__file__).parent.absolute()
    negpos_prefix = "is_negative" if is_negative else "no-is_negative"
    Re = 3e-4 # 8.5e-5 # 7e-5  # 3e-4
    visc = viscs[fluid]
    D = Ds[fluid]
    radius_obs = 12.5e-6
    vel_obs = Re * visc / radius_obs
    Pe = vel_obs * radius_obs / D
    
    # 1. test 45 degree (log), with gravity
    os.system(
        f"python demo_2d_LBM_JRL_fluid_45degree.py --Re {Re} --vel_obs_real {vel_obs} --gravity_strength_real {-9.8}"
    )
    os.system(
        f"python demo_2d_LBM_JRL_C_45degree.py --Re {Re} --Pe {Pe} --vel_obs_real {vel_obs} --gravity_strength_real {-9.8} --{negpos_prefix}"
    )
    
    # 2. test 45 degree (log), with no gravity
    os.system(
        f"python demo_2d_LBM_JRL_fluid_45degree.py --Re {Re} --vel_obs_real {vel_obs} --gravity_strength_real {0}"
    )
    os.system(
        f"python demo_2d_LBM_JRL_C_45degree.py --Re {Re} --Pe {Pe} --vel_obs_real {vel_obs} --gravity_strength_real {0} --{negpos_prefix}"
    )
    
    # 3. test 0  degree (),    with gravity
    os.system(
        f"python demo_2d_LBM_JRL_fluid_cylinder.py --Re {Re} --vel_obs_real {vel_obs} --gravity_strength_real {-9.8}"
    )
    os.system(
        f"python demo_2d_LBM_JRL_C_cylinder.py --Re {Re} --Pe {Pe} --vel_obs_real {vel_obs} --gravity_strength_real {-9.8} --{negpos_prefix}"
    )
    
    # 4. test 90 degree (exp), with gravity
    os.system(
        f"python demo_2d_LBM_JRL_fluid_InfinitePlane.py --Re {Re} --vel_obs_real {vel_obs} --gravity_strength_real {-9.8}"
    )
    os.system(
        f"python demo_2d_LBM_JRL_C_InfinitePlane.py --Re {Re} --Pe {Pe} --vel_obs_real {vel_obs} --gravity_strength_real {-9.8} --{negpos_prefix}"
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
