#!/usr/bin/env python
import argparse
import pathlib
import re

import numpy as np
from colorama import Fore

from moonshine.filepath_tools import load_hjson


def nested_iter(d):
    for key, value in d.items():
        yield key, value
        if isinstance(value, dict):
            yield from nested_iter(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=pathlib.Path)
    args = parser.parse_args()

    log = load_hjson(args.dir / 'logfile.hjson')
    total_dt = 0
    dt_by_key = {}
    dts_by_iter = {}
    for k, v in log.items():
        if re.fullmatch(r'iter\d+', k):
            iter_total_dt = 0
            for dt_k, dt_str in nested_iter(v):
                if 'dt' in dt_k:
                    dt = float(dt_str)
                    if dt_k not in dt_by_key:
                        dt_by_key[dt_k] = 0
                    dt_by_key[dt_k] += dt
                    iter_total_dt += dt
            dts_by_iter[k] = iter_total_dt
            total_dt += iter_total_dt

    print(Fore.CYAN + "By Iteration" + Fore.RESET)
    print("         individual  cumulative")
    cumulative_iter_dt_hr = 0
    for k, iter_total_dt in dts_by_iter.items():
        iter_total_dt_min = iter_total_dt / 60
        cumulative_iter_dt_hr += (iter_total_dt_min / 60)
        print(f"{k:7} {iter_total_dt_min:6.1f} min {cumulative_iter_dt_hr:6.1f} hr")
    print(f"Average: {np.mean(list(dts_by_iter.values())) / 60:5.1f} min")
    print()

    print(Fore.CYAN + "By Type" + Fore.RESET)
    for k, v in dt_by_key.items():
        print(f"{k:25} {v / 3600:6.3f} hr {v / total_dt * 100:3.0f} %")

    print()
    print(Fore.GREEN + f"Total: {(total_dt / 3600):.1f} hr" + Fore.RESET)


if __name__ == '__main__':
    main()
