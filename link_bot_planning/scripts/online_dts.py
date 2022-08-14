#!/usr/bin/env python
import argparse
import pathlib
import re

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
    print(Fore.CYAN + "By Iteration" + Fore.RESET)
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
            print(f"{k:6} {(iter_total_dt/60):6.1f} min")
            total_dt += iter_total_dt
    print()

    print(Fore.CYAN + "By Type" + Fore.RESET)
    for k, v in dt_by_key.items():
        print(f"{k:25} {v/3600:6.3f} hr {v/total_dt*100:3.0f} %")

    print()
    print(Fore.GREEN + f"Total: {(total_dt/3600):.1f} hr" + Fore.RESET)


if __name__ == '__main__':
    main()
