#!/usr/bin/env python3
import psutil

if __name__ == '__main__':
    pids = []
    for proc in psutil.process_iter(['name', 'pid', 'cmdline']):
        c = ''.join(proc.info['cmdline'])
        conds = [
            proc.info['name'] == 'gzserver',
            proc.info['name'] == 'gzclient',
            'ros' in c,
        ]
        if any(conds):
            print('killed', c)
            proc.kill()
