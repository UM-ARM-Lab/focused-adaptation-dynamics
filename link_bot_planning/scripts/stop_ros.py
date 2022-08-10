#!/usr/bin/env python3
import psutil

if __name__ == '__main__':
    pids = []
    for proc in psutil.process_iter(['name', 'pid', 'cmdline']):
        conds = [
            proc.info['name'] == 'gzserver',
            proc.info['name'] == 'gzclient',
            'ros' in proc.info['cmdline'],
        ]
        if any(conds):
            proc.kill()
