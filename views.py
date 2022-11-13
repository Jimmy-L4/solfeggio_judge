import os
import subprocess

# subprocess.Popen('python dtwzmt.py', shell=True)

import shlex

import sys
import time

# for i in range(5):
#     sys.stdout.write('Processing {}\n'.format(i))
#     time.sleep(1)
# for i in range(5):
#     sys.stderr.write('Error {}\n'.format(i))
#     time.sleep(1)

if __name__ == '__main__':
    shell_cmd = 'python dtwzmt.py'
    os.popen(shell_cmd)
    cmd = shlex.split(shell_cmd)

    time.sleep(30)
    # while p.poll() is None:
    #     line = p.stdout.readline()
    #     line = line.strip()
    #     if line:
    #         print('Subprogram output: [{}]'.format(line))
    # if p.returncode == 0:
    #     print('Subprogram success')
    # else:
    #     print('Subprogram failed')
