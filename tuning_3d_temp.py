#!/usr/bin/env python

import subprocess
import sys
command = "make verify-shmem-3d-temp"
proc = subprocess.Popen(command, shell=True)
proc.wait()
nx = input("Input the size of the stencil grid:")
for bx in [4, 8, 16, 32]:
    for by in [4, 8, 16, 32]:
        for bz in [4, 8, 16, 32]:
            if (bx*by*bz<=1024):
                command = "./verify-shmem-3d-temp.o "
                command += str(nx) + " " + str(nx) + " " + str(nx) + " "
                command += str(bx) + " " + str(by) + " " + str(bz) + " "
                command += "2"
                proc = subprocess.Popen(command, shell=True)
                proc.wait()