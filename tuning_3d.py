#!/usr/bin/env python

import subprocess
import sys
command = "make verify-adam"
proc = subprocess.Popen(command, shell=True)
proc.wait()
nx = input("Input the size of the stencil grid:")
for bx in [4, 8, 16, 32]:
    for by in [4, 8, 16, 32]:
        for bz = in [4, 8, 16, 32]:
            if (bx*by*bz<=1024):
                #nx = input("Input the size of the stencil grid:")
                #bx = input("Input the block size of X:")
                #by = input("Input the block size of Y:")
                command = "./verify-adam.o "
                command += "0 "
                command += str(nx) + " " + str(nx) + " " + str(nx) + " "
                command += str(bx) + " " + str(by) + " "
                command += "1"
                proc = subprocess.Popen(command, shell=True)
                proc.wait()