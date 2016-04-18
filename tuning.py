#!/usr/bin/env python

import subprocess
import sys
for bx in [256, 128]:
    for by in [1, 2]:
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