#!/usr/bin/env python

import subprocess
import sys

test = input("Input the size of the stencil grid:")
command = "./verify-adam.o "
command += "0 "
command += test + " " + test + " " + test + " "
command += "1 1"
proc = subprocess.Popen(command, shell=True)
proc.wait()