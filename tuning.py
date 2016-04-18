#!/usr/bin/env python

import subprocess
import sys


proc = subprocess.Popen("./verify-adam.o 0 256 256 256 156 1 1", shell=True)
proc.wait()