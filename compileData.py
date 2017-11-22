import numpy as np
import glob
import sys
import os

read_files = glob.glob("output/results_job_*.txt")

print(read_files)

with open("output/GW170104.dat", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())

#for f in read_files:
#    os.remove(f)
