# CEESD-Y1_Flame1D

Driver and versioning for a 1D flame problem using MIRGE-Com

#build software with 
./buildMirge.sh

#build a version for a specific platform with 
platforms/\<platform\>/buildMirge.sh

#load that environment
conda activate mirgeDriver.flame1d

#get some work done
run drivers with ./run.sh in their respective subdirectory

Run documentation and analysis located at:
https://docs.google.com/document/d/16munEPXG2ckh-B9MHlvohlY1JtgxGtAYu9L1MbF313w/edit?usp=sharing



main driver in baseline, generally accepted as the current way to run the problem
experiments located in <experiments>, these are variations that may or may not derive from the current driver in baseline, although they may have common ancestery
the driver used to create the timing data is located in timing_runs, this is generally a direct deriviate of the baseline driver, with modified time end conditions
