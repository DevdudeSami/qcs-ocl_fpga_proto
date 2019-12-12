#!/bin/bash -x

#Run this script to run AOCL emulation
#Run it with two dots like this: . ./burun_emuild_and_run_emu.sh

echo -e "** Did you run the script with 2 dots like this: . ./run_emu.sh**"
echo -e "================================="
echo -e "Executing"
echo -e "================================="
env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./host.exe kernels.aocx
echo -e "================================="
echo -e "out.dat:"
cat out.dat