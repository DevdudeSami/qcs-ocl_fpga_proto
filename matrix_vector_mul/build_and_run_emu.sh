#!/bin/bash -x

#Run this script to build and run AOCL emulation
#Run it with two dots like this: . ./build_and_run_emu.sh

echo -e "** Did you run the script with 2 dots like this: . ./build_and_run_emu.sh**"
echo -e "================================="
echo -e "Clean previous build and output files"
echo -e "================================="
rm -f host.exe error.log out.dat matrix_vector_mul.aoco matrix_vector_mul.aocx
echo -e "================================="
echo -e "Building Kernel"
echo -e "================================="
aoc -v --report -march=emulator --board p385_hpc_d5 -DTARGET=AOCL -DDEBUG ./device/matrix_vector_mul.cl
echo -e "================================="
echo -e "Building Host"
echo -e "================================="
make
echo -e "================================="
echo -e "Executing"
echo -e "================================="
env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./host.exe matrix_vector_mul.aocx
echo -e "================================="
echo -e "out.dat:"
cat out.dat