#!/bin/bash
# source /opt/miriad/miriad_start.sh
# convert input data cube in frequency to velocity

cube_freq=$1
cube_out=$2
cube_vel=$3

fits in=$cube_freq op=xyin out=$cube_out
puthd in=$cube_out/restfreq value=1.420405751768
velsw in=$cube_out axis=vopt
fits in=$cube_out op=xyout out=$cube_vel
rm -r $cube_out


