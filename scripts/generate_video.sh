#!/bin/sh
#
# script to generate a vidoes from a sequence of images in a directory
cd $1
echo $1
ffmpeg -framerate 5 -i %05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
