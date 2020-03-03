cd $1
ffmpeg -framerate 5 -i %05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
