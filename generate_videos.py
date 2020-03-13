import subprocess
import os
import glob
import threading

def render(x):
    print("rendreing", x)
    subprocess.check_call(["./generate_video.sh", x])

threads = []
experiment_path = "en3/*/"
experiments = glob.glob(experiment_path)
for x in experiments:
    path1 = os.path.join(x, "render")
    x1 = threading.Thread(target=render, args=(path1,))
    x1.start()
    threads.append(x1)

for x in threads:
    x.join()
