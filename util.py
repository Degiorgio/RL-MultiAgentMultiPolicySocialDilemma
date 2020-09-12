import os
import time
import shutil
import logging


class MyTimer():
    def __init__(self, label, aprint=True):
        self.aprint = aprint
        self.label = label
        self.start = time.time()

    def __enter__(self):
        return self

    def get_runtime(self):
        end = time.time()
        return end - self.start

    def __exit__(self, exc_type, exc_val, exc_tb):
        runtime = self.get_runtime()
        if self.aprint:
            msg = f'{self.label} took {runtime} seconds to complete'
            print(msg)


def __thread_report_log(param):
    try:
        param[0].write(
            f"{param[1]},{param[2]:.10f},{param[3]:.10f},{param[4]:.10f},{param[5]:.10f}\n"
        )
        param[0].flush()
    except:
        print("logging issue")
        pass


def __thread_log(param):
    try:
        param[0].info(param[1])
    except:
        print("logging issue")
        pass


def create_dir(directory, clean=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif clean:
        shutil.rmtree(directory)
        os.makedirs(directory)


def setup_log(log_path):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_path,
                        filemode='w')
    return logging.getLogger("main")


def get_stats_file(path):
    file = open(path, "w")
    file.write("episode,avg_reward,min_reward,max_reward,epsilon\n")
    return file


def get_run_id():
    return "%s" % (time.strftime("%Y-%m-%d-%H-%M-%S"))
