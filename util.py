

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
    import os
    import shutil
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif clean:
        shutil.rmtree(directory)
        os.makedirs(directory)


def setup_log(log_path):
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_path,
                        filemode='w')
    return logging.getLogger("main")


def get_stats_file(path):
    file = open(path, "w+")
    file.write(f"episode,avg_reward,min_reward,max_reward,epsilon\n")
    return file


def get_run_id():
    import time
    return "%s" % (time.strftime("%Y-%m-%d-%H-%M-%S"))


