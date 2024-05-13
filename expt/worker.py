import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

import expt

SLEEP = 1


def main(args):
    hostname = socket.gethostname()
    pid = os.getpid()
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    worker_name = f"{hostname}:{pid}:{gpus}"
    print(f"{worker_name=}")

    queue_dir = f"{expt.DFS_ROOT}/queues/{args.project}"
    print(f"{queue_dir=}")

    while True:
        iter_ = Path(f"{queue_dir}/ready").iterdir()

        try:
            ready_path = min(iter_)
        except ValueError:
            time.sleep(SLEEP)
            continue

        task_name = ready_path.name

        Path(f"{queue_dir}/active").mkdir(exist_ok=True)
        try:
            active_path = ready_path.rename(f"{queue_dir}/active/{task_name}")
        except FileNotFoundError:
            print(f"failed to acquire task {task_name}")
            time.sleep(SLEEP)
            continue

        print(f"acquired task {task_name}")
        print(f"tail -f {active_path}")

        with open(active_path, "r") as f:
            task_str = f.readline().strip()
            print(f" + {task_str}")

        try:
            with open(active_path, "a", buffering=1) as f:
                f.write(f"{worker_name}\n")
                subprocess.run(task_str, shell=True, stdout=f, stderr=f, check=True)

        except subprocess.CalledProcessError:
            Path(f"{queue_dir}/failed").mkdir(exist_ok=True)
            failed_path = active_path.rename(f"{queue_dir}/failed/{task_name}")
            print(f"failed task {task_name}")

        except KeyboardInterrupt:
            Path(f"{queue_dir}/failed").mkdir(exist_ok=True)
            failed_path = active_path.rename(f"{queue_dir}/failed/{task_name}")
            print(f"failed task {task_name}")
            break

        else:
            Path(f"{queue_dir}/done").mkdir(exist_ok=True)
            done_path = active_path.rename(f"{queue_dir}/done/{task_name}")
            print(f"done task {task_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="relbench/2024-05-13_dev")
    args = parser.parse_args()
    main(args)
