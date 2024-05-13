import functools
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.auto import tqdm
import wandb


sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=0.75)
plt.rcParams["figure.dpi"] = 157
plt.rcParams["xtick.major.size"] = 0
plt.rcParams["ytick.major.size"] = 0
LINE = 5.5


def save_fig(fig, save_key):
    path = f"fig/{save_key}.pdf"
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    print(f"saved at {path}")


class PathDict(dict):
    def __init__(self, store_dir, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.store_dir = store_dir
        self.device = device

        Path(store_dir).mkdir(parents=True, exist_ok=True)

    def __setitem__(self, key, val):
        path = f"{self.store_dir}/{key}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(val, path)

    def __getitem__(self, key):
        path = f"{self.store_dir}/{key}.pt"
        try:
            val = torch.load(path, map_location=self.device)
        except FileNotFoundError:
            raise KeyError(key)
        return val


DFS_ROOT = "/dfs/scratch1/ranjanr"


def run(main):
    @functools.wraps(main)
    def wrap(args):
        print(f"args={args.__dict__}")

        timestamp = str(time.time_ns())
        print(f"{timestamp=}")

        store_dir = f"{DFS_ROOT}/stores/{args.project}/{timestamp}"
        store = PathDict(store_dir)

        store["info"] = {
            "script_path": str(Path(main.__code__.co_filename).resolve()),
            **args.__dict__,
        }

        wandb.init(project=args.project.replace("/", "_"), name=timestamp, config=args)

        main(store, args)

        wandb.finish()

        store["info"] = {
            **store["info"],
            "done": True,
        }

        diff = {
            k: v
            for k, v in store["info"].items()
            if k not in args.__dict__ or v != args.__dict__[k]
        }
        print(f"info\\args={diff}")

    return wrap


def scan(project):
    project_dir = f"{DFS_ROOT}/stores/{project}"
    out = {}
    for store_dir in tqdm(sorted(Path(project_dir).glob("*"))):
        store = PathDict(store_dir)
        try:
            info = store["info"]
        except KeyError:
            print(f"!rm -r {store_dir}  # no info")
            continue
        if not info.get("done", False):
            print(f"!rm -r {store_dir}  # not done")
            continue
        out[store_dir.name] = info
    out = pd.DataFrame(out).T
    return out


def submit(project, cmd):
    assert "\n" not in cmd

    timestamp = str(time.time_ns())
    submit_path = f"{DFS_ROOT}/queues/{project}/ready/{timestamp}"
    Path(submit_path).parent.mkdir(parents=True, exist_ok=True)

    with open(submit_path, "w") as f:
        f.write(f"{cmd}\n")

    print(f"submitted at {submit_path}")
