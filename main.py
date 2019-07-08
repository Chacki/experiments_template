"""
Description: Entry point. Helps starting experiments and defines appropriate
             folder names for checkpoints and logs based on custom flags
"""
from glob import glob
from os import path, makedirs, environ, system
import argparse
from shutil import rmtree

EXPERIMENT_DIR = "experiment"
DIRECTORIES = {
    "--log_dir": "log",
    "--checkpoint_dir": "checkpoint",
    "--evaluation_dir": "evaluation",
    "--tensorboard_dir": "tensorboard",
}
# Following flags are ignored for folder names.
FILTERED_FLAGS = ["train", "eval"]


def flags_to_dict(flags):
    return dict(tuple(flag.split("=")) for flag in flags)


def dict_to_flags(flags):
    return ["=".join(x).rstrip() for x in flags.items()]


def parse_args():
    """ parse arguments
    """
    parser = argparse.ArgumentParser(description="Helps starting experiment.")
    parser.add_argument("experiment", type=str, help="Experiment to start.")
    parser.add_argument(
        "--gpu", type=str, default="", help="Set visible GPU devices."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Start experiment with debugger."
    )
    parser.add_argument(
        "--flagfile", type=str, default=None, help="Path to flagfile"
    )
    args, custom_flags = parser.parse_known_intermixed_args()
    if args.flagfile is not None and path.isfile(args.flagfile):
        custom_flags = flags_to_dict(custom_flags)
        with open(args.flagfile, "r") as file:
            flags = flags_to_dict(file.readlines())
        flags.update(custom_flags)
        custom_flags = dict_to_flags(flags)
    return args, custom_flags


def manage_dirs(dirs):
    """ Create dir or cleaning dir if exist and user confirms

    :dirs: list of directories to create or clean
    """
    for directory in dirs:
        if path.isdir(directory) and input(f"Delete {directory} ?(y/n)") == "y":
            rmtree(directory)
        makedirs(directory, exist_ok=True)


def main():
    """ main entrypoint

    :experiment_path: path of python script to run
    """
    args, custom_flags = parse_args()
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_path = glob(path.join(EXPERIMENT_DIR, args.experiment + "*.py"))[
        0
    ]
    print(f"Run experiment script: {experiment_path}")
    sorted_flags = sorted(
        filter(
            lambda x: x.split("=")[0][2:] not in FILTERED_FLAGS, custom_flags
        )
    )
    name = experiment_path.split("/")[-1].split(".")[0]
    dir_flags = "".join(sorted_flags)
    updated_dirs = {
        k: path.join(v, name + dir_flags) for k, v in DIRECTORIES.items()
    }
    manage_dirs(updated_dirs.values())
    with open(path.join(updated_dirs["--log_dir"], "flags"), "w+") as file:
        file.write("\n".join(sorted_flags))
    flags = dict_to_flags(updated_dirs)
    flags.extend(custom_flags)
    cmd = f"python -m {EXPERIMENT_DIR}.{name} " + " ".join(flags).strip()
    print(f"Execute: {cmd}")
    return system(cmd)


if __name__ == "__main__":
    main()
