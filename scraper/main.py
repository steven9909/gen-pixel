import argparse
from tasks import GetTask, DownloadTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task", required=True)

    tasks = [GetTask(subparsers), DownloadTask(subparsers)]

    args = parser.parse_args()
    args.func(args)
