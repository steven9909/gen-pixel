from scrapers import get_scraper
from download import Downloader


class BaseTask:
    def __init__(self, subparsers):
        self.parser = subparsers.add_parser(self.task_name, help=self.task_help)
        self.add_arguments()
        self.parser.set_defaults(func=self.run)

    def add_arguments(self):
        raise NotImplementedError

    def run(self, args):
        return {}


class GetTask(BaseTask):
    task_name = "get"
    task_help = "Getting the JSON data"

    def add_arguments(self):
        self.parser.add_argument(
            "--save_dir", type=str, help="Directory to save the JSON"
        )
        self.parser.add_argument("--scraper", type=str, help="Scraper to use")

    def run(self, args):
        scraper = get_scraper(args)
        scraper.save(args.save_dir)


class DownloadTask(BaseTask):
    task_name = "download"
    task_help = "Downloading the data"

    def add_arguments(self):
        self.parser.add_argument(
            "--save_dir", type=str, help="Directory to save the data"
        )
        self.parser.add_argument(
            "--json_dir", type=str, help="Directory to the json file"
        )
        self.parser.add_argument(
            "--thread_count", type=int, default=1, help="Number of threads to use"
        )

    def run(self, args):
        downloader = Downloader(args.json_dir, args.save_dir)
        downloader.download_images()
