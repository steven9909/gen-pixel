import json
import requests
import time
import os
from tqdm import tqdm


class Downloader:
    def __init__(self, json_path, save_dir, retry_count=3):
        self.json_path = json_path
        self.save_dir = save_dir
        self.retry_count = retry_count
        os.makedirs(save_dir, exist_ok=True)

    def download_images(self):
        association_dict = {}
        with open(self.json_path, "r") as f:
            data = json.load(f)
            for i, (name, url) in enumerate(tqdm(data.items())):
                try:
                    response = requests.get(url)
                except requests.exceptions.MissingSchema:
                    print(f"Invalid URL: {url}, skipping...")
                    continue

                if response.status_code != 200:
                    for _ in range(self.retry_count):
                        time.sleep(0.5)
                        response = requests.get(url)
                        if response.status_code == 200:
                            break

                if response.status_code != 200:
                    print(f"Failed to download: {name}, skipping...")
                    continue

                with open(os.path.join(self.save_dir, f"{i}.jpg"), "wb") as f:
                    f.write(response.content)
                    association_dict[name] = f"{i}.jpg"

        with open(os.path.join(self.save_dir, "association.json"), "w") as f:
            json.dump(association_dict, f)
