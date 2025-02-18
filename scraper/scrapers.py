from bs4 import BeautifulSoup
import requests
from constants import WHOLEFOODS_URL, LOBLAW_URL
import functools
import json
import re
from selenium import webdriver
import time
import os
from tqdm import tqdm
from selenium.webdriver.chrome.options import Options


class BaseScraper:
    def __init__(self, url, save_dir="./data"):
        self.url = url
        self.save_dir = save_dir

    def get_data(self):
        raise NotImplementedError("You need to implement this method in your subclass")

    def save(self):
        product_dict = self.get_data()
        os.makedirs(f"{self.save_dir}", exist_ok=True)
        with open(f"{self.save_dir}/{self.__class__.__name__}.json", "w") as f:
            json.dump(product_dict, f)

    def filter_name(self, item):
        return (
            self.filter.sub("", item).strip().encode("ascii", "ignore").decode("utf-8")
        )

    @functools.cached_property
    def filter(self):
        return re.compile(r"\([^)]*\)|,.*$")


class LoblawScraper(BaseScraper):
    def __init__(self, save_dir, sleep=5):
        super().__init__(LOBLAW_URL, save_dir)

        self.sleep = sleep

        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        self.driver = webdriver.Chrome(options=chrome_options)

    @functools.cached_property
    def length(self):
        self.driver.get(self.url)
        time.sleep(self.sleep)

        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        max_num = -1

        for tag in soup(["nav"], {"aria-label": "Pagination"}):
            a_tags = tag.find_all("a")
            for tag in a_tags:
                href = tag.get("href")
                if href is None:
                    continue
                num = int(href[href.rfind("=") + 1 :])
                max_num = max(max_num, num)

        return max_num

    def get_data(self):
        product_dict = {}

        cur_page = 1
        with tqdm(total=self.length) as pbar:
            while cur_page <= self.length:
                self.driver.get(self.url + "?page=" + str(cur_page))
                time.sleep(self.sleep)
                soup = BeautifulSoup(self.driver.page_source, "html.parser")

                for tag in soup(["div"], {"class": "css-yyn1h"}):
                    try:
                        img_tag = tag.find("img")
                        if img_tag:
                            img_url = img_tag.get("src")
                            name = tag.find("h3", {"data-testid": "product-title"}).text
                            product_dict[self.filter_name(name)] = img_url
                    except KeyError:
                        print(
                            f"Warning: Missing key in product with name {name} - skipping"
                        )
                        continue

                cur_page += 1
                pbar.update(1)

        return product_dict


class WholeFoodsScraper(BaseScraper):
    def __init__(self, save_dir, limit=60, rate_limit=0.5):
        super().__init__(WHOLEFOODS_URL, save_dir)

        self.limit = limit
        self.rate_limit = rate_limit

    @functools.cached_property
    def length(self):
        count_response = requests.get(
            self.url, params={"leafCategory": "all-products", "limit": 0, "offset": 0}
        )
        json_dict = json.loads(count_response.text)

        return json_dict["facets"][0]["refinements"][0]["count"]

    def get_data(self):
        product_dict = {}

        offset = 0
        with tqdm(total=self.length) as pbar:
            while offset < self.length:
                start_time = time.time()
                response = requests.get(
                    self.url,
                    params={
                        "leafCategory": "all-products",
                        "limit": min(self.limit, self.length - offset),
                        "offset": offset,
                    },
                )
                end_time = time.time()

                json_dict = json.loads(response.text)

                if response.status_code != 200:
                    raise ValueError(
                        f"Failed to get data from {self.url}, returned {response.status_code} with reason {response.reason}"
                    )

                for product in json_dict["results"]:
                    try:
                        product_dict[self.filter_name(product["name"])] = product[
                            "imageThumbnail"
                        ]
                    except KeyError:
                        print(
                            f"Warning: Missing key in product with name {product['name']} - skipping"
                        )
                        continue

                offset += min(self.limit, self.length - offset)

                while end_time - start_time < self.rate_limit:
                    time.sleep(self.rate_limit / 2)
                    end_time = time.time()
                pbar.update(min(self.limit, self.length - offset))

        return product_dict


def get_scraper(name, args):
    if name == "LoblawScraper":
        return LoblawScraper(**vars(args))
    elif name == "WholeFoodsScraper":
        return WholeFoodsScraper(**vars(args))
    else:
        raise ValueError(f"Unknown scraper {name}")
