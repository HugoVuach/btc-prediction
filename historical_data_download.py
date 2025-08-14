# import yfinance as yf
# import pandas as pd
# import numpy as np

# # Paramètres
# ticker = "BTC-USD"
# start_date = "2014-01-01"  # Date de début des données fiables
# end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# # Téléchargement
# df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

# # Nettoyage
# df = df.sort_index()
# df = df.dropna()

# # Ajout de daily_difference (sur Adj Close)
# df['daily_difference'] = df['Adj Close'].diff()

# # Rendements (returns)
# df['return_1d'] = df['Adj Close'].pct_change()
# df['return_5d'] = df['Adj Close'].pct_change(5)
# df['return_21d'] = df['Adj Close'].pct_change(21)

# # Moyennes mobiles
# df['ma_7'] = df['Adj Close'].rolling(window=7).mean()
# df['ma_21'] = df['Adj Close'].rolling(window=21).mean()

# # Volatilité
# df['std_7'] = df['Adj Close'].rolling(window=7).std()
# df['std_21'] = df['Adj Close'].rolling(window=21).std()

# # Momentum (différences)
# df['momentum_7'] = df['Adj Close'].diff(7)
# df['momentum_21'] = df['Adj Close'].diff(21)

# # RSI (Relative Strength Index)
# delta = df['Adj Close'].diff()
# gain = delta.where(delta > 0, 0)
# loss = -delta.where(delta < 0, 0)
# avg_gain = gain.rolling(14).mean()
# avg_loss = loss.rolling(14).mean()
# rs = avg_gain / avg_loss
# df['rsi_14'] = 100 - (100 / (1 + rs))

# # MACD
# ema_12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
# ema_26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
# df['macd'] = ema_12 - ema_26
# df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# # Nettoyage final
# df = df.dropna()

# # Enregistrement
# filename = "BTC_full_features.csv"
# df.to_csv(filename)
# print(f"✅ Données enregistrées dans : {filename}")
# print(df.head())

# import requests
# import pandas as pd
# import time
# from datetime import datetime, timedelta
# from tqdm import tqdm

# BASE_URL = "https://api.binance.com"
# KLINE_ENDPOINT = "/api/v3/klines"
# SYMBOL = "BTCUSDT"           # Remplacez par la paire souhaitée
# INTERVAL = "1m"
# LIMIT = 1000                 # Max autorisé par appel
# MONTHS = 6

# def get_klines(symbol, interval, start_time, end_time=None, limit=1000):
#     url = BASE_URL + KLINE_ENDPOINT
#     params = {
#         "symbol": symbol,
#         "interval": interval,
#         "startTime": int(start_time.timestamp() * 1000),
#         "limit": limit
#     }
#     if end_time:
#         params["endTime"] = int(end_time.timestamp() * 1000)

#     response = requests.get(url, params=params)
#     if response.status_code != 200:
#         print(f"Error: {response.status_code} - {response.text}")
#         return []

#     return response.json()

# def fetch_ohlc(symbol, interval, months):
#     end_time = datetime.now()
#     start_time = end_time - timedelta(days=30 * months)
#     all_data = []

#     current_time = start_time
#     pbar = tqdm(total=int((end_time - start_time).total_seconds() / 60))

#     while current_time < end_time:
#         klines = get_klines(symbol, interval, current_time)
#         if not klines:
#             break

#         for k in klines:
#             timestamp = datetime.fromtimestamp(k[0] / 1000)
#             if timestamp >= end_time:
#                 break
#             all_data.append([
#                 timestamp, float(k[1]), float(k[2]),
#                 float(k[3]), float(k[4]), float(k[5])
#             ])
#             pbar.update(1)

#         current_time = datetime.fromtimestamp(klines[-1][0] / 1000) + timedelta(minutes=1)
#         time.sleep(0.5)  # éviter le rate limit

#     pbar.close()

#     df = pd.DataFrame(all_data, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
#     df.to_csv(f"{symbol}_{interval}_ohlc.csv", index=False)
#     print(f"✅ Données sauvegardées dans {symbol}_{interval}_ohlc.csv")
#     return df

# if __name__ == "__main__":
#     df = fetch_ohlc(SYMBOL, INTERVAL, MONTHS)

import requests
import os
import json
import time
import csv
import datetime
from time import perf_counter
from termcolor import colored
import click


class Logger(object):
    def __init__(self, tag=None, file_path=None, color="cyan"):
        self.tag = tag
        self.file_path = file_path
        self.color = color
        self.start_time = perf_counter()
        if self.file_path is not None:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            self.fh = open(self.file_path, "w+")

    def __call__(self, *args, **kwargs):
        date_str = colored(f"[{datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')}]", color=self.color)
        took_str = colored(f"[{perf_counter() - self.start_time:.4f}]", color=self.color)
        prefix = f"{date_str} {took_str}"
        if self.tag is not None:
            tag_str = colored(f"[{self.tag}]", color=self.color, attrs=["bold"])
            prefix += f" {tag_str}"
        print(f"{prefix}", *args, **kwargs)
        if self.file_path is not None:
            print(f"{prefix}", *args, **kwargs, file=self.fh)

    def reset_timer(self):
        self.start_time = perf_counter()

    def close(self):
        self.fh.close()


class OHLCDownloader(object):
    def __init__(self, interval, save_dir):
        self.interval = interval
        self.save_dir = save_dir
        checkpoints_dir = f"{self.save_dir}/checkpoints"
        os.makedirs(checkpoints_dir, exist_ok=True)

    def fetch(self, token_address, page_size=100, cursor=None):
        url = "https://api.syve.ai/v1/prices_usd"
        body = {
            "filter": {
                "type": "eq",
                "params": {"field": "token_address", "value": token_address},
            },
            "bucket": {"type": "range", "params": {"field": "timestamp", "interval": self.interval}},
            "aggregate": [
                {"type": "open", "params": {"field": "price_usd_token"}},
                {"type": "max", "params": {"field": "price_usd_token"}},
                {"type": "min", "params": {"field": "price_usd_token"}},
                {"type": "close", "params": {"field": "price_usd_token"}},
            ],
            "options": [
                {"type": "size", "params": {"value": page_size}},
                {"type": "sort", "params": {"field": "timestamp", "value": "desc"}},
            ],
        }
        if cursor is not None:
            body["options"].append({"type": "cursor", "params": {"value": cursor}})
        res = requests.post(url, json=body)
        if res.status_code == 429:
            time.sleep(1)
            return self.fetch(token_address, page_size, cursor)
        data = res.json()
        return data

    def generate(self, token_address, page_size=100, max_pages=None, cursor=None):
        curr_page = 1
        while True:
            data = self.fetch(token_address, page_size, cursor)
            if len(data["results"]) == 0:
                break
            else:
                yield data
            if max_pages is not None and curr_page >= max_pages:
                break
            cursor = data["cursor"]["next"]
            curr_page += 1

    def load_cursor(self, checkpoints_path):
        try:
            with open(checkpoints_path, "r") as file:
                cursor = file.readline().strip()
                if cursor:
                    return cursor
                else:
                    return None
        except FileNotFoundError:
            return None

    def update_cursor(self, cursor, checkpoints_path):
        with open(checkpoints_path, "w+") as file:
            file.write(cursor)

    def round(self, x, n):
        return "%s" % float("%.6g" % x)

    def format_records(self, records):
        output = []
        for x in records:
            output.append(
                {
                    "price_open": self.round(x["price_usd_token_open"], 6) if "price_usd_token_open" in x else None,
                    "price_high": self.round(x["price_usd_token_max"], 6) if "price_usd_token_max" in x else None,
                    "price_low": self.round(x["price_usd_token_min"], 6) if "price_usd_token_min" in x else None,
                    "price_close": self.round(x["price_usd_token_close"], 6) if "price_usd_token_close" in x else None,
                    "timestamp": x["timestamp"],
                    "date": x["date_time"],
                }
            )
        return output

    def save_records(self, records, records_path):
        if len(records) == 0:
            return None
        if not os.path.exists(records_path):
            with open(records_path, "w+") as file:
                headers = ",".join(records[0].keys())
                file.write(headers + "\n")
        with open(records_path, "a+") as file:
            for record in records:
                values = ",".join(str(value) for value in record.values())
                file.write(values + "\n")

    def run(self, token_address, page_size=100, max_pages=None):
        log = Logger(f"{token_address}")
        token_address = token_address.lower()
        checkpoints_path = f"{self.save_dir}/checkpoints/{token_address}.txt"
        records_path = f"{self.save_dir}/{token_address}.csv"
        cursor = self.load_cursor(checkpoints_path)
        for data in self.generate(token_address, page_size, max_pages, cursor):
            records = data["results"]
            cursor = data["cursor"]["next"]
            records = self.format_records(records)
            self.save_records(records, records_path)
            self.update_cursor(cursor, checkpoints_path)
            log(f"Downloaded {len(records)} records. Last cursor: {cursor}.")
            time.sleep(1.01)


class TokenMetadataDownloader(object):
    def __init__(self, token_addresses, token_metadata_path):
        self.token_addresses = token_addresses
        self.token_metadata_path = token_metadata_path

    def batch(self, a: list, sz: int):
        i = 0
        while i < len(a):
            j = i + sz
            yield a[i:j]
            i += sz

    def save_records(self, records, records_path):
        if len(records) == 0:
            return None
        if not os.path.exists(records_path):
            with open(records_path, "w+") as file:
                headers = ",".join(records[0].keys())
                file.write(headers + "\n")
        with open(records_path, "a+") as file:
            for record in records:
                values = ",".join(str(value) for value in record.values())
                file.write(values + "\n")

    def get_visited_addresses(self):
        visited_addresses = set()
        if os.path.exists(self.token_metadata_path):
            with open(self.token_metadata_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                _ = next(reader)  # skip header
                for row in reader:
                    visited_addresses.add(row[0])
        return visited_addresses

    def run(self):
        visited_addresses = self.get_visited_addresses()
        print("Running token metadata downloader... Number of addresses visited:", len(visited_addresses))
        self.token_addresses = [x for x in self.token_addresses if x not in visited_addresses]
        total_saved = len(visited_addresses)
        print("Total to download: ", len(self.token_addresses))
        for b in self.batch(self.token_addresses, 10):
            token_addresses_csv = ",".join(b)
            res = requests.get(f"https://api.syve.ai/v1/metadata/erc20?address={token_addresses_csv}")
            res_data = res.json()
            time.sleep(1.01)
            records = res_data["results"]["results"]
            self.save_records(records, self.token_metadata_path)
            total_saved += len(records)
            msg = f"Finished downloading metadata for {len(records)} tokens"
            msg += f" - Total: {total_saved}/{len(self.token_addresses)}."
            print(msg)


def fetch_token_list():
    token_list = json.load(open(f"{os.environ['REPO_DIR']}/data/token_list.json"))
    return token_list


def download_ohlc(token_list, resolution="1m"):
    downloader = OHLCDownloader(interval=resolution, save_dir=f"{os.environ['REPO_DIR']}/data/ohlc/{resolution}")
    for i, token_address in enumerate(token_list):
        downloader.run(token_address=token_address, page_size=100000, max_pages=None)
        print(f"Finished downloading token {i + 1}/{len(token_list)} (address = {token_address}).")


def download_token_metadata(token_list):
    downloader = TokenMetadataDownloader(token_list, f"{os.environ['REPO_DIR']}/data/token_metadata.csv")
    downloader.run()


@click.command()
@click.option("--metadata", is_flag=True)
@click.option("--ohlc", is_flag=True)
@click.option("--resolution", type=str, default="1m")
def main(metadata, ohlc, resolution):
    if not metadata and not ohlc:
        print("Please specify --metadata or --ohlc.")
        return None
    os.environ["REPO_DIR"] = os.getcwd()
    token_list = fetch_token_list()
    if metadata:
        download_token_metadata(token_list)
    if ohlc:
        download_ohlc(token_list, resolution=resolution)


if __name__ == "__main__":
    main()