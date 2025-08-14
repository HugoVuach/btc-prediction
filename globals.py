import pandas as pd
import requests



# === Importer un indicateur qui se plote pas sur la courbe de prix / INDICATOR_YLIM != None ===
# === for 3 plots configuration ===
# === ADX ===
# from indicators.trend_indicator.adx import ADX
# INDICATOR_CLASS = ADX
# INDICATOR_NAME = "ADX_Value"
# INDICATOR_PARAMS = {"period": 14}
# INDICATOR_COLOR = "darkred"
# INDICATOR_LINE_LEVELS = [20, 40]
# INDICATOR_YLIM = (0, 100)

# pour un indicateur avec INDICATOR_YLIM = None ===
# === pour 2 plots configuration === 
# === SMA ===
from indicators.trend_indicator.sma import SMA
INDICATOR_CLASS = SMA
INDICATOR_NAME = "SMA_Value"
INDICATOR_PARAMS = {"period": 20}
INDICATOR_COLOR = "orange"
INDICATOR_LINE_LEVELS = []
INDICATOR_YLIM = None  

# === Ichimoku ===
# from indicators.ichimoku_indicator.ichimoku import Ichimoku
# INDICATOR_CLASS = Ichimoku
# INDICATOR_NAME = "Ichimoku_Main"
# INDICATOR_PARAMS = {"tenkan": 9, "kijun": 26, "senkou": 52}
# INDICATOR_COLOR = "purple"
# INDICATOR_LINE_LEVELS = []
# INDICATOR_YLIM = None



SYMBOL = "BTCUSDT"
INTERVAL = "1m"


# === Charger les 301 dernières bougies ===
def fetch_initial_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": 301}
    data = requests.get(url, params=params).json()

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        '_', '_', '_', '_', '_', '_'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['timestamp_local'] = df['timestamp'].dt.tz_convert('Europe/Paris')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

candles = fetch_initial_data()
