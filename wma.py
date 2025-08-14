from .base import Indicator
import numpy as np

class WMA(Indicator):
    def compute(self):
        period = self.params.get("period", 20)
        weights = np.arange(1, period + 1)

        def weighted_ma(series):
            return np.dot(series, weights) / weights.sum()

        wma = self.df['close'].rolling(window=period).apply(weighted_ma, raw=True)
        self.df["WMA"] = wma
        self.df["WMA_Value"] = wma
        return self.df

# === WMA ===
# from indicators.trend_indicator.wma import WMA
# INDICATOR_CLASS = WMA
# INDICATOR_NAME = "WMA_Value"
# INDICATOR_PARAMS = {"period": 20}
# INDICATOR_COLOR = "orange"
# INDICATOR_LINE_LEVELS = []
# INDICATOR_YLIM = None  