from .base import Indicator

class SMA(Indicator):
    def compute(self):
        period = self.params.get("period", 20)
        sma = self.df['close'].rolling(window=period).mean()
        self.df["SMA"] = sma
        self.df["SMA_Value"] = sma
        return self.df

# === SMA ===
# from indicators.trend_indicator.sma import SMA
# INDICATOR_CLASS = SMA
# INDICATOR_NAME = "SMA_Value"
# INDICATOR_PARAMS = {"period": 20}
# INDICATOR_COLOR = "orange"
# INDICATOR_LINE_LEVELS = []
# INDICATOR_YLIM = None  