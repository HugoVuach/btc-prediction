from .base import Indicator

class EMA(Indicator):
    def compute(self):
        period = self.params.get("period", 20)
        ema = self.df['close'].ewm(span=period, adjust=False).mean()
        self.df["EMA"] = ema
        self.df["EMA_Value"] = ema
        return self.df

# === EMA ===
# from indicators.trend_indicator.ema import EMA
# INDICATOR_CLASS = EMA
# INDICATOR_NAME = "EMA_Value"
# INDICATOR_PARAMS = {"period": 20}
# INDICATOR_COLOR = "orange"
# INDICATOR_LINE_LEVELS = []
# INDICATOR_YLIM = None  
