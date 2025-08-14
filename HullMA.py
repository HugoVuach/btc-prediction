from .base import Indicator
import numpy as np

class HullMA(Indicator):
    def compute(self):
        period = self.params.get("period", 20)
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))

        wma = lambda p: self.df['close'].rolling(p).apply(
            lambda x: np.dot(np.arange(1, p+1), x) / np.sum(np.arange(1, p+1)),
            raw=True
        )

        wma_half = wma(half_period)
        wma_full = wma(period)

        hma_input = 2 * wma_half - wma_full
        hma = hma_input.rolling(sqrt_period).apply(
            lambda x: np.dot(np.arange(1, sqrt_period+1), x) / np.sum(np.arange(1, sqrt_period+1)),
            raw=True
        )

        self.df["HullMA"] = hma
        self.df["HullMA_Value"] = hma
        return self.df

# === HMA ===
# from indicators.trend_indicator.HullMA import HullMA
# INDICATOR_CLASS = HullMA
# INDICATOR_NAME = "HullMA_Value"
# INDICATOR_PARAMS = {"period": 20}
# INDICATOR_COLOR = "orange"
# INDICATOR_LINE_LEVELS = []
# INDICATOR_YLIM = None  
