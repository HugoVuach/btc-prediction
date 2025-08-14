from .base import Indicator
import numpy as np

class UltimateOscillator(Indicator):
    def compute(self):
        short = self.params.get("short", 7)
        medium = self.params.get("medium", 14)
        long = self.params.get("long", 28)
        w1, w2, w3 = 4, 2, 1

        # True low = min(low, previous close)
        prev_close = self.df['close'].shift(1)
        true_low = self.df[['low', 'close']].min(axis=1)

        # Buying pressure
        bp = self.df['close'] - true_low

        # True range
        tr = self.df[['high', 'close']].max(axis=1) - true_low

        avg1 = bp.rolling(window=short).sum() / tr.rolling(window=short).sum()
        avg2 = bp.rolling(window=medium).sum() / tr.rolling(window=medium).sum()
        avg3 = bp.rolling(window=long).sum() / tr.rolling(window=long).sum()

        uo = 100 * (w1 * avg1 + w2 * avg2 + w3 * avg3) / (w1 + w2 + w3)
        self.df["Ultimate"] = uo
        return self.df


# === Ultimate oscillator ===
# from indicators.momentum_indicator.ultimate import UltimateOscillator
# INDICATOR_CLASS = UltimateOscillator
# INDICATOR_NAME = "Ultimate"
# INDICATOR_PARAMS = {"short": 7, "medium": 14, "long": 28}
# INDICATOR_COLOR = "darkgreen"
# INDICATOR_LINE_LEVELS = [70, 30]
# INDICATOR_YLIM = (0, 100)