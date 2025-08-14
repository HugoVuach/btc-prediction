from .base import Indicator
import pandas as pd

class ATR(Indicator):
    def compute(self):
        period = self.params.get("period", 14)

        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # True Range (TR)
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Moyenne mobile simple ou exponentielle
        atr = tr.rolling(window=period).mean()
        self.df["ATR"] = atr

        # Pour affichage dynamique
        self.df["ATR_Value"] = atr
        return self.df


# === ATR ===
# from indicators.atr import ATR
# INDICATOR_CLASS = ATR
# INDICATOR_NAME = "ATR_Value"
# INDICATOR_PARAMS = {"period": 14}
# INDICATOR_COLOR = "orange"
# INDICATOR_LINE_LEVELS = []
# INDICATOR_YLIM = (0, 60)  # axe automatique