from .base import Indicator
import pandas as pd

class ADX(Indicator):
    def compute(self):
        period = self.params.get("period", 14)

        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = (high - low).abs()
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        self.df['ADX'] = adx
        self.df['+DI'] = plus_di
        self.df['-DI'] = minus_di

        # Pour compatibilité graphique
        self.df['ADX_Value'] = self.df['ADX']
        return self.df


# === ADX ===
# from indicators.trend_indicator.adx import ADX
# INDICATOR_CLASS = ADX
# INDICATOR_NAME = "ADX_Value"
# INDICATOR_PARAMS = {"period": 14}
# INDICATOR_COLOR = "darkred"
# INDICATOR_LINE_LEVELS = [20, 40]
# INDICATOR_YLIM = (0, 100)