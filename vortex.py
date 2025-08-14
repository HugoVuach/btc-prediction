from .base import Indicator
import pandas as pd

class VortexIndicator(Indicator):
    def compute(self):
        period = self.params.get("period", 14)

        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # VM+ et VM−
        vm_plus = (high - low.shift()).abs()
        vm_minus = (low - high.shift()).abs()

        # True Range (TR)
        tr1 = (high - low).abs()
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        tr_sum = tr.rolling(window=period).sum()
        vi_plus = vm_plus.rolling(window=period).sum() / tr_sum
        vi_minus = vm_minus.rolling(window=period).sum() / tr_sum

        self.df['VI+'] = vi_plus
        self.df['VI-'] = vi_minus

        # Pour compatibilité générique
        self.df['Vortex'] = vi_plus
        return self.df


# === Vortex ===
# from indicators.trend_indicator.vortex import VortexIndicator
# INDICATOR_CLASS = VortexIndicator
# INDICATOR_NAME = "Vortex"
# INDICATOR_PARAMS = {"period": 14}
# INDICATOR_COLOR = "blue"
# INDICATOR_LINE_LEVELS = [1]
# INDICATOR_YLIM = (0, 3)