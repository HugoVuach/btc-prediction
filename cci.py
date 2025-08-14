from .base import Indicator

class CCI(Indicator):
    def compute(self):
        period = self.params.get("period", 20)

        # Prix typique = (high + low + close) / 3
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        self.df['TP'] = tp

        # Moyenne mobile du TP
        tp_sma = tp.rolling(window=period).mean()

        # Écart absolu moyen
        mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean(), raw=True)

        self.df['CCI'] = (tp - tp_sma) / (0.015 * mad)

        # Pour intégration graphique
        self.df['CCI_Value'] = self.df['CCI']
        return self.df


# === CCI (Commodity Channel Index) ===
# from indicators.momentum_indicator.cci import CCI
# INDICATOR_CLASS = CCI
# INDICATOR_NAME = "CCI_Value"
# INDICATOR_PARAMS = {"period": 20}
# INDICATOR_COLOR = "orange"
# INDICATOR_LINE_LEVELS = [100, -100]
# INDICATOR_YLIM = (-200, 200)