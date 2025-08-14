from .base import Indicator

class BollingerBands(Indicator):
    def compute(self):
        period = self.params.get("period", 20)
        std_mult = self.params.get("std_mult", 2)

        sma = self.df['close'].rolling(window=period).mean()
        std = self.df['close'].rolling(window=period).std()

        upper_band = sma + std_mult * std
        lower_band = sma - std_mult * std

        self.df['Bollinger_Mid'] = sma
        self.df['Bollinger_Upper'] = upper_band
        self.df['Bollinger_Lower'] = lower_band

        # Pour compatibilité graphique
        self.df['Bollinger'] = sma
        return self.df


# === Bollinger ===
# from indicators.volatility_indicator.bollinger import BollingerBands
# INDICATOR_CLASS = BollingerBands
# INDICATOR_NAME = "Bollinger"
# INDICATOR_PARAMS = {"period": 20, "std_mult": 2}
# INDICATOR_COLOR = "blue"
# INDICATOR_LINE_LEVELS = []
# INDICATOR_YLIM = None