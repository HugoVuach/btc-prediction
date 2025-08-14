from .base import Indicator

class WilliamsR(Indicator):
    def compute(self):
        period = self.params.get("period", 14)

        high_max = self.df['high'].rolling(window=period).max()
        low_min = self.df['low'].rolling(window=period).min()

        self.df['Williams_%R'] = -100 * (high_max - self.df['close']) / (high_max - low_min)

        # Pour affichage automatique
        self.df['WilliamsR'] = self.df['Williams_%R']
        return self.df

# === Williams %R ===
# from indicators.momentum_indicator.williamsr import WilliamsR
# INDICATOR_CLASS = WilliamsR
# INDICATOR_NAME = "WilliamsR"
# INDICATOR_PARAMS = {"period": 14}
# INDICATOR_COLOR = "darkblue"
# INDICATOR_LINE_LEVELS = [-20, -80]
# INDICATOR_YLIM = (-100, 0)