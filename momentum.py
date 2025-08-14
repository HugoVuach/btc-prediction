from .base import Indicator

class Momentum(Indicator):
    def compute(self):
        period = self.params.get("period", 10)

        # Momentum = close(t) - close(t - n)
        self.df["Momentum"] = self.df["close"] - self.df["close"].shift(period)

        # Pour intégration dans le graphe dynamique
        self.df["Momentum_Value"] = self.df["Momentum"]
        return self.df


# === Momentum ===
# from indicators.momentum_indicator.momentum import Momentum
# INDICATOR_CLASS = Momentum
# INDICATOR_NAME = "Momentum_Value"
# INDICATOR_PARAMS = {"period": 10}
# INDICATOR_COLOR = "black"
# INDICATOR_LINE_LEVELS = [0]
# INDICATOR_YLIM = (-1000, 1000)  # à ajuster selon volatilité