from .base import Indicator

class StochasticOscillator(Indicator):
    def compute(self):
        k_period = self.params.get("k_period", 14)
        d_period = self.params.get("d_period", 3)

        low_min = self.df['low'].rolling(window=k_period).min()
        high_max = self.df['high'].rolling(window=k_period).max()

        self.df["%K"] = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        self.df["%D"] = self.df["%K"].rolling(window=d_period).mean()

        # Pour affichage dynamique sur le graphique
        self.df["Stochastic"] = self.df["%K"]
        return self.df


# === Stochastic Oscillator ===
# from indicators.momentum_indicator.stochastic import StochasticOscillator
# INDICATOR_CLASS = StochasticOscillator
# INDICATOR_NAME = "Stochastic"
# INDICATOR_PARAMS = {"k_period": 14, "d_period": 3}
# INDICATOR_COLOR = "blue"
# INDICATOR_LINE_LEVELS = [80, 20]
# INDICATOR_YLIM = (0, 100)