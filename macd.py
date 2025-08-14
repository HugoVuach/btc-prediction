from .base import Indicator

class MACD(Indicator):
    def compute(self):
        fast = self.params.get("fast", 12)
        slow = self.params.get("slow", 26)
        signal = self.params.get("signal", 9)

        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()

        self.df['MACD_Line'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD_Line'].ewm(span=signal, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD_Line'] - self.df['MACD_Signal']

        # Pour affichage automatique
        self.df['MACD'] = self.df['MACD_Line']
        return self.df

# === MACD ===
# from indicators.trend_indicator.macd import MACD
# INDICATOR_CLASS = MACD
# INDICATOR_NAME = "MACD"
# INDICATOR_PARAMS = {"fast": 12, "slow": 26, "signal": 9}
# INDICATOR_COLOR = "blue"
# INDICATOR_LINE_LEVELS = [0]
# INDICATOR_YLIM = (-60, 60)  # à adapter selon la volatilité
