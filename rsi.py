from .base import Indicator

class RSI(Indicator):
    def compute(self):
        period = self.params.get('period', 14)
        delta = self.df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        return self.df


# === RSI ===
# from indicators.momentum_indicator.rsi import RSI
# INDICATOR_CLASS = RSI
# INDICATOR_NAME = "RSI"
# INDICATOR_PARAMS = {"period": 14}
# INDICATOR_COLOR = "purple"
# INDICATOR_LINE_LEVELS = [70, 30]
# INDICATOR_YLIM = (0, 100)