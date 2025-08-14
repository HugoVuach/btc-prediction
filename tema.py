from .base import Indicator

class TEMA(Indicator):
    def compute(self):
        period = self.params.get("period", 20)

        ema1 = self.df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()

        tema = 3 * (ema1 - ema2) + ema3

        self.df["TEMA"] = tema
        self.df["TEMA_Value"] = tema
        return self.df
