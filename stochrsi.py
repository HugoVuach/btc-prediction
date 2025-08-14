from dashboard.A.indicators.momentum_indicator.base import Indicator

class StochasticRSI(Indicator):
    def compute(self):
        period = self.params.get("rsi_period", 14)
        stoch_period = self.params.get("stoch_period", 14)
        smooth_k = self.params.get("smooth_k", 3)
        smooth_d = self.params.get("smooth_d", 3)

        # === Calcul du RSI
        delta = self.df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # === Calcul du Stochastic RSI
        min_rsi = rsi.rolling(window=stoch_period).min()
        max_rsi = rsi.rolling(window=stoch_period).max()

        stoch_rsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
        self.df['StochRSI_%K'] = stoch_rsi.rolling(window=smooth_k).mean()
        self.df['StochRSI_%D'] = self.df['StochRSI_%K'].rolling(window=smooth_d).mean()

        # Pour affichage dynamique : on trace %K
        self.df['StochasticRSI'] = self.df['StochRSI_%K']
        return self.df


# === Stochastic RSI ===
# from indicators.momentum_indicator.stochrsi import StochasticRSI
# INDICATOR_CLASS = StochasticRSI
# INDICATOR_NAME = "StochasticRSI"
# INDICATOR_PARAMS = {
 #   "rsi_period": 14,
 #   "stoch_period": 14,
  #  "smooth_k": 3,
   # "smooth_d": 3
#}
# INDICATOR_COLOR = "blue"
# INDICATOR_LINE_LEVELS = [80, 20]
# INDICATOR_YLIM = (0, 100)