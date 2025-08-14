from dashboard.A.indicators.momentum_indicator.base import Indicator

class ROC(Indicator):
    def compute(self):
        period = self.params.get("period", 12)

        # Calcul ROC : ((prix actuel - prix n périodes avant) / prix n périodes avant) * 100
        self.df["ROC"] = ((self.df["close"] - self.df["close"].shift(period)) / self.df["close"].shift(period)) * 100

        # Pour affichage dynamique
        self.df["ROC_Value"] = self.df["ROC"]
        return self.df


# === Roc (Rate of Change) ===
# from indicators.momentum_indicator.roc import ROC
# INDICATOR_CLASS = ROC
# INDICATOR_NAME = "ROC_Value"
# INDICATOR_PARAMS = {"period": 12}
# INDICATOR_COLOR = "darkcyan"
# INDICATOR_LINE_LEVELS = [0]
# INDICATOR_YLIM = (-0.6, 0.6)