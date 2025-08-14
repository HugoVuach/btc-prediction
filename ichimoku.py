from ..trend_indicator.base import Indicator
import pandas as pd



class Ichimoku(Indicator):
    def compute(self):
        tenkan = self.params.get("tenkan", 9)
        kijun = self.params.get("kijun", 26)
        senkou = self.params.get("senkou", 52)

        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # Tenkan-sen (Conversion Line)
        self.df['Tenkan_sen'] = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2

        # Kijun-sen (Base Line)
        self.df['Kijun_sen'] = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2

        # Senkou Span A (Leading Span A)
        self.df['Senkou_span_a'] = ((self.df['Tenkan_sen'] + self.df['Kijun_sen']) / 2).shift(kijun)

        # Senkou Span B (Leading Span B)
        self.df['Senkou_span_b'] = ((high.rolling(window=senkou).max() + low.rolling(window=senkou).min()) / 2).shift(kijun)

        # Chikou Span (Lagging Span)
        self.df['Chikou_span'] = close.shift(-kijun)

        # Ligne principale à afficher (par défaut : Kijun)
        self.df['Ichimoku_Main'] = self.df['Kijun_sen']

        return self.df


