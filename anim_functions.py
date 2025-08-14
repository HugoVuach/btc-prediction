import matplotlib.dates as mdates
import pytz
import matplotlib.pyplot as plt

from globals import candles, INDICATOR_CLASS, INDICATOR_PARAMS, INDICATOR_COLOR, INDICATOR_NAME, INDICATOR_YLIM, INDICATOR_LINE_LEVELS



def animate_3_plots(i,ax1,ax2,ax3):
    if candles.empty:
        return

    ax1.clear()
    ax2.clear()
    ax1.set_title("BTC/USDT – 300 dernières bougies 1m (Heure de Paris)", fontsize=14)
    ax1.set_ylabel("Prix (USDT)")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Heure (Europe/Paris)")

    df = candles.copy()

    # === Appliquer l'indicateur dynamique ===
    indicator = INDICATOR_CLASS(df, **INDICATOR_PARAMS)
    df = indicator.compute()


    # === Graphique principal : chandeliers + prix moyen pondéré ===
    for idx in range(len(df)):
        row = df.iloc[idx]
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax1.vlines(row['timestamp_local'], row['low'], row['high'], color=color, linewidth=1, zorder=1)
        ax1.vlines(row['timestamp_local'], row['open'], row['close'], color=color, linewidth=4, zorder=2)

    # === Prix moyen pondéré
    df['weighted_close'] = df['close'] * df['volume']
    wc_sum = df['weighted_close'].rolling(window=20).sum()
    vol_sum = df['volume'].rolling(window=20).sum()
    df['volume_avg_price'] = wc_sum / vol_sum
    ax1.plot(df['timestamp_local'], df['volume_avg_price'], label='Prix moyen pondéré (20 bougies)', color='blue', linewidth=2)
    
    # === Lignes horizontales : plus haut / plus bas ===
    highest = df['high'].max()
    lowest = df['low'].min()
    
    last_time = df['timestamp_local'].iloc[-1]

    ax1.axhline(y=highest, color='green', linestyle='--', linewidth=1.5, label=f'Plus haut: {highest:.2f}')
    ax1.axhline(y=lowest, color='red', linestyle='--', linewidth=1.5, label=f'Plus bas: {lowest:.2f}')


    ax1.text(last_time, highest, f'{highest:.2f}', color='green', va='bottom', ha='right', fontsize=10)
    ax1.text(last_time, lowest, f'{lowest:.2f}', color='red', va='top', ha='right', fontsize=10)
    ax1.legend()


    # === Graphique secondaire : volumes ===
    ax2.bar(df['timestamp_local'], df['volume'], width=0.0005, color='gray')

    # === Indicateur dynamique ===
    ax3.set_ylabel(INDICATOR_NAME)
    # ax3.set_ylim(*INDICATOR_YLIM)
    if INDICATOR_YLIM is not None:
        ax3.set_ylim(*INDICATOR_YLIM)
    else:
        last_val = df[INDICATOR_NAME].dropna().iloc[-1]
        ax3.set_ylim(last_val * 0.99, last_val * 1.01)

    for level in INDICATOR_LINE_LEVELS:
        color = 'red' if level > 50 else 'green'
        ax3.axhline(level, color=color, linestyle='--', linewidth=1)
    ax3.plot(df['timestamp_local'], df[INDICATOR_NAME], color=INDICATOR_COLOR)
    ax3.legend([f"{INDICATOR_NAME} ({', '.join(str(v) for v in INDICATOR_PARAMS.values())})"])

    # Format de l'heure sur l'axe X
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Europe/Paris')))
    ax3.tick_params(axis='x', rotation=45)

    # Format de l'heure
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Europe/Paris')))
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()


def animate_2_plots(i,ax1,ax2):
    if candles.empty:
        return
    
    ax1.clear()
    ax2.clear()
    ax1.set_title("BTC/USDT – 300 dernières bougies 1m (Heure de Paris)", fontsize=14)
    ax1.set_ylabel("Prix (USDT)")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Heure (Europe/Paris)")

    df = candles.copy()

    # === Appliquer l'indicateur dynamique ===
    indicator = INDICATOR_CLASS(df, **INDICATOR_PARAMS)
    df = indicator.compute() 


    # === Graphique principal : chandeliers + prix moyen pondéré ===
    for idx in range(len(df)):
        row = df.iloc[idx]
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax1.vlines(row['timestamp_local'], row['low'], row['high'], color=color, linewidth=1, zorder=1)
        ax1.vlines(row['timestamp_local'], row['open'], row['close'], color=color, linewidth=4, zorder=2)


    # Prix moyen pondéré
    df['weighted_close'] = df['close'] * df['volume']
    wc_sum = df['weighted_close'].rolling(window=20).sum()
    vol_sum = df['volume'].rolling(window=20).sum()
    df['volume_avg_price'] = wc_sum / vol_sum
    ax1.plot(df['timestamp_local'], df['volume_avg_price'], label='Prix moyen pondéré (20 bougies)', color='blue', linewidth=2)

    # === Lignes horizontales : plus haut / plus bas ===
    highest = df['high'].max()
    lowest = df['low'].min()

    last_time = df['timestamp_local'].iloc[-1]

    ax1.axhline(y=highest, color='green', linestyle='--', linewidth=1.5, label=f'Plus haut: {highest:.2f}')
    ax1.axhline(y=lowest, color='red', linestyle='--', linewidth=1.5, label=f'Plus bas: {lowest:.2f}')
    
    ax1.text(last_time, highest, f'{highest:.2f}', color='green', va='bottom', ha='right', fontsize=10)
    ax1.text(last_time, lowest, f'{lowest:.2f}', color='red', va='top', ha='right', fontsize=10)
    ax1.legend()

    # Indicateur superposé
    ax1.plot(df['timestamp_local'], df[INDICATOR_NAME], color=INDICATOR_COLOR, label=INDICATOR_NAME)

    # YLim adapté aux prix
    low = df['low'].min()
    high = df['high'].max()
    margin = (high - low) * 0.1
    ax1.set_ylim(low - margin, high + margin)


    # Volume
    ax2.bar(df['timestamp_local'], df['volume'], width=0.0005, color='gray')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Europe/Paris')))
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()


def animate_ichimoku(i,ax1,ax2):
    if candles.empty:
        return
    

    df = candles.copy()
    indicator = INDICATOR_CLASS(df, **INDICATOR_PARAMS)
    df = indicator.compute()
  
    ax1.clear()
    ax2.clear()

    ax1.set_title("BTC/USDT – 300 dernières bougies 1m (Heure de Paris)", fontsize=14)
    ax1.set_ylabel("Prix (USDT)")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Heure (Europe/Paris)")

    # === Chandeliers ===
    for idx in range(len(df)):
        row = df.iloc[idx]
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax1.vlines(row['timestamp_local'], row['low'], row['high'], color=color, linewidth=1, zorder=1)
        ax1.vlines(row['timestamp_local'], row['open'], row['close'], color=color, linewidth=4, zorder=2)

    # === Prix moyen pondéré ===
    df['weighted_close'] = df['close'] * df['volume']
    wc_sum = df['weighted_close'].rolling(window=20).sum()
    vol_sum = df['volume'].rolling(window=20).sum()
    df['volume_avg_price'] = wc_sum / vol_sum
    ax1.plot(df['timestamp_local'], df['volume_avg_price'], label='Prix moyen pondéré (20 bougies)', color='blue', linewidth=2)

    # === Lignes Ichimoku ===
    ax1.plot(df['timestamp_local'], df['Tenkan_sen'], label='Tenkan-sen (bleu)', color='blue')
    ax1.plot(df['timestamp_local'], df['Kijun_sen'], label='Kijun-sen (marron)', color='brown')
    ax1.plot(df['timestamp_local'], df['Senkou_span_a'], label='Senkou Span A (orange)', color='orange')
    ax1.plot(df['timestamp_local'], df['Senkou_span_b'], label='Senkou Span B (rouge)', color='red')
    ax1.plot(df['timestamp_local'], df['Chikou_span'], label='Chikou Span (vert)', color='green')

    # === Nuage Ichimoku ===
    ax1.fill_between(df['timestamp_local'], df['Senkou_span_a'], df['Senkou_span_b'],
                     where=df['Senkou_span_a'] >= df['Senkou_span_b'], color='lightgreen', alpha=0.4)
    ax1.fill_between(df['timestamp_local'], df['Senkou_span_a'], df['Senkou_span_b'],
                     where=df['Senkou_span_a'] < df['Senkou_span_b'], color='lightcoral', alpha=0.4)

    df_ts_set = set(df['timestamp_local'].dt.floor('min'))

 

    # === Lignes horizontales haut/bas ===
    highest = df['high'].max()
    lowest = df['low'].min()
    last_time = df['timestamp_local'].iloc[-1]
    ax1.axhline(y=highest, color='green', linestyle='--', linewidth=1.5, label=f'Plus haut: {highest:.2f}')
    ax1.axhline(y=lowest, color='red', linestyle='--', linewidth=1.5, label=f'Plus bas: {lowest:.2f}')
    ax1.text(last_time, highest, f'{highest:.2f}', color='green', va='bottom', ha='right', fontsize=10)
    ax1.text(last_time, lowest, f'{lowest:.2f}', color='red', va='top', ha='right', fontsize=10)

    # === Limites de l'axe Y ===
    margin = (highest - lowest) * 0.1
    ax1.set_ylim(lowest - margin, highest + margin)

    ax1.legend()

    # === Volume ===
    ax2.bar(df['timestamp_local'], df['volume'], width=0.0005, color='gray')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Europe/Paris')))
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()