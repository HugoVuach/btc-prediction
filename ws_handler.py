import json
import pandas as pd
import websocket
import threading
from globals import candles


# === WebSocket ===
def on_message(ws, message):
    global candles
    data = json.loads(message)
    k = data['k']
    ts = pd.to_datetime(k['t'], unit='ms', utc=True)
    ts_local = ts.tz_convert('Europe/Paris')

    new_candle = {
        'timestamp': ts,
        'timestamp_local': ts_local,
        'open': float(k['o']),
        'high': float(k['h']),
        'low': float(k['l']),
        'close': float(k['c']),
        'volume': float(k['v']),
    }

    if not candles.empty and candles.iloc[-1]['timestamp'] == ts:
        candles.iloc[-1] = new_candle
    elif ts > candles.iloc[-1]['timestamp']:
        candles = pd.concat([candles, pd.DataFrame([new_candle])], ignore_index=True)

    if len(candles) > 300:
        candles = candles.iloc[-300:]

def on_open(ws):
    print("✅ WebSocket connecté")
    ws.send(json.dumps({
        "method": "SUBSCRIBE",
        "params": ["btcusdt@kline_1m"],
        "id": 1
    }))

def start_websocket():
    # === Thread WebSocket ===
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
        on_message=on_message,
        on_open=on_open
    )
    thread = threading.Thread(target=ws.run_forever)
    thread.daemon = True
    thread.start()
