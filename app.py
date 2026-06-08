import html
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# =========================================================
# CONFIGURAZIONE GENERALE
# =========================================================

st.set_page_config(
    page_title="BBAI Trade Republic Dashboard",
    page_icon="📈",
    layout="wide"
)

TICKER_USD = "BBAI"
FX_SYMBOL = "EUR/USD"

SECTOR_TICKERS = {
    "QQQ": "Nasdaq / tecnologia growth",
    "IWM": "Small cap USA",
    "BOTZ": "Robotics & AI",
    "AIQ": "Artificial Intelligence",
    "ITA": "Aerospace & Defense",
    "VIXY": "Volatilità mercato"
}


# =========================================================
# STILE GRAFICO
# =========================================================

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.main-title {
    font-size: 42px;
    font-weight: 850;
    letter-spacing: -1px;
    margin-bottom: 6px;
    color: #111827;
}

.subtitle {
    color: #6b7280;
    font-size: 17px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 28px;
    font-weight: 780;
    margin-top: 34px;
    margin-bottom: 18px;
    color: #111827;
}

.card-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    margin-bottom: 22px;
}

.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
    min-height: 118px;
}

.card-label {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 10px;
}

.card-value {
    font-size: 31px;
    font-weight: 800;
    color: #111827;
    line-height: 1.15;
    white-space: normal;
}

.card-value-small {
    font-size: 21px;
    font-weight: 800;
    color: #111827;
    line-height: 1.25;
    white-space: normal;
}

.positive {
    color: #047857;
}

.negative {
    color: #b91c1c;
}

.signal-box {
    border-radius: 18px;
    padding: 20px 22px;
    margin-top: 8px;
    margin-bottom: 28px;
    line-height: 1.55;
    font-size: 16px;
}

.signal-red {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
}

.signal-yellow {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
}

.signal-green {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}

.signal-blue {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
}

@media (max-width: 1100px) {
    .card-grid-4 {
        grid-template-columns: 1fr;
    }

    .main-title {
        font-size: 34px;
    }

    .card-value {
        font-size: 28px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# FORMATTAZIONE
# =========================================================

def fmt_eur(value):
    try:
        return f"{float(value):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def fmt_eur_3(value):
    try:
        return f"{float(value):,.3f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def fmt_pct(value):
    try:
        return f"{float(value):.2f}%".replace(".", ",")
    except Exception:
        return "-"


def fmt_num(value):
    try:
        return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def safe_text(value):
    return html.escape(str(value))


# =========================================================
# API KEY TWELVE DATA
# =========================================================

def get_twelve_key():
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except Exception:
        return ""


# =========================================================
# FUNZIONI TECNICHE
# =========================================================

def clean_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    required_cols = ["Open", "High", "Low", "Close"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = 0

    return df


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def atr(df, length=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(length).mean()


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI14"] = rsi(df["Close"], 14)

    macd_line, macd_signal, macd_hist = macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"] = macd_hist

    df["ATR14"] = atr(df, 14)
    df["VOLUME_MA20"] = df["Volume"].rolling(20).mean()

    return df.dropna()


# =========================================================
# DOWNLOAD DATI: TWELVE DATA + FALLBACK YAHOO
# =========================================================

@st.cache_data(ttl=600)
def download_twelve_data(symbol, interval, outputsize=200):
    api_key = get_twelve_key()

    if not api_key:
        return pd.DataFrame()

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
        "order": "ASC"
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        data = response.json()

        if not isinstance(data, dict):
            return pd.DataFrame()

        if "values" not in data:
            return pd.DataFrame()

        rows = data["values"]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        if "datetime" not in df.columns:
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }

        df = df.rename(columns=rename_map)

        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        else:
            df["Volume"] = 0

        return clean_df(df)

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def download_yahoo_data(ticker, period, interval):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False
        )
        return clean_df(df)
    except Exception:
        return pd.DataFrame()


def convert_usd_df_to_eur(df, eurusd):
    df = df.copy()

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col] / eurusd

    return df


@st.cache_data(ttl=900)
def get_eurusd():
    df = download_twelve_data(FX_SYMBOL, "1day", 10)

    if not df.empty:
        return float(df["Close"].iloc[-1]), "Twelve Data EUR/USD"

    yahoo_fx = download_yahoo_data("EURUSD=X", "5d", "1d")

    if not yahoo_fx.empty:
        return float(yahoo_fx["Close"].iloc[-1]), "Yahoo EUR/USD fallback"

    return None, "Cambio non disponibile"


def get_price_source():
    eurusd, fx_source = get_eurusd()

    df = download_twelve_data(TICKER_USD, "15min", 200)

    if eurusd and not df.empty:
        df_eur = convert_usd_df_to_eur(df, eurusd)
        last_price = float(df_eur["Close"].iloc[-1])
        return last_price, f"Twelve Data BBAI convertito EUR ({fx_source})", df_eur

    yahoo_df = download_yahoo_data(TICKER_USD, "10d", "15m")

    if eurusd and not yahoo_df.empty:
        yahoo_df = convert_usd_df_to_eur(yahoo_df, eurusd)
        last_price = float(yahoo_df["Close"].iloc[-1])
        return last_price, f"Yahoo BBAI convertito EUR ({fx_source})", yahoo_df

    return None, "Dato automatico non disponibile", pd.DataFrame()


def get_tf_data():
    eurusd, fx_source = get_eurusd()

    if not eurusd:
        return {}, "Cambio EUR/USD non disponibile"

    df_15m = download_twelve_data(TICKER_USD, "15min", 500)
    df_1h = download_twelve_data(TICKER_USD, "1h", 500)
    df_1d = download_twelve_data(TICKER_USD, "1day", 300)

    source = f"Twelve Data BBAI convertito EUR ({fx_source})"

    if df_15m.empty or df_1h.empty or df_1d.empty:
        df_15m = download_yahoo_data(TICKER_USD, "10d", "15m")
        df_1h = download_yahoo_data(TICKER_USD, "3mo", "1h")
        df_1d = download_yahoo_data(TICKER_USD, "1y", "1d")
        source = f"Yahoo BBAI convertito EUR ({fx_source})"

    if df_15m.empty or df_1h.empty or df_1d.empty:
        return {}, source

    df_15m = convert_usd_df_to_eur(df_15m, eurusd)
    df_1h = convert_usd_df_to_eur(df_1h, eurusd)
    df_1d = convert_usd_df_to_eur(df_1d, eurusd)

    df_30m = resample_30m(df_15m)
    df_4h = resample_4h(df_1h)

    return {
        "15m": df_15m,
        "30m": df_30m,
        "1h": df_1h,
        "4h": df_4h,
        "1D": df_1d,
    }, source


def resample_30m(df_15m):
    df = pd.DataFrame()
    df["Open"] = df_15m["Open"].resample("30min").first()
    df["High"] = df_15m["High"].resample("30min").max()
    df["Low"] = df_15m["Low"].resample("30min").min()
    df["Close"] = df_15m["Close"].resample("30min").last()
    df["Volume"] = df_15m["Volume"].resample("30min").sum()
    return df.dropna()


def resample_4h(df_1h):
    df = pd.DataFrame()
    df["Open"] = df_1h["Open"].resample("4h").first()
    df["High"] = df_1h["High"].resample("4h").max()
    df["Low"] = df_1h["Low"].resample("4h").min()
    df["Close"] = df_1h["Close"].resample("4h").last()
    df["Volume"] = df_1h["Volume"].resample("4h").sum()
    return df.dropna()


# =========================================================
# ANALISI TECNICA BBAI
# =========================================================

def support_resistance(df, lookback=30):
    recent = df.tail(lookback)
    return float(recent["Low"].min()), float(recent["High"].max())


def trend_state(last):
    price = last["Close"]
    ema20_value = last["EMA20"]
    ema50_value = last["EMA50"]

    if price > ema20_value > ema50_value:
        return "Rialzista"

    if price < ema20_value < ema50_value:
        return "Ribassista"

    return "Neutrale"


def rsi_state(value):
    if value >= 70:
        return "Ipercomprato"
    if value >= 55:
        return "Forte"
    if value >= 45:
        return "Neutrale"
    if value >= 30:
        return "Debole"
    return "Ipervenduto"


def macd_state(last, prev):
    if last["MACD_HIST"] > 0 and last["MACD_HIST"] > prev["MACD_HIST"]:
        return "Rialzista in rafforzamento"

    if last["MACD_HIST"] > 0 and last["MACD_HIST"] < prev["MACD_HIST"]:
        return "Positivo ma in rallentamento"

    if last["MACD_HIST"] < 0 and last["MACD_HIST"] > prev["MACD_HIST"]:
        return "Negativo ma in recupero"

    return "Ribassista"


def volume_state(last):
    if last["Volume"] > last["VOLUME_MA20"] * 1.3:
        return "Forti"

    if last["Volume"] > last["VOLUME_MA20"]:
        return "Sopra media"

    return "Sotto media"


def technical_score(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0

    if last["Close"] > last["EMA20"]:
        score += 1

    if last["EMA20"] > last["EMA50"]:
        score += 1

    if last["RSI14"] > 50:
        score += 1

    if last["MACD_HIST"] > prev["MACD_HIST"]:
        score += 1

    if last["Volume"] > last["VOLUME_MA20"]:
        score += 1

    return score


def analyze_timeframe(df, label):
    df = add_indicators(df)

    if df.empty or len(df) < 2:
        raise ValueError(f"Dati insufficienti per {label}")

    last = df.iloc[-1]
    prev = df.iloc[-2]

    support, resistance = support_resistance(df)
    score = technical_score(df)

    return {
        "label": label,
        "df": df,
        "price": float(last["Close"]),
        "trend": trend_state(last),
        "rsi": float(last["RSI14"]),
        "rsi_state": rsi_state(float(last["RSI14"])),
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "macd": macd_state(last, prev),
        "volume": volume_state(last),
        "atr": float(last["ATR14"]),
        "support": support,
        "resistance": resistance,
        "score": score,
    }


def build_operational_signal(analysis):
    tf15 = analysis["15m"]
    tf30 = analysis["30m"]
    tf1h = analysis["1h"]
    tf4h = analysis["4h"]
    tf1d = analysis["1D"]

    weighted = (
        tf15["score"] * 0.15 +
        tf30["score"] * 0.20 +
        tf1h["score"] * 0.30 +
        tf4h["score"] * 0.25 +
        tf1d["score"] * 0.10
    )

    price = tf1h["price"]
    atr_1h = tf1h["atr"]

    stop = price - 1.2 * atr_1h
    tp1 = price + 1.5 * atr_1h
    tp2 = price + 2.5 * atr_1h

    if weighted >= 4:
        signal = "LONG CONFERMATO"
        css = "signal-green"
        action = (
            "La struttura tecnica è costruttiva. La tenuta o l’ingresso sono più coerenti se il prezzo resta sopra EMA20 su 1h e 4h, "
            "con RSI sopra 50 e volumi in conferma."
        )

    elif weighted >= 3:
        signal = "LONG PRUDENTE"
        css = "signal-blue"
        action = (
            "Il titolo mostra segnali di recupero, ma la conferma non è ancora completa. Meglio attendere la rottura della resistenza 1h/4h "
            "prima di aumentare l’esposizione."
        )

    elif weighted >= 2:
        signal = "ATTENDERE"
        css = "signal-yellow"
        action = (
            "Il quadro è misto. Non c’è ancora un vantaggio tecnico sufficiente per mediare. Il recupero diventa più credibile solo con RSI sopra 50 "
            "e prezzo sopra EMA20 su 1h."
        )

    else:
        signal = "NO LONG / RISCHIO RIBASSISTA"
        css = "signal-red"
        action = (
            "La struttura resta debole. In questa fase è preferibile proteggere il capitale, evitare mediazioni impulsive e attendere una conferma tecnica reale."
        )

    return {
        "signal": signal,
        "css": css,
        "action": action,
        "weighted": weighted,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
    }


# =========================================================
# ANALISI MERCATO / SETTORE
# =========================================================

def analyze_market_asset(ticker, description):
    df = download_twelve_data(ticker, "1day", 100)

    if df.empty:
        df = download_yahoo_data(ticker, "3mo", "1d")

    if df.empty or len(df) < 40:
        return {
            "Ticker": ticker,
            "Settore": description,
            "Variazione oggi": "-",
            "Trend breve": "Dato non disponibile",
            "RSI": "-",
            "Lettura": "Neutrale",
            "Score": 0
        }

    df = add_indicators(df)

    if df.empty or len(df) < 2:
        return {
            "Ticker": ticker,
            "Settore": description,
            "Variazione oggi": "-",
            "Trend breve": "Dato non disponibile",
            "RSI": "-",
            "Lettura": "Neutrale",
            "Score": 0
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    current_price = float(last["Close"])
    previous_close = float(prev["Close"])

    daily_change = ((current_price - previous_close) / previous_close) * 100
    above_ema20 = current_price > float(last["EMA20"])
    rsi_value = float(last["RSI14"])

    score = 0

    if ticker == "VIXY":
        if daily_change < 0:
            score += 1
        if current_price < float(last["EMA20"]):
            score += 1
        if rsi_value < 50:
            score += 1

        trend = "Sotto EMA20" if current_price < float(last["EMA20"]) else "Sopra EMA20"

    else:
        if daily_change > 0:
            score += 1
        if above_ema20:
            score += 1
        if rsi_value > 50:
            score += 1

        trend = "Sopra EMA20" if above_ema20 else "Sotto EMA20"

    if score >= 2:
        reading = "Favorevole"
    elif score == 1:
        reading = "Neutrale"
    else:
        reading = "Sfavorevole"

    return {
        "Ticker": ticker,
        "Settore": description,
        "Variazione oggi": f"{daily_change:.2f}%".replace(".", ","),
        "Trend breve": trend,
        "RSI": f"{rsi_value:.1f}".replace(".", ","),
        "Lettura": reading,
        "Score": score
    }


def build_market_context():
    rows = []

    for ticker, description in SECTOR_TICKERS.items():
        rows.append(analyze_market_asset(ticker, description))

    total_score = sum(row["Score"] for row in rows)

    if total_score >= 12:
        context = "FAVOREVOLE"
        css = "signal-green"
        message = (
            "Il contesto di mercato è favorevole: tecnologia, AI/small cap o difesa stanno sostenendo il sentiment, "
            "mentre la volatilità non mostra pressioni rilevanti. In questo scenario un eventuale rimbalzo di BBAI ha maggiore qualità, "
            "ma resta necessario attendere conferme tecniche sul titolo."
        )

    elif total_score >= 7:
        context = "NEUTRALE / MISTO"
        css = "signal-yellow"
        message = (
            "Il contesto di mercato è misto: alcuni settori aiutano il trade, altri non confermano ancora. "
            "In questa situazione BBAI può rimbalzare, ma la mediazione deve essere subordinata a conferme su 1h e 4h."
        )

    else:
        context = "SFAVOREVOLE"
        css = "signal-red"
        message = (
            "Il contesto di mercato è sfavorevole: tecnologia, small cap o settore AI non stanno sostenendo il movimento, "
            "oppure la volatilità è in aumento. In questo scenario è preferibile evitare mediazioni impulsive e proteggere il capitale."
        )

    return {
        "rows": rows,
        "total_score": total_score,
        "context": context,
        "css": css,
        "message": message
    }


# =========================================================
# SIDEBAR INPUT
# =========================================================

st.sidebar.markdown("## La tua posizione")

entry_price = st.sidebar.number_input(
    "Prezzo medio di carico (€)",
    min_value=0.0,
    value=0.0,
    step=0.01,
    format="%.3f",
    key="entry_price_input"
)

capital = st.sidebar.number_input(
    "Capitale investito (€)",
    min_value=0.0,
    value=0.0,
    step=50.0,
    format="%.2f",
    key="capital_input"
)

manual_quantity = st.sidebar.number_input(
    "Numero azioni possedute (opzionale)",
    min_value=0.0,
    value=0.0,
    step=1.0,
    format="%.4f",
    key="manual_quantity_input"
)

manual_price = st.sidebar.number_input(
    "Prezzo attuale Trade Republic manuale (opzionale)",
    min_value=0.0,
    value=0.0,
    step=0.01,
    format="%.3f",
    key="manual_price_input"
)

st.sidebar.caption(
    "Se inserisci il prezzo manuale visto su Trade Republic, la webapp userà quello per calcolare guadagno/perdita. "
    "Se lo lasci a zero, userà il prezzo automatico Twelve Data/Yahoo convertito in euro."
)


# =========================================================
# CARICAMENTO DATI
# =========================================================

with st.spinner("Aggiorno prezzo, posizione, analisi tecnica e contesto di mercato..."):
    auto_price, price_source, price_df = get_price_source()
    tf_data, tf_source = get_tf_data()
    market_context = build_market_context()

    analyses = {}

    min_candles = {
        "15m": 50,
        "30m": 40,
        "1h": 50,
        "4h": 25,
        "1D": 60,
    }

    if tf_data:
        for label, df in tf_data.items():
            required = min_candles.get(label, 50)

            if not df.empty and len(df) >= required:
                try:
                    analyses[label] = analyze_timeframe(df, label)
                except Exception:
                    pass

    technical_available = all(label in analyses for label in ["15m", "30m", "1h", "4h", "1D"])
    operational = build_operational_signal(analyses) if technical_available else None


if manual_price > 0:
    current_price = manual_price
    used_price_source = "Prezzo manuale Trade Republic"
elif auto_price is not None:
    current_price = auto_price
    used_price_source = price_source
else:
    current_price = None
    used_price_source = "Prezzo non disponibile"


# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="main-title">BBAI – Dashboard posizione Trade Republic</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Monitoraggio posizione, utile/perdita in euro, contesto di mercato e analisi tecnica multi-timeframe su 15m, 30m, 1h, 4h e 1D.</div>',
    unsafe_allow_html=True
)


if current_price is None:
    st.markdown("""
    <div class="signal-box signal-yellow">
        Il prezzo automatico non è disponibile in questo momento.
        Inserisci il prezzo attuale che vedi su Trade Republic nel campo laterale per calcolare comunque la posizione.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# RISULTATO POSIZIONE
# =========================================================

if current_price is not None and entry_price > 0 and capital > 0:
    quantity = manual_quantity if manual_quantity > 0 else capital / entry_price
    current_value = quantity * current_price
    pnl_eur = current_value - capital
    pnl_pct = (pnl_eur / capital) * 100
    breakeven_gap = ((entry_price - current_price) / current_price) * 100

    pnl_class = "positive" if pnl_eur >= 0 else "negative"

    st.markdown('<div class="section-title">Risultato attuale della posizione</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card-grid-4">
        <div class="card">
            <div class="card-label">Prezzo attuale usato</div>
            <div class="card-value">{fmt_eur_3(current_price)}</div>
        </div>
        <div class="card">
            <div class="card-label">Valore attuale posizione</div>
            <div class="card-value">{fmt_eur(current_value)}</div>
        </div>
        <div class="card">
            <div class="card-label">Guadagno / perdita</div>
            <div class="card-value {pnl_class}">{fmt_eur(pnl_eur)}</div>
        </div>
        <div class="card">
            <div class="card-label">Performance posizione</div>
            <div class="card-value {pnl_class}">{fmt_pct(pnl_pct)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card-grid-4">
        <div class="card">
            <div class="card-label">Capitale investito</div>
            <div class="card-value-small">{fmt_eur(capital)}</div>
        </div>
        <div class="card">
            <div class="card-label">Prezzo medio di carico</div>
            <div class="card-value-small">{fmt_eur_3(entry_price)}</div>
        </div>
        <div class="card">
            <div class="card-label">Azioni considerate</div>
            <div class="card-value-small">{fmt_num(quantity)}</div>
        </div>
        <div class="card">
            <div class="card-label">Fonte prezzo</div>
            <div class="card-value-small">{safe_text(used_price_source)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if pnl_eur < 0:
        st.markdown(f"""
        <div class="signal-box signal-yellow">
            La posizione è sotto carico. Per tornare al prezzo medio di carico serve un recupero di circa
            <b>{fmt_pct(breakeven_gap)}</b> dal prezzo attuale.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="signal-box signal-green">
            La posizione è sopra il prezzo medio di carico. In questa fase diventa importante proteggere il profitto e valutare eventuali prese parziali.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown('<div class="section-title">Risultato attuale della posizione</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="signal-box signal-blue">
        Inserisci nella barra laterale prezzo medio di carico e capitale investito.
        Se il prezzo automatico non è disponibile, inserisci anche il prezzo attuale visto su Trade Republic.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# INDICAZIONE OPERATIVA
# =========================================================

st.markdown('<div class="section-title">Indicazione operativa</div>', unsafe_allow_html=True)

if operational is not None:
    combined_note = ""

    if market_context["context"] == "SFAVOREVOLE" and operational["signal"] in ["LONG CONFERMATO", "LONG PRUDENTE"]:
        combined_note = (
            "<br><br><b>Nota di prudenza:</b> il titolo mostra segnali tecnici positivi, "
            "ma il contesto di mercato non conferma ancora. Meglio ridurre la dimensione dell’operazione "
            "o attendere una conferma più solida."
        )

    elif market_context["context"] == "FAVOREVOLE" and operational["signal"] in ["ATTENDERE", "NO LONG / RISCHIO RIBASSISTA"]:
        combined_note = (
            "<br><br><b>Nota:</b> il mercato sta migliorando, ma BBAI non ha ancora confermato tecnicamente. "
            "Il contesto aiuta, ma il titolo deve comunque recuperare EMA20 su 1h e mostrare volumi in aumento."
        )

    elif market_context["context"] == "FAVOREVOLE" and operational["signal"] in ["LONG CONFERMATO", "LONG PRUDENTE"]:
        combined_note = (
            "<br><br><b>Conferma positiva:</b> il titolo e il mercato si stanno muovendo nella stessa direzione. "
            "Il setup è più interessante, purché il prezzo non perda i supporti di breve."
        )

    st.markdown(f"""
    <div class="signal-box {operational["css"]}">
        <b>{safe_text(operational["signal"])}</b><br>
        {safe_text(operational["action"])}
        {combined_note}
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="signal-box signal-yellow">
        Analisi tecnica BBAI non disponibile in questo momento.
        Verifica che la API key Twelve Data sia nei Secrets di Streamlit e attendi qualche minuto se hai fatto molti refresh.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# CONTESTO MERCATO E SETTORE
# =========================================================

st.markdown('<div class="section-title">Contesto mercato e settore</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="signal-box {market_context["css"]}">
    <b>Contesto mercato: {safe_text(market_context["context"])}</b><br>
    {safe_text(market_context["message"])}
</div>
""", unsafe_allow_html=True)

market_df = pd.DataFrame(market_context["rows"])

if not market_df.empty:
    market_df_view = market_df.drop(columns=["Score"])
    st.dataframe(market_df_view, use_container_width=True, hide_index=True)

st.caption(
    "La lettura del contesto confronta BBAI con Nasdaq, small cap USA, ETF AI/robotics, difesa e volatilità tramite VIXY."
)


# =========================================================
# LIVELLI OPERATIVI
# =========================================================

if operational is not None:
    st.markdown('<div class="section-title">Livelli operativi dinamici</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card-grid-4">
        <div class="card">
            <div class="card-label">Supporto 1h</div>
            <div class="card-value-small">{fmt_eur_3(analyses["1h"]["support"])}</div>
        </div>
        <div class="card">
            <div class="card-label">Resistenza 1h</div>
            <div class="card-value-small">{fmt_eur_3(analyses["1h"]["resistance"])}</div>
        </div>
        <div class="card">
            <div class="card-label">Stop dinamico ATR</div>
            <div class="card-value-small">{fmt_eur_3(operational["stop"])}</div>
        </div>
        <div class="card">
            <div class="card-label">Take Profit 1 / 2</div>
            <div class="card-value-small">{fmt_eur_3(operational["tp1"])} / {fmt_eur_3(operational["tp2"])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# GRAFICO TECNICO
# =========================================================

if technical_available:
    st.markdown('<div class="section-title">Grafico tecnico 1h</div>', unsafe_allow_html=True)

    chart_df = analyses["1h"]["df"].tail(140)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"],
        name="BBAI"
    ))

    fig.add_trace(go.Scatter(
        x=chart_df.index,
        y=chart_df["EMA20"],
        mode="lines",
        name="EMA20"
    ))

    fig.add_trace(go.Scatter(
        x=chart_df.index,
        y=chart_df["EMA50"],
        mode="lines",
        name="EMA50"
    ))

    fig.add_hline(
        y=analyses["1h"]["support"],
        line_dash="dash",
        annotation_text="Supporto 1h",
        annotation_position="bottom right"
    )

    fig.add_hline(
        y=analyses["1h"]["resistance"],
        line_dash="dash",
        annotation_text="Resistenza 1h",
        annotation_position="top right"
    )

    fig.update_layout(
        height=620,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# ANALISI MULTI-TIMEFRAME
# =========================================================

st.markdown('<div class="section-title">Analisi tecnica multi-timeframe</div>', unsafe_allow_html=True)

if technical_available:
    summary_rows = []

    for label in ["15m", "30m", "1h", "4h", "1D"]:
        tf = analyses[label]

        summary_rows.append({
            "Timeframe": label,
            "Trend": tf["trend"],
            "RSI": round(tf["rsi"], 1),
            "Stato RSI": tf["rsi_state"],
            "MACD": tf["macd"],
            "Volumi": tf["volume"],
            "Supporto": fmt_eur_3(tf["support"]),
            "Resistenza": fmt_eur_3(tf["resistance"]),
            "Forza tecnica": f'{tf["score"]}/5'
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    for label in ["15m", "30m", "1h", "4h", "1D"]:
        tf = analyses[label]

        with st.expander(f"Dettaglio timeframe {label}"):
            c1, c2, c3 = st.columns(3)

            c1.write(f"**Prezzo:** {fmt_eur_3(tf['price'])}")
            c1.write(f"**Trend:** {tf['trend']}")
            c1.write(f"**EMA20:** {fmt_eur_3(tf['ema20'])}")
            c1.write(f"**EMA50:** {fmt_eur_3(tf['ema50'])}")

            c2.write(f"**RSI 14:** {tf['rsi']:.1f} – {tf['rsi_state']}")
            c2.write(f"**MACD:** {tf['macd']}")
            c2.write(f"**Volumi:** {tf['volume']}")

            c3.write(f"**Supporto:** {fmt_eur_3(tf['support'])}")
            c3.write(f"**Resistenza:** {fmt_eur_3(tf['resistance'])}")
            c3.write(f"**ATR:** {fmt_eur_3(tf['atr'])}")
            c3.write(f"**Forza tecnica:** {tf['score']}/5")

else:
    available_labels = list(analyses.keys())

    if available_labels:
        st.markdown(f"""
        <div class="signal-box signal-yellow">
            Alcuni dati tecnici sono stati caricati, ma non tutti i timeframe sono disponibili.
            Timeframe disponibili: <b>{safe_text(", ".join(available_labels))}</b>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="signal-box signal-yellow">
            Al momento non sono disponibili dati sufficienti per costruire l’analisi tecnica.
            Controlla che la API key Twelve Data sia stata salvata correttamente nei Secrets.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# NOTA FINALE
# =========================================================

st.caption(
    f"Ultimo aggiornamento: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} — "
    f"Fonte tecnica: {tf_source if 'tf_source' in globals() else 'non disponibile'}. "
    "La webapp usa Twelve Data come fonte principale e Yahoo Finance come fallback. "
    "Per la massima aderenza a Trade Republic puoi inserire manualmente il prezzo attuale visto sul broker. "
    "Le indicazioni sono basate su analisi tecnica e non costituiscono consulenza finanziaria personalizzata."
)
