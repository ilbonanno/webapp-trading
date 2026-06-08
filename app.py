import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime


# =========================
# CONFIGURAZIONE PAGINA
# =========================

st.set_page_config(
    page_title="BBAI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("BBAI Trading Dashboard")
st.caption("Dashboard tecnica in euro per monitorare BigBear.ai, posizione personale, rischio, stop loss e take profit.")


# =========================
# FUNZIONI INDICATORI
# =========================

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
    histogram = macd_line - signal
    return macd_line, signal, histogram


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
        axis=1,
    ).max(axis=1)

    return tr.rolling(length).mean()


def clean_yfinance_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def add_indicators(df):
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI14"] = rsi(df["Close"], 14)

    macd_line, signal, histogram = macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = signal
    df["MACD_HIST"] = histogram

    df["ATR14"] = atr(df, 14)
    df["VOLUME_MA20"] = df["Volume"].rolling(20).mean()

    return df.dropna()


# =========================
# DATI DI MERCATO
# =========================

@st.cache_data(ttl=300)
def get_eurusd():
    fx = yf.download("EURUSD=X", period="5d", interval="1d", progress=False)
    fx = clean_yfinance_df(fx)

    if fx.empty:
        return None

    return float(fx["Close"].iloc[-1])


@st.cache_data(ttl=300)
def get_market_data(period, interval):
    df = yf.download(
        "BBAI",
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True
    )

    df = clean_yfinance_df(df)

    if df.empty:
        return pd.DataFrame()

    return df


def convert_to_eur(df, eurusd):
    df = df.copy()

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col] / eurusd

    return df


def resample_4h(df):
    df_4h = pd.DataFrame()
    df_4h["Open"] = df["Open"].resample("4h").first()
    df_4h["High"] = df["High"].resample("4h").max()
    df_4h["Low"] = df["Low"].resample("4h").min()
    df_4h["Close"] = df["Close"].resample("4h").last()
    df_4h["Volume"] = df["Volume"].resample("4h").sum()
    return df_4h.dropna()


# =========================
# ANALISI TECNICA
# =========================

def get_trend(row):
    price = row["Close"]
    ema20 = row["EMA20"]
    ema50 = row["EMA50"]

    if price > ema20 > ema50:
        return "Rialzista"
    elif price < ema20 < ema50:
        return "Ribassista"
    else:
        return "Neutrale / laterale"


def get_rsi_state(value):
    if value >= 70:
        return "Ipercomprato"
    elif value >= 55:
        return "Forte"
    elif value >= 45:
        return "Neutrale"
    elif value >= 30:
        return "Debole"
    else:
        return "Ipervenduto"


def get_macd_state(last, prev):
    if last["MACD_HIST"] > 0 and last["MACD_HIST"] > prev["MACD_HIST"]:
        return "Momentum rialzista in rafforzamento"
    elif last["MACD_HIST"] > 0 and last["MACD_HIST"] < prev["MACD_HIST"]:
        return "Momentum positivo ma in rallentamento"
    elif last["MACD_HIST"] < 0 and last["MACD_HIST"] > prev["MACD_HIST"]:
        return "Momentum negativo ma in recupero"
    else:
        return "Momentum ribassista"


def get_volume_state(last):
    if last["Volume"] > last["VOLUME_MA20"] * 1.3:
        return "Volumi forti"
    elif last["Volume"] > last["VOLUME_MA20"]:
        return "Volumi sopra media"
    else:
        return "Volumi sotto media"


def support_resistance(df, lookback=30):
    recent = df.tail(lookback)
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    return support, resistance


def score_timeframe(df):
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


def analyze_df(df, label):
    df = add_indicators(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    support, resistance = support_resistance(df)
    score = score_timeframe(df)

    return {
        "label": label,
        "df": df,
        "price": float(last["Close"]),
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "rsi": float(last["RSI14"]),
        "atr": float(last["ATR14"]),
        "support": support,
        "resistance": resistance,
        "trend": get_trend(last),
        "rsi_state": get_rsi_state(float(last["RSI14"])),
        "macd_state": get_macd_state(last, prev),
        "volume_state": get_volume_state(last),
        "score": score,
    }


def build_signal(tf_15m, tf_1h, tf_4h, tf_1d):
    total_score = (
        tf_15m["score"] * 0.20 +
        tf_1h["score"] * 0.35 +
        tf_4h["score"] * 0.30 +
        tf_1d["score"] * 0.15
    )

    price = tf_1h["price"]
    atr_value = tf_1h["atr"]

    stop_loss = price - (1.2 * atr_value)
    tp1 = price + (1.5 * atr_value)
    tp2 = price + (2.5 * atr_value)

    if total_score >= 4:
        signal = "LONG CONFERMATO"
        action = "Il setup tecnico è costruttivo. L’ingresso ha senso solo se il prezzo resta sopra EMA20 sul timeframe 1h e i volumi non si indeboliscono."
    elif total_score >= 3:
        signal = "LONG PRUDENTE / ATTENDERE CONFERMA"
        action = "Il titolo mostra un recupero, ma non ancora pienamente confermato. Meglio attendere la rottura della prima resistenza 1h con volumi sopra media."
    elif total_score >= 2:
        signal = "NEUTRALE"
        action = "Non c’è ancora un vantaggio tecnico sufficiente. Meglio evitare nuove entrate finché RSI 1h non torna sopra 50 e il prezzo non recupera EMA20."
    else:
        signal = "NO LONG / RISCHIO RIBASSISTA"
        action = "La struttura tecnica resta debole. In questa fase è preferibile proteggere il capitale e non mediare al ribasso senza conferme."

    return {
        "score": total_score,
        "signal": signal,
        "action": action,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
    }


def fmt(value):
    return f"{value:.3f} €"


# =========================
# SIDEBAR INPUT UTENTE
# =========================

st.sidebar.header("La tua posizione")

entry_price = st.sidebar.number_input(
    "Prezzo medio di carico (€)",
    min_value=0.0,
    value=0.0,
    step=0.01,
    format="%.3f"
)

capital = st.sidebar.number_input(
    "Capitale investito (€)",
    min_value=0.0,
    value=0.0,
    step=50.0,
    format="%.2f"
)

manual_quantity = st.sidebar.number_input(
    "Numero azioni possedute (opzionale)",
    min_value=0.0,
    value=0.0,
    step=1.0,
    format="%.4f"
)

st.sidebar.caption("Se non inserisci il numero di azioni, la dashboard lo stima dividendo capitale investito per prezzo medio di carico.")


# =========================
# CARICAMENTO DATI
# =========================

with st.spinner("Carico dati BBAI e calcolo indicatori tecnici..."):
    eurusd = get_eurusd()

    if eurusd is None:
        st.error("Impossibile recuperare il cambio EUR/USD.")
        st.stop()

    df_15m = get_market_data("5d", "15m")
    df_1h = get_market_data("1mo", "1h")
    df_daily = get_market_data("6mo", "1d")

    if df_15m.empty or df_1h.empty or df_daily.empty:
        st.error("Impossibile recuperare i dati di mercato da Yahoo Finance.")
        st.stop()

    df_15m = convert_to_eur(df_15m, eurusd)
    df_1h = convert_to_eur(df_1h, eurusd)
    df_daily = convert_to_eur(df_daily, eurusd)
    df_4h = resample_4h(df_1h)

    tf_15m = analyze_df(df_15m, "15m")
    tf_1h = analyze_df(df_1h, "1h")
    tf_4h = analyze_df(df_4h, "4h")
    tf_1d = analyze_df(df_daily, "Daily")

    signal = build_signal(tf_15m, tf_1h, tf_4h, tf_1d)


current_price = tf_1h["price"]


# =========================
# HEADER OPERATIVO
# =========================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Prezzo attuale stimato", fmt(current_price))
col2.metric("Segnale", signal["signal"])
col3.metric("Score tecnico", f"{signal['score']:.2f} / 5")
col4.metric("Cambio EUR/USD", f"{eurusd:.4f}")

st.info(signal["action"])


# =========================
# POSIZIONE UTENTE
# =========================

st.subheader("Analisi della tua posizione")

if entry_price > 0 and capital > 0:
    if manual_quantity > 0:
        quantity = manual_quantity
    else:
        quantity = capital / entry_price

    pnl_pct = ((current_price - entry_price) / entry_price) * 100
    pnl_eur = (current_price - entry_price) * quantity
    breakeven_gap = ((entry_price - current_price) / current_price) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Prezzo medio inserito", fmt(entry_price))
    col2.metric("Quantità stimata", f"{quantity:.2f}")
    col3.metric("P/L percentuale", f"{pnl_pct:.2f}%")
    col4.metric("P/L stimato", f"{pnl_eur:.2f} €")

    if pnl_pct < 0:
        st.warning(
            f"La posizione è sotto carico. Per tornare al prezzo medio di carico serve un recupero di circa {breakeven_gap:.2f}% dal prezzo attuale."
        )
    else:
        st.success("La posizione è sopra il prezzo medio di carico.")

else:
    st.warning("Inserisci prezzo medio di carico e capitale investito nella barra laterale per calcolare P/L e rischio della posizione.")


# =========================
# LIVELLI OPERATIVI
# =========================

st.subheader("Livelli operativi dinamici")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Stop loss dinamico", fmt(signal["stop_loss"]))
col2.metric("Take profit 1", fmt(signal["tp1"]))
col3.metric("Take profit 2", fmt(signal["tp2"]))
col4.metric("ATR 1h", fmt(tf_1h["atr"]))

if entry_price > 0:
    risk_from_entry = ((entry_price - signal["stop_loss"]) / entry_price) * 100
    tp1_from_entry = ((signal["tp1"] - entry_price) / entry_price) * 100
    tp2_from_entry = ((signal["tp2"] - entry_price) / entry_price) * 100

    st.write(
        f"Rispetto al tuo prezzo medio, lo stop dinamico implica una distanza teorica di **{risk_from_entry:.2f}%**, "
        f"mentre TP1 e TP2 corrispondono rispettivamente a **{tp1_from_entry:.2f}%** e **{tp2_from_entry:.2f}%**."
    )


# =========================
# GRAFICO
# =========================

st.subheader("Grafico BBAI 1h in EUR con EMA20 / EMA50")

chart_df = tf_1h["df"].tail(120)

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"],
        name="BBAI"
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_df.index,
        y=chart_df["EMA20"],
        mode="lines",
        name="EMA20"
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_df.index,
        y=chart_df["EMA50"],
        mode="lines",
        name="EMA50"
    )
)

fig.add_hline(
    y=tf_1h["support"],
    line_dash="dash",
    annotation_text="Supporto 1h",
    annotation_position="bottom right"
)

fig.add_hline(
    y=tf_1h["resistance"],
    line_dash="dash",
    annotation_text="Resistenza 1h",
    annotation_position="top right"
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# TIMEFRAME
# =========================

st.subheader("Analisi multi-timeframe")

timeframes = [tf_15m, tf_1h, tf_4h, tf_1d]

for tf in timeframes:
    with st.expander(f"Timeframe {tf['label']} — Score {tf['score']}/5 — Trend {tf['trend']}"):
        col1, col2, col3 = st.columns(3)

        col1.write(f"**Prezzo:** {fmt(tf['price'])}")
        col1.write(f"**EMA20:** {fmt(tf['ema20'])}")
        col1.write(f"**EMA50:** {fmt(tf['ema50'])}")

        col2.write(f"**RSI 14:** {tf['rsi']:.1f} — {tf['rsi_state']}")
        col2.write(f"**MACD:** {tf['macd_state']}")
        col2.write(f"**Volumi:** {tf['volume_state']}")

        col3.write(f"**Supporto:** {fmt(tf['support'])}")
        col3.write(f"**Resistenza:** {fmt(tf['resistance'])}")
        col3.write(f"**ATR:** {fmt(tf['atr'])}")


# =========================
# REGOLA OPERATIVA
# =========================

st.subheader("Regola operativa")

st.write(
    """
    La mediazione non dovrebbe essere eseguita solo perché il prezzo scende. 
    Il rafforzamento della posizione diventa tecnicamente più sensato solo quando il timeframe 1h mostra una combinazione coerente di segnali:
    prezzo sopra EMA20, RSI sopra 50, MACD in miglioramento e volumi almeno sopra la media.
    """
)

st.caption(
    f"Ultimo aggiornamento: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} — "
    "Dati USA convertiti in euro. Possono esserci differenze rispetto al prezzo mostrato su Trade Republic / LSX."
)
