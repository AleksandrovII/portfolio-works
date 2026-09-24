"""
Portfolio Dashboard — Streamlit app.

Run:
    export INVEST_TOKEN='your_token'
    streamlit run app.py
"""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "data", "portfolio.csv")

TRADABLE_TYPES = {"share", "bond", "etf", "precious_metal"}
TYPE_LABELS = {
    "share": "Акции",
    "bond": "Облигации",
    "etf": "ETF",
    "precious_metal": "Металлы",
}

st.set_page_config(page_title="Портфель", page_icon="📊", layout="wide")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_api(token: str) -> pd.DataFrame:
    """Fetch live portfolio from T-Invest API. Token in signature makes cache key unique."""
    try:
        # Import here — library raises SystemExit at module level if token not set,
        # so we guard with the token check in the caller.
        from t_tech.invest import Client
        from portfolio_works_library import fetch_all_positions, get_rates

        with Client(token) as client:
            rates = get_rates(client)
            rows = fetch_all_positions(client, rates)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("quantity", "avg_price", "cur_price", "rub_value"):
            if col in df.columns:
                df[col] = df[col].apply(lambda x: float(x) if x is not None else None)
        df["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return df

    except Exception as exc:
        st.error(f"Ошибка API: {exc}")
        return pd.DataFrame()


def load_portfolio() -> tuple[pd.DataFrame, str]:
    token = os.getenv("INVEST_TOKEN", "")
    if token:
        with st.spinner("Загружаю данные из T-Invest API…"):
            df = _fetch_api(token)
        if not df.empty:
            return df, "api"

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        return df, "csv"

    return pd.DataFrame(), "none"


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_rub(value: float) -> str:
    return f"{value:,.0f} ₽".replace(",", " ")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Инвестиционный портфель")

    df, source = load_portfolio()

    # Source badge + refresh
    c_badge, c_refresh = st.columns([6, 1])
    with c_badge:
        if source == "api":
            st.success("Данные из T-Invest API · кэш 60 мин", icon="✅")
        elif source == "csv":
            ts = df["updated_at"].iloc[0] if "updated_at" in df.columns else "—"
            st.info(f"Данные из локального CSV · Обновлено: {ts}", icon="📁")
        else:
            st.error(
                "Нет данных. Установите INVEST_TOKEN или убедитесь, что data/portfolio.csv существует."
            )
            return
    with c_refresh:
        if st.button("Обновить", use_container_width=True):
            _fetch_api.clear()
            st.rerun()

    # Work only with tradable positions
    df_t = df[df["instrument_type"].isin(TRADABLE_TYPES)].copy()
    df_t["rub_value"] = pd.to_numeric(df_t["rub_value"], errors="coerce").fillna(0.0)

    total_rub = df_t["rub_value"].sum()

    # P&L (RUB-denominated positions only, to avoid stale FX rate issues)
    df_rub = df_t[
        (df_t["currency"] == "rub")
        & df_t["avg_price"].notna()
        & df_t["cur_price"].notna()
        & df_t["quantity"].notna()
    ].copy()
    df_rub["pl"] = (df_rub["cur_price"] - df_rub["avg_price"]) * df_rub["quantity"]
    total_pl = df_rub["pl"].sum()
    cost_basis = (df_rub["avg_price"] * df_rub["quantity"]).sum()
    pl_pct = (total_pl / cost_basis * 100) if cost_basis > 0 else 0.0

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Стоимость портфеля", fmt_rub(total_rub))
    k2.metric(
        "P&L (рублёвые позиции)",
        fmt_rub(total_pl),
        f"{pl_pct:+.1f}%",
        delta_color="normal",
    )
    k3.metric("Позиций", f"{len(df_t)}")

    st.divider()

    # ── Pie chart ─────────────────────────────────────────────────────────────
    st.subheader("Распределение по активам")

    group = st.radio(
        "Группировать по",
        ["Тикер", "Тип инструмента", "Счёт"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if group == "Тикер":
        agg = df_t.groupby("ticker", as_index=False)["rub_value"].sum()
        agg.columns = ["label", "value"]
    elif group == "Тип инструмента":
        df_t["type_label"] = df_t["instrument_type"].map(TYPE_LABELS).fillna(df_t["instrument_type"])
        agg = df_t.groupby("type_label", as_index=False)["rub_value"].sum()
        agg.columns = ["label", "value"]
    else:
        agg = df_t.groupby("account", as_index=False)["rub_value"].sum()
        agg.columns = ["label", "value"]

    fig = px.pie(
        agg,
        names="label",
        values="value",
        hole=0.38,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} ₽<br>%{percent}<extra></extra>",
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=1, y=0.5),
        height=480,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Positions table ───────────────────────────────────────────────────────
    st.subheader("Позиции")

    # Merge P&L back
    if "pl" in df_rub.columns:
        df_t = df_t.merge(
            df_rub[["figi", "account", "pl"]],
            on=["figi", "account"],
            how="left",
        )

    col_map = {
        "ticker": "Тикер",
        "name": "Название",
        "instrument_type": "Тип",
        "account": "Счёт",
        "quantity": "Кол-во",
        "avg_price": "Ср. цена",
        "cur_price": "Тек. цена",
        "rub_value": "Стоимость, ₽",
        "pl": "P&L, ₽",
    }
    cols = [c for c in col_map if c in df_t.columns]
    table = (
        df_t[cols]
        .rename(columns=col_map)
        .sort_values("Стоимость, ₽", ascending=False)
    )

    type_col_cfg = {}
    if "Стоимость, ₽" in table.columns:
        type_col_cfg["Стоимость, ₽"] = st.column_config.NumberColumn(format="%.0f")
    if "Ср. цена" in table.columns:
        type_col_cfg["Ср. цена"] = st.column_config.NumberColumn(format="%.2f")
    if "Тек. цена" in table.columns:
        type_col_cfg["Тек. цена"] = st.column_config.NumberColumn(format="%.2f")
    if "Кол-во" in table.columns:
        type_col_cfg["Кол-во"] = st.column_config.NumberColumn(format="%.0f")
    if "P&L, ₽" in table.columns:
        type_col_cfg["P&L, ₽"] = st.column_config.NumberColumn(format="%+.0f")

    st.dataframe(table, use_container_width=True, hide_index=True, column_config=type_col_cfg)


main()
