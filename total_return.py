import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
import io

# ------------------------------
# データクリーニング
# ------------------------------
def clean_data(df):
    """CSVの生データを使いやすい型に変換します"""
    num_cols = ["数量", "売却/決済金額", "費用", "取得/新規金額", "地方税"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace(['', '--'], None), errors='coerce')
    if "損益金額/徴収額" in df.columns:
        df["損益金額/徴収額"] = (
            df["損益金額/徴収額"].astype(str).str.replace("+", "", regex=False).replace(['', '--'], None)
        )
        df["損益金額/徴収額"] = pd.to_numeric(df["損益金額/徴収額"], errors='coerce')
    for c in ["約定日", "受渡日", "取得/新規年月日"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%Y/%m/%d", errors="coerce")
    for c in ["銘柄", "取引", "銘柄コード"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

# ------------------------------
# 取引種別データの抽出
# ------------------------------
def extract_sections(df):
    """売却、配当、税金など種類ごとにデータを分ける"""
    TAX_SALE = ["譲渡益税徴収額"]
    TAX_SALE_REF = ["譲渡益税還付金"]
    TAX_DIV = ["配当所得税徴収額"]
    mask_div = df["取引"].str.contains("配当", na=False)
    df_div = df[mask_div & df["受渡日"].notnull() & df["損益金額/徴収額"].notnull()]
    tax_div = df[df["銘柄コード"].isin(TAX_DIV)]
    mask_sale = df["取引"].str.contains("売", na=False)
    df_sale = df[mask_sale & df["受渡日"].notnull() & df["損益金額/徴収額"].notnull()]
    tax_sale = df[df["銘柄コード"].isin(TAX_SALE)]
    tax_sale_ref = df[df["銘柄コード"].isin(TAX_SALE_REF)]
    return df_div, tax_div, df_sale, tax_sale, tax_sale_ref

# ------------------------------
# 日毎・累積損益の計算
# ------------------------------
def make_daily_series(df_data, df_tax, df_tax_ref, tax_col="地方税"):
    """日次・累積損益の推移を算出"""
    if df_data.empty:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    main = df_data.groupby("受渡日")["損益金額/徴収額"].sum()
    tax_index = df_tax["受渡日"] if not df_tax.empty else pd.Series(dtype='datetime64[ns]')
    tax_ref_index = df_tax_ref["受渡日"] if not df_tax_ref.empty else pd.Series(dtype='datetime64[ns]')
    tax_total = (
        df_tax.groupby("受渡日")["損益金額/徴収額"].sum().fillna(0) +
        df_tax.groupby("受渡日")[tax_col].sum().fillna(0)
    ) if not df_tax.empty else pd.Series(dtype='float64')
    tax_ref_total = (
        df_tax_ref.groupby("受渡日")["損益金額/徴収額"].sum().fillna(0) +
        df_tax_ref.groupby("受渡日")[tax_col].sum().fillna(0)
    ) if not df_tax_ref.empty else pd.Series(dtype='float64')
    idx = main.index.union(tax_index).union(tax_ref_index).sort_values()
    net = main.reindex(idx, fill_value=0) - tax_total.reindex(idx, fill_value=0) + tax_ref_total.reindex(idx, fill_value=0)
    cum = net.cumsum()
    return net, cum

# ------------------------------
# 英語表記グラフの描画
# ------------------------------
def plot_two_panels(sale, sale_cum, div, div_cum, file_title):
    """キャピタルゲイン・配当金を英語でグラフ表示（matplotlib）"""
    fig, axs = plt.subplots(2, 1, figsize=(7, 12), sharex=True)
    ax1, ax2 = axs[0], axs[1]

    all_vals = np.concatenate([s.values for s in [sale, sale_cum, div, div_cum] if not s.empty])
    if all_vals.size == 0:
        ylim = (-1000, 1000)
    else:
        vmin, vmax = all_vals.min(), all_vals.max()
        padding = max((vmax - vmin) * 0.1, 1000)
        ylim = (vmin - padding, vmax + padding)

    # Capital Gains (上段)
    if not sale.empty:
        ax1.bar(sale.index, sale.values,
                color=["green" if x >= 0 else "red" for x in sale.values], width=2)
    if not sale_cum.empty:
        ax1.plot(sale_cum.index, sale_cum.values, color="blue", marker="o", markersize=4, linestyle='-')
        x, y = sale_cum.index[-1], sale_cum.values[-1]
        ax1.annotate(f"Total: ¥{y:,.0f}", xy=(x, y), xytext=(10, 10),
                     textcoords="offset points", fontsize=12, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8))
    ax1.set_title("Capital Gains", fontsize=14)
    ax1.grid(axis="y", linestyle='--', alpha=0.6)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylim(ylim)

    # Dividends (下段)
    if not div.empty:
        ax2.bar(div.index, div.values,
                color=["green" if x >= 0 else "red" for x in div.values], width=2)
    if not div_cum.empty:
        ax2.plot(div_cum.index, div_cum.values, color="blue", marker="o", markersize=4, linestyle='-')
        x, y = div_cum.index[-1], div_cum.values[-1]
        ax2.annotate(f"Total: ¥{y:,.0f}", xy=(x, y), xytext=(10, 10),
                     textcoords="offset points", fontsize=12, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8))
    ax2.set_title("Dividends (Income)", fontsize=14)
    ax2.grid(axis="y", linestyle='--', alpha=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylim(ylim)
    ax2.set_xlabel("Settlement Date", fontsize=12)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(f"[{file_title}] Capital Gains & Dividends\n(Daily and Cumulative Profit)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    return fig

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("SBI証券 日次・累積損益グラフアプリ")
st.markdown("- アップロードした取引明細CSV（SBI証券など）の日ごとの損益・累積利益を可視化します")
st.markdown("※ グラフ内のみ英語表記です。")

uploaded_file = st.file_uploader("SBI証券の取引明細CSVをアップロードしてください", type="csv")
if uploaded_file is not None:
    content = uploaded_file.read()
    # csv部分エンコード: Windows版エクスポートならcp932、Macならutf-8
    try:
        csv_io = io.StringIO(content.decode("cp932"))
    except:
        csv_io = io.StringIO(content.decode("utf-8"))
    # 12列/13列 どちらにも対応
    cols_12 = [
        "銘柄コード", "銘柄", "譲渡益取消区分", "約定日", "数量", "取引", "受渡日",
        "売却/決済金額", "費用", "取得/新規年月日", "取得/新規金額", "損益金額/徴収額"
    ]
    cols_13 = cols_12 + ["地方税"]
    rec12, rec13 = [], []
    for idx, row in enumerate(csv.reader(csv_io)):
        if idx < 21:
            continue  # ヘッダー等をスキップ
        if not row or all(x.strip() == '' for x in row):
            continue
        if len(row) == 12:
            rec12.append(row)
        elif len(row) == 13:
            rec13.append(row)
    df12 = pd.DataFrame(rec12, columns=cols_12) if rec12 else pd.DataFrame(columns=cols_12)
    df13 = pd.DataFrame(rec13, columns=cols_13) if rec13 else pd.DataFrame(columns=cols_13)
    df = pd.concat([df12, df13], ignore_index=True)
    df = clean_data(df)

    # 種別別にデータ抽出と集計
    df_div, tax_div, df_sale, tax_sale, tax_sale_ref = extract_sections(df)
    sale_daily, sale_cum = make_daily_series(df_sale, tax_sale, tax_sale_ref)
    div_daily, div_cum = make_daily_series(
        df_div, tax_div, pd.DataFrame(columns=["受渡日", "損益金額/徴収額", "地方税"])
    )
    all_sale = sale_cum.values[-1] if not sale_cum.empty else 0
    all_div = div_cum.values[-1] if not div_cum.empty else 0

    st.subheader("取引データ（全行表示）")
    st.dataframe(df)

    fig = plot_two_panels(sale_daily, sale_cum, div_daily, div_cum, uploaded_file.name.replace('.csv', ''))
    st.pyplot(fig, use_container_width=True)

    st.subheader("損益サマリー")
    st.markdown(f"- 売却益（税引後）：**{all_sale:,.0f} 円**")
    st.markdown(f"- 配当金（税引後）：**{all_div:,.0f} 円**")
    st.markdown(f"- 総損益（税引後）：**{all_sale + all_div:,.0f} 円**")

else:
    st.info("CSVファイルをアップロードしてください。")
