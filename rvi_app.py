# rvi_app.py
import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from matplotlib import font_manager as fm

# -------------------------
# 全域視覺設定 & 字型（微軟正黑體）
# -------------------------
warnings.filterwarnings(
    "ignore",
    message="Glyph .* missing from current font",
    category=UserWarning,
    module="matplotlib"
)

st.set_page_config(
    page_title="Risk Vibe Indicator (RVI) Dashboard", layout="wide")
sns.set(style="whitegrid")
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def try_set_chinese_font():
    """嘗試載入中文字型（優先使用專案內字型，再嘗試系統字型）。"""
    # 優先順序：專案內字型 > Windows 系統字型 > Linux 系統字型
    font_paths = [
        "NotoSansTC-Regular.ttf",  # 專案根目錄的字型檔（需自行下載放置）
        r"C:\Windows\Fonts\msjh.ttc",  # Windows 微軟正黑體
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 文泉驿微米黑
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux Noto Sans CJK
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                fm._load_fontmanager(try_read_cache=False)
                font_name = fm.FontProperties(fname=font_path).get_name()
                mpl.rcParams["font.family"] = font_name
                mpl.rcParams["font.sans-serif"] = [font_name]
                return f"✅ 已載入字型：{font_name}（{os.path.basename(font_path)}）"
            except Exception as e:
                continue  # 載入失敗，嘗試下一個字型

    return "⚠️ 未找到中文字型，圖表中文可能顯示為方框。請下載 NotoSansTC-Regular.ttf 並放置於專案根目錄，或在側邊欄上傳字型檔。"


font_msg = try_set_chinese_font()

# -------------------------
# 工具函式
# -------------------------


def normalize_minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = np.nanmin(s), np.nanmax(s)
    if np.isclose(mx - mn, 0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def apply_direction(series: pd.Series, larger_is_worse: bool) -> pd.Series:
    """
    方向一致化：回傳「數值越大 = 越糟」的尺度
    如果 larger_is_worse=False，代表數值越大越好 → 先乘以 -1 反向，再做正規化
    """
    return series if larger_is_worse else -series


def heat_to_vibe(h: float) -> str:
    if h < 0.25:
        return "🟢 Calm"
    elif h < 0.5:
        return "🟡 Neutral"
    elif h < 0.75:
        return "🟠 Alert"
    else:
        return "🔴 Critical"


def vibe_color(vibe: str) -> str:
    mapping = {
        "🟢 Calm": "#10B981",     # 綠
        "🟡 Neutral": "#F59E0B",  # 黃
        "🟠 Alert": "#EF4444",    # 橘紅
        "🔴 Critical": "#7F1D1D"  # 深紅
    }
    return mapping.get(vibe, "#6B7280")


@st.cache_data
def load_excel(file_bytes: bytes, sheet: str | int | None = None) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, engine="openpyxl")


def to_bytes_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def build_template_df(n_days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range((pd.Timestamp.today(
    ) - pd.Timedelta(days=n_days)).normalize(), periods=n_days, freq="D")

    df = pd.DataFrame({
        "date": dates,
        "pd": np.clip(rng.normal(0.015, 0.004, len(dates)), 0.005, 0.05),
        "npl": np.clip(rng.normal(0.022, 0.004, len(dates)), 0.01, 0.08),
        "var": np.clip(rng.normal(5.0, 1.2, len(dates)), 2.0, 10.0),
        "liquidity_gap": np.clip(rng.normal(-40, 15, len(dates)), -120, 10),
        "ews_score": np.clip(rng.normal(50, 10, len(dates)), 10, 100)
    })
    return df


def universal_date_parser(series):
    """
    強韌萬用日期解析器：
    支援以下日期格式：
    - YYYYMMDD (20240131)
    - YYYY-MM-DD / YYYY/MM/DD
    - Excel serial date (e.g., 45800)
    - datetime / Timestamp
    - 字串與混合格式
    """
    import pandas as pd
    import numpy as np
    import datetime

    s = series.copy()

    # Step 1: 若為 datetime / pandas Timestamp，直接轉 datetime
    s = s.apply(lambda x: x if not isinstance(x, str) else x)

    # Step 2: 將整數轉字串（用於處理 20250118）
    def convert_int_to_str(x):
        if isinstance(x, (int, float)) and not pd.isna(x):
            x_int = int(x)
            # Excel serial 判斷：小於 300000 （依 Excel epoch）
            if x_int < 300000:
                try:
                    return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=x_int)).strftime("%Y-%m-%d")
                except:
                    return str(x_int)
            else:
                return str(x_int)
        return x

    s = s.apply(convert_int_to_str)

    # Step 3: 嘗試多種日期格式解析
    def try_parse(x):
        # 若本身已是 datetime
        if isinstance(x, (pd.Timestamp, datetime.date, datetime.datetime)):
            return pd.to_datetime(x, errors="coerce")

        # 主要日期格式
        fmts = ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]
        for fmt in fmts:
            try:
                return datetime.datetime.strptime(str(x), fmt)
            except:
                pass

        # 最終 fallback：pandas 自己 parse
        try:
            return pd.to_datetime(x, errors="coerce")
        except:
            return pd.NaT

    s = s.apply(try_parse)

    return s


# -------------------------
# 側邊欄：資料、字型、欄位對映、方向、權重
# -------------------------
st.sidebar.header("📤 資料來源 & 字型")
uploaded = st.sidebar.file_uploader("上傳 Excel（.xlsx）", type=["xlsx"])
sheet_name = st.sidebar.text_input("指定工作表名稱（留空則第一張）", "")
font_file = st.sidebar.file_uploader(
    "（可選）上傳中文字型檔 .ttf/.otf/.ttc", type=["ttf", "otf", "ttc"])

if font_file is not None:
    try:
        fm.fontManager.addfont(font_file)
        fm._load_fontmanager(try_read_cache=False)
        font_name = fm.FontProperties(fname=font_file).get_name()
        mpl.rcParams["font.family"] = font_name
        mpl.rcParams["font.sans-serif"] = [font_name]
        font_msg = f"已改用上傳的字型：{font_name}"
    except Exception as e:
        font_msg = f"上傳字型載入失敗：{e}"

with st.sidebar.expander("⚙️ 一般設定", expanded=True):
    threshold_mode = st.radio(
        "Vibe 門檻模式",
        ["固定分段（0.25/0.5/0.75）", "歷史分位數（25%/50%/75%）"],
        index=0
    )
    show_table = st.checkbox("顯示處理後資料表", value=False)

st.sidebar.info(font_msg)

# 讀資料（支援自動偵測或手動指定工作表）
if uploaded:
    try:
        raw = load_excel(uploaded.getvalue(),
                         sheet=sheet_name if sheet_name.strip() else None)

        # 若回傳 dict（多工作表），取第一張或你指定的 sheet
        if isinstance(raw, dict):
            if sheet_name.strip() and sheet_name in raw:
                df_raw = raw[sheet_name]
            else:
                df_raw = list(raw.values())[0]
        else:
            df_raw = raw

    except Exception as e:
        st.error(f"讀取 Excel 失敗：{e}")
        st.stop()
else:
    st.warning("尚未上傳 Excel，以下示範以隨機模板資料呈現。")
    df_raw = build_template_df(30)

# 日期欄位與指標選取
st.sidebar.header("🧭 欄位對映與方向")
all_cols = df_raw.columns.tolist()
date_col = st.sidebar.selectbox("日期欄位", options=all_cols, index=all_cols.index(
    "date") if "date" in all_cols else 0)

# 指標候選（排除日期欄）
metric_candidates = [c for c in all_cols if c != date_col]
default_metrics = [m for m in ["pd", "npl", "var",
                               "liquidity_gap", "ews_score"] if m in metric_candidates]
metrics = st.sidebar.multiselect(
    "指標欄位（可多選）", options=metric_candidates, default=default_metrics or metric_candidates)

if len(metrics) == 0:
    st.error("請至少選擇一個指標欄位。")
    st.stop()

# 每個指標的「越大越糟？」方向與權重
dir_cols = {}
w_cols = {}
st.sidebar.markdown("**指標方向與權重**")
for m in metrics:
    cols = st.sidebar.columns([1, 1.2])
    with cols[0]:
        dir_cols[m] = st.checkbox(f"{m} 越大越糟？", value=(
            m != "liquidity_gap"))  # 預設 liquidity_gap 反向
    with cols[1]:
        w_cols[m] = st.slider(
            f"{m} 權重", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key=f"w_{m}")

# 權重正規化
w_sum = sum(w_cols.values())
if w_sum == 0:
    weights = {m: 1 / len(metrics) for m in metrics}
else:
    weights = {m: w_cols[m] / w_sum for m in metrics}

st.sidebar.caption("（權重會自動正規化為總和=1）")

# -------------------------
# 主區塊：處理、計算、視覺化
# -------------------------
st.title("Risk Vibe Indicator (RVI) Dashboard")
st.write("上傳 Excel → 對映欄位 → 設定方向/權重 → 即時出圖與匯出")

# 基本清洗
df = df_raw.copy()
# 轉日期
try:
    df[date_col] = universal_date_parser(df[date_col])
    # 若解析失敗，提示後移除空值
    if df[date_col].isna().any():
        st.warning("⚠ 部分日期無法解析（已自動移除）。")
        df = df.dropna(subset=[date_col])
except Exception as e:
    st.error(f"日期欄位解析失敗：{e}")
    st.stop()

# 保留有選取的指標 & 轉 float
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors="coerce")
df = df[[date_col] + metrics].dropna().sort_values(by=date_col).reset_index(drop=True)

# 方向一致化 & 正規化
norm_cols = []
for m in metrics:
    s_dir = apply_direction(df[m], larger_is_worse=dir_cols[m])
    s_norm = normalize_minmax(s_dir)
    norm_col = f"{m}_norm"
    df[norm_col] = s_norm
    norm_cols.append(norm_col)

# 風險熱度分數
risk_heat = np.zeros(len(df))
for m in metrics:
    risk_heat += df[f"{m}_norm"].values * weights[m]
df["risk_heat"] = risk_heat

# Vibe 門檻
if threshold_mode.startswith("歷史分位數"):
    q25, q50, q75 = df["risk_heat"].quantile([0.25, 0.5, 0.75]).tolist()

    def heat_to_vibe_quantile(h):
        if h < q25:
            return "🟢 Calm"
        elif h < q50:
            return "🟡 Neutral"
        elif h < q75:
            return "🟠 Alert"
        else:
            return "🔴 Critical"
    df["vibe"] = df["risk_heat"].apply(heat_to_vibe_quantile)
else:
    df["vibe"] = df["risk_heat"].apply(heat_to_vibe)

df["color"] = df["vibe"].apply(vibe_color)

# 指標貢獻度（最後一天）
latest = df.iloc[-1]
contrib = {m: latest[f"{m}_norm"] * weights[m] for m in metrics}
ser_contrib = pd.Series(contrib).sort_values(ascending=False)

# ---- 上方 KPI 區 ----
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("最新日期", latest[date_col].date().strftime("%Y-%m-%d"))
with kpi2:
    st.metric("Risk Heat", f"{latest['risk_heat']:.3f}")
with kpi3:
    st.markdown(f"**Vibe**")
    st.markdown(
        f"<div style='padding:8px 12px;background:{vibe_color(latest['vibe'])};color:white;border-radius:6px;width:120px;text-align:center'>{latest['vibe']}</div>",
        unsafe_allow_html=True
    )

# ---- 圖 1：色帶圖 ----
fig1, ax1 = plt.subplots(figsize=(10, 2.2))
ax1.bar(df[date_col].dt.strftime("%Y-%m-%d"), [1] *
        len(df), color=df["color"], edgecolor="none")
ax1.set_yticks([])
ax1.set_title("Bank Risk Vibe Indicator（每日氛圍色帶）")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
st.pyplot(fig1, use_container_width=True)
img1 = to_bytes_png(fig1)
plt.close(fig1)

# ---- 圖 2：熱度趨勢 ----
fig2, ax2 = plt.subplots(figsize=(10, 4))
# 背景區間
if threshold_mode.startswith("歷史分位數"):
    ax2.axhspan(0.0, df["risk_heat"].min(), facecolor="#10B981", alpha=0.08)
    ax2.axhspan(df["risk_heat"].min(), df["risk_heat"].quantile(
        0.25), facecolor="#10B981", alpha=0.12, label="Calm")
    ax2.axhspan(df["risk_heat"].quantile(0.25), df["risk_heat"].quantile(
        0.5), facecolor="#F59E0B", alpha=0.12, label="Neutral")
    ax2.axhspan(df["risk_heat"].quantile(0.5), df["risk_heat"].quantile(
        0.75), facecolor="#EF4444", alpha=0.10, label="Alert")
    ax2.axhspan(df["risk_heat"].quantile(0.75), 1.0,
                facecolor="#7F1D1D", alpha=0.10, label="Critical")
else:
    ax2.axhspan(0.00, 0.25, facecolor="#10B981", alpha=0.12, label="Calm")
    ax2.axhspan(0.25, 0.50, facecolor="#F59E0B", alpha=0.12, label="Neutral")
    ax2.axhspan(0.50, 0.75, facecolor="#EF4444", alpha=0.10, label="Alert")
    ax2.axhspan(0.75, 1.00, facecolor="#7F1D1D", alpha=0.10, label="Critical")

ax2.plot(df[date_col], df["risk_heat"], color="#111827",
         linewidth=2, marker="o", markersize=4)
ax2.set_ylim(0, 1)
ax2.set_ylabel("Risk Heat（0~1）")
ax2.set_title("Risk Heat 趨勢（含 Vibe 區間）")
ax2.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
img2 = to_bytes_png(fig2)
plt.close(fig2)

# ---- 圖 3：當日貢獻度 ----
fig3, ax3 = plt.subplots(figsize=(6, 3.5))
sns.barplot(x=ser_contrib.values, y=ser_contrib.index, ax=ax3, color="#374151")
ax3.set_title(f"當日（{latest[date_col].date()}）風險熱度貢獻度")
ax3.set_xlabel("貢獻度")
ax3.set_ylabel("")
plt.tight_layout()
st.pyplot(fig3, use_container_width=False)
img3 = to_bytes_png(fig3)
plt.close(fig3)

# ---- 匯出 ----
st.subheader("下載結果")
col_d1, col_d2, col_d3, col_d4 = st.columns(4)
with col_d1:
    # 處理後資料
    out_csv = df.copy()
    out_csv[date_col] = out_csv[date_col].dt.strftime("%Y-%m-%d")
    st.download_button("⬇️ 下載處理後 CSV", data=out_csv.to_csv(index=False).encode("utf-8-sig"),
                       file_name="risk_vibe_result.csv", mime="text/csv")
with col_d2:
    st.download_button("⬇️ 色帶圖 PNG", data=img1,
                       file_name="risk_vibe_band.png", mime="image/png")
with col_d3:
    st.download_button("⬇️ 趨勢圖 PNG", data=img2,
                       file_name="risk_heat_trend.png", mime="image/png")
with col_d4:
    st.download_button("⬇️ 貢獻度圖 PNG", data=img3,
                       file_name="risk_heat_contrib.png", mime="image/png")

st.divider()

# ---- 模板下載 & 原始資料檢視 ----
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("下載 Excel 模板")
    df_tmpl = build_template_df(30)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_tmpl.to_excel(writer, index=False, sheet_name="risk_metrics")
    buf.seek(0)
    st.download_button("⬇️ 下載範例模板.xlsx", data=buf.getvalue(), file_name="risk_metrics_template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with c2:
    st.subheader("原始資料預覽")
    st.dataframe(df_raw.head(20), use_container_width=True)

if show_table:
    st.subheader("處理後資料（含標準化/分數/Vibe）")
    st.dataframe(df, use_container_width=True)
