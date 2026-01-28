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
# 全域色彩系統 & 設計規範
# -------------------------
COLOR_SCHEME = {
    # UI 新色盤 - 主色與次要色
    "ui": {
        "primary": "#9A0036",              # 紅色 - 品牌主色
        "secondary": {
            "pink": "#EEBAC0",             # 粉紅色
            "purple": "#941C61",           # 紫紅色
            "white": "#FFFFFF",            # 白色 - 背景主色
            "light_gray": "#F5F5F5",       # 淺灰色 - 區塊分隔
            "dark_gray": "#333333"         # 深灰色 - 文字內容
        },
        "accent": {
            "gold": "#FFD700",             # 金色 - 重要資訊和CTA
            "orange": "#FF8C00"            # 橙色 - 重要資訊和CTA
        }
    },
    # 風險色保留 - 風險等級識別
    "risk": {
        "calm": "#10B981",                 # 綠色 - 低風險
        "neutral": "#F59E0B",              # 黃色 - 中性
        "alert": "#EF4444",                # 橘紅 - 警示
        "critical": "#7F1D1D"              # 深紅 - 嚴重
    }
}


def get_ui_color(path: str, default: str = "#333333") -> str:
    """
    快速獲取UI色彩。
    用法: get_ui_color("primary"), get_ui_color("secondary.pink")
    """
    keys = path.split(".")
    color = COLOR_SCHEME["ui"]
    for key in keys:
        if isinstance(color, dict) and key in color:
            color = color[key]
        else:
            return default
    return color if isinstance(color, str) else default


def get_risk_color(vibe: str) -> str:
    """根據 Vibe 標籤返回風險色彩"""
    mapping = {
        "🟢 Calm": COLOR_SCHEME["risk"]["calm"],
        "🟡 Neutral": COLOR_SCHEME["risk"]["neutral"],
        "🟠 Alert": COLOR_SCHEME["risk"]["alert"],
        "🔴 Critical": COLOR_SCHEME["risk"]["critical"]
    }
    return mapping.get(vibe, COLOR_SCHEME["ui"]["secondary"]["dark_gray"])


# -------------------------
# 全域視覺設定 & 字型
# -------------------------
warnings.filterwarnings(
    "ignore",
    message="Glyph .* missing from current font",
    category=UserWarning,
    module="matplotlib"
)

st.set_page_config(
    page_title="Risk Indicator (RI) Dashboard", layout="wide")

# 顯示商標
if os.path.exists("logo.png"):
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image("logo.png", width=120)
    with col_title:
        st.markdown("<h1 style='margin-top: 20px;'>Risk Indicator (RI) Dashboard</h1>", unsafe_allow_html=True)
else:
    st.title("Risk Indicator (RI) Dashboard")

# 注入全域CSS樣式（在所有其他Streamlit調用前執行）
GLOBAL_CSS = f"""
<style>
    /* 根字體與背景 */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft JhengHei', 'Noto Sans TC', sans-serif !important;
        background-color: {COLOR_SCHEME['ui']['secondary']['white']};
        color: {COLOR_SCHEME['ui']['secondary']['dark_gray']};
    }}
    
    /* 主容器背景 */
    .stApp {{
        background-color: {COLOR_SCHEME['ui']['secondary']['white']};
    }}
    
    /* 標題樣式 */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLOR_SCHEME['ui']['primary']};
        font-weight: 600;
    }}
    
    /* 文字層級 */
    p, span, div {{
        color: {COLOR_SCHEME['ui']['secondary']['dark_gray']};
    }}
    
    /* 卡片容器樣式 */
    .stMetric, .stContainer {{
        background-color: {COLOR_SCHEME['ui']['secondary']['white']};
        border: 1px solid {COLOR_SCHEME['ui']['secondary']['light_gray']};
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    
    /* 分隔線 */
    hr {{
        border-color: {COLOR_SCHEME['ui']['secondary']['light_gray']};
        margin: 20px 0;
    }}
    
    /* 按鈕樣式 - 覆蓋Streamlit預設 */
    .stDownloadButton button, .stButton button {{
        background-color: {COLOR_SCHEME['ui']['accent']['gold']} !important;
        color: {COLOR_SCHEME['ui']['secondary']['dark_gray']} !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease-in-out !important;
    }}
    
    .stDownloadButton button:hover, .stButton button:hover {{
        background-color: {COLOR_SCHEME['ui']['accent']['orange']} !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.12) !important;
        transform: translateY(-1px) !important;
    }}
    
    /* 側邊欄背景 */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_SCHEME['ui']['secondary']['white']};
    }}
    
    /* Expander 樣式 */
    .streamlit-expanderContent {{
        background-color: {COLOR_SCHEME['ui']['secondary']['light_gray']};
        border-radius: 6px;
    }}
    
    /* 複選框和滑塊 */
    .stCheckbox, .stSlider {{
        color: {COLOR_SCHEME['ui']['secondary']['dark_gray']};
    }}
    
    /* 指標容器 */
    .stMetric {{
        border-left: 4px solid {COLOR_SCHEME['ui']['primary']};
    }}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

sns.set(style="whitegrid")
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["figure.facecolor"] = COLOR_SCHEME['ui']['secondary']['white']
mpl.rcParams["axes.facecolor"] = COLOR_SCHEME['ui']['secondary']['light_gray']
mpl.rcParams["axes.edgecolor"] = COLOR_SCHEME['ui']['secondary']['light_gray']
mpl.rcParams["text.color"] = COLOR_SCHEME['ui']['secondary']['dark_gray']
mpl.rcParams["xtick.color"] = COLOR_SCHEME['ui']['secondary']['dark_gray']
mpl.rcParams["ytick.color"] = COLOR_SCHEME['ui']['secondary']['dark_gray']
mpl.rcParams["grid.color"] = COLOR_SCHEME['ui']['secondary']['light_gray']
mpl.rcParams["grid.linestyle"] = "-"
mpl.rcParams["grid.linewidth"] = 0.5


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
    # 檢查是否為空陣列或全為 NaN
    if len(s) == 0 or s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
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

with st.sidebar.expander("⚙️ 一般設定", expanded=True):
    threshold_mode = st.radio(
        "風險等級門檻模式",
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
st.sidebar.markdown(
    f"<hr style='border-color: {COLOR_SCHEME['ui']['secondary']['light_gray']};'>", unsafe_allow_html=True)
st.sidebar.markdown("### 🧭 欄位對映與方向")
st.sidebar.markdown(
    f"<div style='background-color: {COLOR_SCHEME['ui']['secondary']['light_gray']}; padding: 12px; border-radius: 6px;'>", unsafe_allow_html=True)
all_cols = df_raw.columns.tolist()

# 智能偵測日期欄位
date_col_candidates = [c for c in all_cols if any(
    keyword in str(c).lower() for keyword in ["date", "日期", "時間"])]
default_date_idx = all_cols.index(
    date_col_candidates[0]) if date_col_candidates else 0
date_col = st.sidebar.selectbox(
    "日期欄位", options=all_cols, index=default_date_idx)

# 指標候選（排除日期欄與非數值欄位）
metric_candidates = [c for c in all_cols if c != date_col]

# 智能偵測數值欄位作為預設指標
numeric_cols = []
for col in metric_candidates:
    try:
        # 嘗試轉換前幾筆資料，判斷是否為數值欄位
        test_series = pd.to_numeric(df_raw[col].head(10), errors='coerce')
        # 檢查轉換後至少有一半以上的資料是有效數字
        valid_count = test_series.notna().sum()
        if valid_count >= len(test_series) * 0.5:
            numeric_cols.append(col)
    except:
        pass

# 如果沒有找到數值欄位，顯示警告
if not numeric_cols:
    st.sidebar.warning("⚠️ 未自動偵測到數值欄位，請手動選擇。")
    default_metrics = []
else:
    default_metrics = numeric_cols

metrics = st.sidebar.multiselect(
    "指標欄位（可多選）", options=metric_candidates, default=default_metrics)

if len(metrics) == 0:
    st.error("請至少選擇一個指標欄位。")
    st.stop()

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# 每個指標的「越大越糟？」方向與權重
st.sidebar.markdown(
    f"<hr style='border-color: {COLOR_SCHEME['ui']['secondary']['light_gray']};'>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div style='background-color: {COLOR_SCHEME['ui']['accent']['gold']}20; padding: 12px; border-radius: 6px; margin-bottom: 16px;'>", unsafe_allow_html=True)
st.sidebar.markdown("**📊 指標方向與權重**")
dir_cols = {}
w_cols = {}
for m in metrics:
    cols = st.sidebar.columns([1, 1.2])
    with cols[0]:
        # 智能判斷預設方向：買入價格、賣出價格等越大越好的指標
        is_worse = not any(keyword in str(m).lower()
                           for keyword in ["買入", "賣出", "liquidity", "gap"])
        dir_cols[m] = st.checkbox(f"{m} 越大越糟？", value=is_worse)
    with cols[1]:
        w_cols[m] = st.slider(
            f"{m} 權重", min_value=0.0, max_value=1.0, value=1.0/len(metrics), step=0.01, key=f"w_{m}")

# 權重正規化
w_sum = sum(w_cols.values())
if w_sum == 0:
    weights = {m: 1 / len(metrics) for m in metrics}
else:
    weights = {m: w_cols[m] / w_sum for m in metrics}

st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.caption("（權重會自動正規化為總和=1）")

# -------------------------
# 主區塊：處理、計算、視覺化
# -------------------------
# 頁面標題 & 說明
st.markdown(f"""
<div style='border-bottom: 3px solid {COLOR_SCHEME['ui']['primary']}; padding-bottom: 16px; margin-bottom: 24px;'>
    <h1 style='margin: 0; color: {COLOR_SCHEME['ui']['primary']};'>📊 Risk Indicator (RI) Dashboard</h1>
    <p style='margin: 8px 0 0 0; color: {COLOR_SCHEME['ui']['secondary']['dark_gray']};'>上傳 Excel → 對映欄位 → 設定方向/權重 → 即時出圖與匯出</p>
</div>
""", unsafe_allow_html=True)

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
conversion_info = []
for m in metrics:
    if m not in df.columns:
        st.error(f"❌ 指標 '{m}' 不在資料欄位中！")
        st.stop()

    original_values = df[m].copy()
    df[m] = pd.to_numeric(df[m], errors="coerce")
    valid_count = df[m].notna().sum()
    total_count = len(df[m])

    conversion_info.append({
        "欄位": m,
        "有效數值": valid_count,
        "總筆數": total_count,
        "轉換率": f"{valid_count/total_count*100:.1f}%" if total_count > 0 else "0%"
    })

# 顯示轉換資訊（展開查看）
with st.expander("📊 查看資料轉換詳情"):
    st.dataframe(pd.DataFrame(conversion_info), use_container_width=True)
    st.caption("所選欄位必須至少有部分有效數值才能進行分析")

df_before = len(df)
df = df[[date_col] + metrics].dropna().sort_values(by=date_col).reset_index(drop=True)
df_after = len(df)

# 檢查是否有有效數據
if len(df) == 0:
    st.error("❌ 清理後沒有有效數據。")
    st.error(f"原始資料：{df_before} 筆 → 清理後：{df_after} 筆")
    st.error("可能原因：")
    st.error("1. 所選指標欄位包含非數值資料（如文字：「1公克」、「新台幣(TWD)」）")
    st.error("2. 日期欄位解析失敗")
    st.info("💡 建議：請在側邊欄重新選擇「僅包含純數字」的欄位（如價格、數量等）")
    st.stop()

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

df["color"] = df["vibe"].apply(get_risk_color)

# 指標貢獻度（最後一天）
latest = df.iloc[-1]
contrib = {m: latest[f"{m}_norm"] * weights[m] for m in metrics}
ser_contrib = pd.Series(contrib).sort_values(ascending=False)

# ---- 上方 KPI 區（卡片式） ----
st.markdown("### 📈 即時指標")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

with kpi_col1:
    st.markdown(f"""
    <div style='
        background-color: {COLOR_SCHEME['ui']['secondary']['white']};
        border: 2px solid {COLOR_SCHEME['ui']['primary']};
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    '>
        <div style='color: {COLOR_SCHEME['ui']['secondary']['dark_gray']}; font-size: 14px; font-weight: 600; margin-bottom: 8px;'>最新日期</div>
        <div style='color: {COLOR_SCHEME['ui']['primary']}; font-size: 24px; font-weight: 700;'>{latest[date_col].date().strftime('%Y-%m-%d')}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div style='
        background-color: {COLOR_SCHEME['ui']['secondary']['white']};
        border: 2px solid {COLOR_SCHEME['ui']['primary']};
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    '>
        <div style='color: {COLOR_SCHEME['ui']['secondary']['dark_gray']}; font-size: 14px; font-weight: 600; margin-bottom: 8px;'>Risk Heat Score</div>
        <div style='color: {COLOR_SCHEME['ui']['primary']}; font-size: 24px; font-weight: 700;'>{latest['risk_heat']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    vibe_label = latest['vibe']
    vibe_bg = get_risk_color(vibe_label)
    st.markdown(f"""
    <div style='
        background-color: {COLOR_SCHEME['ui']['secondary']['white']};
        border: 2px solid {COLOR_SCHEME['ui']['primary']};
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    '>
        <div style='color: {COLOR_SCHEME['ui']['secondary']['dark_gray']}; font-size: 14px; font-weight: 600; margin-bottom: 8px;'>當前風險等級</div>
        <div style='background-color: {vibe_bg}; color: {COLOR_SCHEME['ui']['secondary']['white']}; padding: 8px 16px; border-radius: 6px; font-size: 18px; font-weight: 700; display: inline-block;'>{vibe_label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")  # 空行分隔

# ---- 圖 1：色帶圖（卡片） ----
st.markdown("### 📊 Bank Risk Indicator（每日氛圍色帶）")
st.markdown(
    f"<div style='background-color: {COLOR_SCHEME['ui']['secondary']['white']}; border: 1px solid {COLOR_SCHEME['ui']['secondary']['light_gray']}; border-radius: 8px; padding: 16px;'>", unsafe_allow_html=True)

fig1, ax1 = plt.subplots(figsize=(12, 2.5))
ax1.bar(range(len(df)), [1] * len(df),
        color=df["color"].tolist(), edgecolor="none", width=0.95)
ax1.set_xticks(range(0, len(df), max(1, len(df)//10)))
ax1.set_xticklabels([df[date_col].iloc[i].strftime("%Y-%m-%d") if i < len(df) else ""
                     for i in range(0, len(df), max(1, len(df)//10))], rotation=45, ha="right", fontsize=9)
ax1.set_yticks([])
ax1.set_ylabel("")
ax1.set_title("每日風險等級色帶（Calm → Neutral → Alert → Critical）",
              fontsize=12, fontweight="bold", pad=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
plt.tight_layout()
st.pyplot(fig1, use_container_width=True)
img1 = to_bytes_png(fig1)
plt.close(fig1)

st.markdown("</div>", unsafe_allow_html=True)

# ---- 圖 2：熱度趨勢（卡片） ----
st.markdown("### 📈 Risk Heat 趨勢（含風險等級區間）")
st.markdown(
    f"<div style='background-color: {COLOR_SCHEME['ui']['secondary']['white']}; border: 1px solid {COLOR_SCHEME['ui']['secondary']['light_gray']}; border-radius: 8px; padding: 16px;'>", unsafe_allow_html=True)

fig2, ax2 = plt.subplots(figsize=(12, 5))
# 背景區間
if threshold_mode.startswith("歷史分位數"):
    ax2.axhspan(0.0, df["risk_heat"].min(),
                facecolor=COLOR_SCHEME['risk']['calm'], alpha=0.08)
    ax2.axhspan(df["risk_heat"].min(), df["risk_heat"].quantile(
        0.25), facecolor=COLOR_SCHEME['risk']['calm'], alpha=0.12, label="Calm")
    ax2.axhspan(df["risk_heat"].quantile(0.25), df["risk_heat"].quantile(
        0.5), facecolor=COLOR_SCHEME['risk']['neutral'], alpha=0.12, label="Neutral")
    ax2.axhspan(df["risk_heat"].quantile(0.5), df["risk_heat"].quantile(
        0.75), facecolor=COLOR_SCHEME['risk']['alert'], alpha=0.10, label="Alert")
    ax2.axhspan(df["risk_heat"].quantile(0.75), 1.0,
                facecolor=COLOR_SCHEME['risk']['critical'], alpha=0.10, label="Critical")
else:
    ax2.axhspan(
        0.00, 0.25, facecolor=COLOR_SCHEME['risk']['calm'], alpha=0.12, label="Calm")
    ax2.axhspan(
        0.25, 0.50, facecolor=COLOR_SCHEME['risk']['neutral'], alpha=0.12, label="Neutral")
    ax2.axhspan(
        0.50, 0.75, facecolor=COLOR_SCHEME['risk']['alert'], alpha=0.10, label="Alert")
    ax2.axhspan(
        0.75, 1.00, facecolor=COLOR_SCHEME['risk']['critical'], alpha=0.10, label="Critical")

ax2.plot(range(len(df)), df["risk_heat"].values, color=COLOR_SCHEME['ui']['primary'],
         linewidth=2.5, marker="o", markersize=5, markeredgecolor=COLOR_SCHEME['ui']['secondary']['white'],
         markeredgewidth=1.5, zorder=3)
ax2.set_ylim(0, 1)
ax2.set_xticks(range(0, len(df), max(1, len(df)//10)))
ax2.set_xticklabels([df[date_col].iloc[i].strftime("%Y-%m-%d") if i < len(df) else ""
                     for i in range(0, len(df), max(1, len(df)//10))], rotation=45, ha="right", fontsize=9)
ax2.set_ylabel("Risk Heat（0~1）", fontweight="bold", fontsize=11)
ax2.set_title("風險熱度趨勢分析", fontsize=12, fontweight="bold", pad=12)
ax2.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
ax2.legend(loc="upper left", framealpha=0.95, fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
img2 = to_bytes_png(fig2)
plt.close(fig2)

st.markdown("</div>", unsafe_allow_html=True)

# ---- 圖 3：當日貢獻度（卡片） ----
st.markdown("### 📊 當日風險熱度貢獻度分析")
st.markdown(
    f"<div style='background-color: {COLOR_SCHEME['ui']['secondary']['white']}; border: 1px solid {COLOR_SCHEME['ui']['secondary']['light_gray']}; border-radius: 8px; padding: 16px;'>", unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(8, 4))
bars = ax3.barh(ser_contrib.index, ser_contrib.values,
                color=COLOR_SCHEME['ui']['primary'], edgecolor="none", height=0.6)
# 添加數值標籤
for i, (idx, val) in enumerate(ser_contrib.items()):
    ax3.text(val + 0.01, i, f"{val:.3f}",
             va="center", fontsize=10, fontweight="bold")
ax3.set_xlabel("貢獻度", fontweight="bold", fontsize=11)
ax3.set_ylabel("")
ax3.set_title(f"當日（{latest[date_col].date()}）風險熱度貢獻度排名",
              fontsize=12, fontweight="bold", pad=12)
ax3.set_xlim(0, max(ser_contrib.values) * 1.15)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
plt.tight_layout()
st.pyplot(fig3, use_container_width=False)
img3 = to_bytes_png(fig3)
plt.close(fig3)

st.markdown("</div>", unsafe_allow_html=True)

# ---- 匯出操作區 ----
st.markdown("### 💾 下載結果")
st.markdown(
    f"<div style='background-color: {COLOR_SCHEME['ui']['secondary']['light_gray']}; padding: 16px; border-radius: 8px; margin-bottom: 16px;'>", unsafe_allow_html=True)

col_d1, col_d2, col_d3, col_d4 = st.columns(4)
with col_d1:
    # 處理後資料
    out_csv = df.copy()
    out_csv[date_col] = out_csv[date_col].dt.strftime("%Y-%m-%d")
    st.download_button(
        label="⬇️ CSV 分析結果",
        data=out_csv.to_csv(index=False).encode("utf-8-sig"),
        file_name="risk_result.csv",
        mime="text/csv"
    )
with col_d2:
    st.download_button(
        label="⬇️ 色帶圖",
        data=img1,
        file_name="risk_band.png",
        mime="image/png"
    )
with col_d3:
    st.download_button(
        label="⬇️ 趨勢圖",
        data=img2,
        file_name="risk_heat_trend.png",
        mime="image/png"
    )
with col_d4:
    st.download_button(
        label="⬇️ 貢獻度圖",
        data=img3,
        file_name="risk_heat_contrib.png",
        mime="image/png"
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---- 模板下載 & 原始資料檢視 ----
col_tmpl, col_preview = st.columns([1, 1])

with col_tmpl:
    st.markdown("### 📋 下載 Excel 模板")
    st.markdown(
        f"<div style='background-color: {COLOR_SCHEME['ui']['secondary']['white']}; border: 1px solid {COLOR_SCHEME['ui']['secondary']['light_gray']}; border-radius: 8px; padding: 16px;'>", unsafe_allow_html=True)
    df_tmpl = build_template_df(30)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_tmpl.to_excel(writer, index=False, sheet_name="risk_metrics")
    buf.seek(0)
    st.download_button(
        label="⬇️ 範例模板.xlsx",
        data=buf.getvalue(),
        file_name="risk_metrics_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_preview:
    st.markdown("### 🔍 原始資料預覽")
    st.markdown(
        f"<div style='background-color: {COLOR_SCHEME['ui']['secondary']['white']}; border: 1px solid {COLOR_SCHEME['ui']['secondary']['light_gray']}; border-radius: 8px; padding: 16px;'>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- 處理後資料詳細檢視 ----
if show_table:
    st.markdown("---")
    st.markdown("### 📊 處理後完整資料表")
    st.markdown(
        f"<div style='background-color: {COLOR_SCHEME['ui']['secondary']['white']}; border: 1px solid {COLOR_SCHEME['ui']['secondary']['light_gray']}; border-radius: 8px; padding: 16px;'>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=400)
    st.markdown("</div>", unsafe_allow_html=True)
