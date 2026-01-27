# Risk Vibe Indicator (RVI) Dashboard

Streamlit 建立的金融風險「氛圍」儀表板。上傳 Excel 後設定日期欄位、指標方向與權重，計算風險熱度並產出 Vibe 色帶、趨勢與貢獻度圖，可下載處理後 CSV 與圖片。

## 專案概要

- 架構：單一 Streamlit 應用 [rvi_app.py](rvi_app.py)。
- 流程：上傳/讀取 Excel → 日期解析 → 指標選取 → 方向一致化/正規化 → 權重加總得到 risk heat → 依門檻映射 Vibe → 顯示色帶、趨勢、貢獻度並提供下載。
- 主要函式：
  - `try_set_chinese_font()`：載入專案或系統中文字型，解決中文方框。
  - `universal_date_parser()`：強韌日期解析，支援常見格式與 Excel serial。
  - `apply_direction()`：將「越大越好」指標反向，使越大越糟後再正規化。
  - `normalize_minmax()`：0-1 最小最大正規化，常數列回傳 0。
  - `heat_to_vibe()`/`vibe_color()`：將熱度分數映射為 Vibe 標籤與色碼。
  - `load_excel()`：快取式 Excel 載入（openpyxl）。
- 視覺化：色帶圖、風險熱度趨勢（含門檻區間）、最新日指標貢獻度長條。
- 匯出：處理後 CSV、三張 PNG 圖、Excel 範例模板下載。

## 需求安裝

- Python 3.12（相容版本亦可）
- 依照 [requirements.txt](requirements.txt) 安裝：
  ```sh
  pip install -r requirements.txt
  ```

## 執行方式

1. 啟動服務：
   ```sh
   streamlit run rvi_app.py
   ```
2. 瀏覽器開啟 http://localhost:8501（預設），依介面指示上傳 Excel 或使用示範資料。

## 使用流程

- 上傳 Excel（可選工作表；未上傳則使用隨機模板資料）。
- 指定日期欄位與指標欄位；每個指標設定「越大越糟？」與權重（自動正規化為總和 1）。
- 選擇 Vibe 門檻模式：
  - 固定分段：0.25/0.5/0.75。
  - 歷史分位數：25%/50%/75%。
- 觀察指標值經方向一致化與 0-1 正規化後，以
  $risk\_heat = \sum_i w_i \cdot norm_i$
  得到熱度，轉換為 Vibe（Calm/Neutral/Alert/Critical）。
- 下載處理後 CSV、PNG 圖檔，或下載 Excel 範例模板。

## Excel 資料要求

- 必備：日期欄位（支援 YYYYMMDD、YYYY-MM-DD、YYYY/MM/DD、Excel serial、datetime）。
- 指標欄位：至少一個數值指標；缺值列會被移除。
- 權重全為 0 時，系統自動平均分配權重。

## 字型與中文顯示

- 若出現中文方框，可將 `NotoSansTC-Regular.ttf` 放在專案根目錄，或於側邊欄上傳字型檔（ttf/otf/ttc）。

## 常見問題

- 日期無法解析：介面會警告並移除無效日期列，可檢查原始資料格式。
- 權重設定：滑桿值會再正規化，確保總和為 1；可勾選/取消指標調整影響。
- 匯出路徑：下載鍵於頁面下方「下載結果」區塊，含 CSV 與三張圖。
