"""Candlecast — premium desktop app for candlestick forecasting.

Dark fintech theme, KV-cache engine, free market data (crypto via ccxt,
equities/forex via Yahoo). UI languages: Turkish / English.
"""
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(APP_DIR))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from candlecast import CandlePredictor, fetch_ccxt, fetch_yfinance, next_timestamps

APP_NAME = "Candlecast"
CONFIG_PATH = APP_DIR / "candlecast.config.json"

BG = "#0d1017"
PANEL = "#151a23"
PANEL2 = "#1b2230"
BORDER = "#2a3344"
TEXT = "#e8ecf4"
MUTED = "#8d97ad"
ACCENT_DIM = "#c98a16"
UP = "#22c55e"
DOWN = "#ef4444"
WARN = "#fbbf24"
CHART_BG = "#12161f"
HIST_LINE = "#9aa4b8"

FONT = "Segoe UI"
MONO = "Consolas"

CRYPTO_SOURCES = ["binance", "bybit", "okx", "kraken", "coinbase", "kucoin",
                  "gate", "bitget", "mexc"]
CRYPTO_TFS = ["5m", "15m", "1h", "4h", "1d"]
YF_TFS = ["1h", "1d", "1wk"]
TF_FREQ = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h",
           "1d": "1d", "1wk": "1wk"}

CRYPTO_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "TRX/USDT", "LTC/USDT",
    "POL/USDT", "ATOM/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT",
    "TON/USDT", "SHIB/USDT", "UNI/USDT", "ETC/USDT", "FIL/USDT", "APT/USDT",
]
YF_SYMBOLS = [
    "THYAO.IS", "GARAN.IS", "ASELS.IS", "TUPRS.IS", "BIMAS.IS", "KCHOL.IS",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "NFLX",
    "EURUSD=X", "USDTRY=X", "GBPUSD=X", "^GSPC", "^XU100", "GC=F", "CL=F",
    "BTC-USD", "ETH-USD",
]
QUICK_CRYPTO = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                "DOGE/USDT", "AVAX/USDT", "LINK/USDT"]
QUICK_YF = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "TUPRS.IS", "AAPL", "TSLA",
            "EURUSD=X", "^XU100"]

CHIP_COLORS = ["#22c55e", "#38bdf8", "#f472b6", "#a78bfa", "#fbbf24",
               "#fb923c", "#2dd4bf", "#e879f9"]
FIELD_COLORS = ["#38bdf8", "#fbbf24", "#a78bfa", "#34d399", "#60a5fa",
                "#f472b6", "#fb923c"]

ACCENTS = {
    "amber": "#ffb020",
    "teal": "#2dd4bf",
    "violet": "#a78bfa",
    "rose": "#fb7185",
    "blue": "#60a5fa",
}

MODEL_LABELS = {
    "Light   ·   4M params   ·   2048 ctx": "light",
    "Balanced   ·   25M params   ·   512 ctx": "balanced",
    "Large   ·   102M params   ·   512 ctx": "large",
}
LABEL_BY_KEY = {v: k for k, v in MODEL_LABELS.items()}
MAX_LOOKBACK = {"light": 1024, "balanced": 512, "large": 512}

LANG_LABEL = {"Türkçe": "tr", "English": "en"}
LANG_CODE = {v: k for k, v in LANG_LABEL.items()}

STATE_KEYS = {"ready": "state_ready", "working": "state_working",
              "done": "state_done", "error": "state_error"}
STATE_STYLE = {"ready": (UP, "#07130b"), "working": (WARN, "#171103"),
               "done": (UP, "#07130b"), "error": (DOWN, "#190608")}

STR = {
    "tr": {
        "tagline": "Finansal Mum Tahmin Süiti",
        "lang_cap": "DİL",
        "btn_guide": "ⓘ  Kılavuz",
        "cap_market": "BAZ",
        "cap_symbol": "SEMBOL",
        "cap_interval": "ARALIK",
        "cap_model": "MODEL",
        "cap_context": "BAĞLAM",
        "cap_horizon": "UFUK",
        "cap_paths": "YOLLAR",
        "cap_quick": "HIZLI SEMBOLLER",
        "btn_run": "▶  Tahmini Çalıştır",
        "kpi_last": "SON KAPANIŞ",
        "kpi_hor": "UFUK KAPANIŞI",
        "kpi_chg": "TAHMİN DEĞİŞİMİ",
        "kpi_band": "q10 – q90 (KAPANIŞ)",
        "kpi_time": "HESAP SÜRESİ",
        "col_time": "ZAMAN",
        "col_open": "AÇIK",
        "col_high": "YÜKSEK",
        "col_low": "DÜŞÜK",
        "col_close": "KAPANIŞ",
        "col_chg": "Δ%",
        "sec_forecast": "TAHMİN",
        "sec_activity": "AKTİVİTE",
        "btn_png": "⤓  PNG Aktar",
        "btn_csv": "⤓  CSV Aktar",
        "t_line": "Çizgi",
        "t_area": "Alan",
        "t_candle": "Mum",
        "t_log": "Log",
        "t_grid": "Izgara",
        "chart_btn": "Grafik: {t}",
        "state_ready": "HAZIR",
        "state_working": "ÇALIŞIYOR",
        "state_done": "BİTTİ",
        "state_error": "HATA",
        "placeholder": "Tahmini çalıştırmak için grafik burada görünecek",
        "ready": "{app} hazır. Piyasayı seçin ve «Tahmini Çalıştır»a basın.",
        "working": "Çalışıyor…",
        "status_done": "Bitti · {sym} {tf} · {s:.1f}s",
        "status_error": "Hata — aktivite günlüğüne bakın",
        "data_note": "Ücretsiz veri · ccxt + Yahoo Finance",
        "warn_symbol": "Sembol girin.",
        "err_title": APP_NAME,
        "acc_cap": "VURGU RENGİ",
        "acc_amber": "Kehribar",
        "acc_teal": "Turkuaz",
        "acc_violet": "Mor",
        "acc_rose": "Gül",
        "acc_blue": "Mavi",
        "log_fetching": "{src} / {sym} / {tf} verisi alınıyor...",
        "log_candles": "{n} mum alındı · son kapanış {c}",
        "log_loading": "«{m}» modeli yükleniyor (ilk seferde ağırlıklar iner)...",
        "log_forecasting": "{n} mum {s} örnekleme yolu ile tahmin ediliyor...",
        "log_completed": "{d:.1f}sn içinde tamamlandı",
        "log_chart_exported": "Grafik dışa aktarıldı → {p}",
        "log_csv_exported": "CSV dışa aktarıldı → {p}",
        "log_band": "{t} için kapanış bandı: {lo} … {hi}",
        "hist_legend": "Geçmiş",
        "fcst_legend": "Tahmin",
        "band_legend": "q10–q90",
        "chart_title": "{sym}  ·  {tf}  ·  {app}",
    },
    "en": {
        "tagline": "Candlestick Forecasting Suite",
        "lang_cap": "LANGUAGE",
        "btn_guide": "ⓘ  Guide",
        "cap_market": "MARKET VENUE",
        "cap_symbol": "SYMBOL",
        "cap_interval": "INTERVAL",
        "cap_model": "MODEL",
        "cap_context": "CONTEXT",
        "cap_horizon": "HORIZON",
        "cap_paths": "PATHS",
        "cap_quick": "QUICK SYMBOLS",
        "btn_run": "▶  Run Forecast",
        "kpi_last": "LAST CLOSE",
        "kpi_hor": "HORIZON CLOSE",
        "kpi_chg": "PROJECTED Δ",
        "kpi_band": "q10 – q90 (CLOSE)",
        "kpi_time": "COMPUTE TIME",
        "col_time": "TIME",
        "col_open": "OPEN",
        "col_high": "HIGH",
        "col_low": "LOW",
        "col_close": "CLOSE",
        "col_chg": "Δ%",
        "sec_forecast": "FORECAST",
        "sec_activity": "ACTIVITY",
        "btn_png": "⤓  Export PNG",
        "btn_csv": "⤓  Export CSV",
        "t_line": "Line",
        "t_area": "Area",
        "t_candle": "Candlestick",
        "t_log": "Log",
        "t_grid": "Grid",
        "chart_btn": "Chart: {t}",
        "state_ready": "READY",
        "state_working": "WORKING",
        "state_done": "DONE",
        "state_error": "ERROR",
        "placeholder": "Run a forecast to display the chart",
        "ready": "{app} ready. Choose a market and press «Run Forecast».",
        "working": "Working…",
        "status_done": "Done · {sym} {tf} · {s:.1f}s",
        "status_error": "Error — see activity log",
        "data_note": "Free data · ccxt + Yahoo Finance",
        "warn_symbol": "Enter a symbol.",
        "err_title": APP_NAME,
        "acc_cap": "ACCENT COLOR",
        "acc_amber": "Amber",
        "acc_teal": "Teal",
        "acc_violet": "Violet",
        "acc_rose": "Rose",
        "acc_blue": "Blue",
        "log_fetching": "Fetching {src} / {sym} / {tf} ...",
        "log_candles": "{n} candles received · last close {c}",
        "log_loading": "Loading model «{m}» (first run downloads weights)...",
        "log_forecasting": "Forecasting {n} candles with {s} sample paths...",
        "log_completed": "Completed in {d:.1f}s",
        "log_chart_exported": "Chart exported → {p}",
        "log_csv_exported": "CSV exported → {p}",
        "log_band": "Close band for {t}: {lo} … {hi}",
        "hist_legend": "History",
        "fcst_legend": "Forecast",
        "band_legend": "q10–q90",
        "chart_title": "{sym}  ·  {tf}  ·  {app}",
    },
}

GUIDE_LANGS = ["Türkçe", "English", "中文", "Русский", "Español", "العربية"]
GUIDE_LANG_CODE = {"Türkçe": "tr", "English": "en", "中文": "zh",
                   "Русский": "ru", "Español": "es", "العربية": "ar"}
GUIDE_CODE_LANG = {v: k for k, v in GUIDE_LANG_CODE.items()}

GUIDE_STR = {
    "tr": {
        "title": "Candlecast Kılavuzu",
        "body": """Candlecast, finansal mum verileri üzerinde tahmin yapan ücretsiz bir masaüstü uygulamasıdır.

NASIL KULLANILIR
1. BAZ: kripto için bir ccxt borsası (binance, bybit, okx…); hisse/forex/endeks için «yahoo».
2. SEMBOL: kripto örn. BTC/USDT; yahoo örn. THYAO.IS (BIST), AAPL, EURUSD=X. Hızlı sembol çipleri tek tıkla seçim sunar.
3. ARALIK: mum aralığı (kripto: 5m…1d; yahoo: 1h/1d/1wk).
4. MODEL: Light (hızlı), Balanced (önerilen), Large (daha hassas, yavaş).
5. BAĞLAM: modelin gördüğü son mum sayısı. UFUK: tahmin edilecek mum sayısı. YOLLAR: belirsizlik için örneklem sayısı.
6. «TAHMİNİ ÇALIŞTIR» (veya Enter). İlk çalıştırmada model ağırlıkları (~100 MB) Hugging Face'ten iner; sonrası önbellekten gelir.

GRAFİK AYARI (grafik çubuğundaki «Grafik» butonu)
• Çizgi — kapanış serisi
• Alan — kapanış + dolgu
• Mum — OHLC mumları; yeşil yükseliş, kırmızı düşüş. Tahmin bölümü kesik çizgiden sonra mum olarak da çizilir.
• Log ve Izgara düğmeleri eksen ölçeğini ve ızgarayı anında değiştirir.

TAHMİN TABLOSU
Her satır ortalama (mean) yoldur. q10–q90 bandı çoklu örnekleme yollarından hesaplanır: dar bant = kararlı tahmin, geniş bant = yüksek belirsizlik. Δ%, kapanışın son fiyata göre yüzdesel değişimidir. Satıra tıklayınca bandın detayı günlüğe düşer.

VERİ KAYNAKLARI
Kripto: ccxt (100+ borsa). Hisse/forex/endeks: Yahoo Finance. Veriler gecikmeli olabilir. Bu araç yatırım tavsiyesi değildir.

ARAYÜZ
• Üstteki DİL kutusu arayüzü Türkçe/İngilizce yapar; bu pencere kendi dil seçicisine sahiptir.
• ⚙ menüsünden vurgu rengini değiştirebilir, tüm tercihler (dil, renk, sembol, ayarlar) kapanırken kaydedilir.

TEKNİK
Motor, MIT lisanslı temel mimari üzerine KV-cache hızlandırması uygular: CPU'da orijinal döngüye göre ~3,8–5,4× hız, bit-eşit doğruluk.
""",
    },
    "en": {
        "title": "Candlecast Guide",
        "body": """Candlecast is a free desktop app for forecasting financial candlestick data.

HOW TO USE
1. MARKET VENUE: a ccxt crypto exchange (binance, bybit, okx…), or «yahoo» for stocks/forex/indices.
2. SYMBOL: crypto e.g. BTC/USDT; yahoo e.g. THYAO.IS (Borsa Istanbul), AAPL, EURUSD=X. Quick-symbol chips select with one click.
3. INTERVAL: candle timeframe (crypto: 5m…1d; yahoo: 1h/1d/1wk).
4. MODEL: Light (fast), Balanced (recommended), Large (finer, slower).
5. CONTEXT: how many recent candles the model sees. HORIZON: how many candles to forecast. PATHS: sample count for uncertainty.
6. Press «RUN FORECAST» (or Enter). The first run downloads model weights (~100 MB) from Hugging Face; later runs use the cache.

CHART SETTING (the «Chart» button above the chart)
• Line — close series
• Area — filled close series
• Candlestick — OHLC candles; green = up, red = down. The forecast section after the dashed divider is drawn as candles too.
• The Log and Grid buttons instantly change the axis scale and gridlines.

FORECAST TABLE
Every row is the mean path. The q10–q90 band comes from multiple sample paths: a narrow band = confident forecast, a wide band = high uncertainty. Δ% is that close's change vs the last price. Click a row to log the band detail.

DATA SOURCES
Crypto: ccxt (100+ exchanges). Stocks/forex/indices: Yahoo Finance. Data may be delayed. Not investment advice.

INTERFACE
• The LANGUAGE box at the top switches the UI between Turkish and English; this window has its own language selector.
• Use ⚙ to change the accent color. All preferences (language, color, symbol, settings) are saved when the app closes.

TECHNIQUE
The engine applies a KV-cache speedup on top of the upstream MIT-licensed architecture: ~3.8–5.4× faster than the original loop on CPU, bit-exact outputs.
""",
    },
    "zh": {
        "title": "Candlecast 使用指南",
        "body": """Candlecast 是一款免费的金融 K 线预测桌面应用。

使用方法
1. 市场来源: 加密货币选择 ccxt 交易所 (binance, bybit, okx…); 股票/外汇/指数选择「yahoo」。
2. 交易对: 加密货币如 BTC/USDT; yahoo 如 THYAO.IS (伊斯坦布尔交易所), AAPL, EURUSD=X。快捷交易对按钮一键选择。
3. 时间周期: K 线周期 (加密: 5m…1d; yahoo: 1h/1d/1wk)。
4. 模型: Light (快速), Balanced (推荐), Large (更精细、较慢)。
5. 上下文: 模型可见的历史 K 线数量。预测长度: 需要预测的 K 线数量。路径数: 表示不确定性的采样数量。
6. 点击「运行预测」(或按 Enter)。首次运行会从 Hugging Face 下载模型权重 (~100 MB), 之后使用缓存。

图表设置 (图表上方的「图表」按钮)
• 折线 — 收盘价序列
• 面积 — 带填充的收盘价
• K 线 — OHLC 蜡烛; 绿色为涨, 红色为跌。虚线之后的预测部分同样以蜡烛绘制。
• 「Log」与「Izgara/Grid」按钮可即时切换坐标轴比例与网格。

预测表格
每行为平均 (mean) 路径。q10–q90 区间由多条采样路径计算: 区间越窄预测越确定, 越宽不确定性越高。Δ% 为该收盘价相对最新价格的变化。点击某行可在日志中查看区间详情。

数据来源
加密货币: ccxt (100+ 交易所)。股票/外汇/指数: Yahoo Finance。数据可能延迟。本工具不构成投资建议。

界面
• 顶部「DİL/LANGUAGE」框切换土耳其语/英语; 本窗口拥有独立的语言选择。
• 通过 ⚙ 更改强调色; 所有偏好 (语言、颜色、交易对、设置) 在退出时保存。

技术
引擎在 MIT 许可的上游架构 之上应用 KV-cache 加速: 在 CPU 上比原始循环快约 3.8–5.4 倍, 输出逐位一致。
""",
    },
    "ru": {
        "title": "Руководство Candlecast",
        "body": """Candlecast — бесплатное настольное приложение для прогнозирования финансовых свечей.

КАК ПОЛЬЗОВАТЬСЯ
1. Площадка: для криптовалюты — биржа ccxt (binance, bybit, okx…); для акций/форекса/индексов — «yahoo».
2. Символ: крипто — напр. BTC/USDT; yahoo — напр. THYAO.IS (Стамбульская биржа), AAPL, EURUSD=X. Быстрые кнопки символов выбирают пару в один клик.
3. Таймфрейм: период свечей (крипто: 5m…1d; yahoo: 1h/1d/1wk).
4. Модель: Light (быстрая), Balanced (рекомендуется), Large (точнее, медленнее).
5. Контекст: сколько последних свечей видит модель. Горизонт: сколько свечей предсказать. Пути: число выборок для неопределённости.
6. Нажмите «Запустить прогноз» (или Enter). При первом запуске веса (~100 МБ) загружаются с Hugging Face; далее используется кэш.

НАСТРОЙКА ГРАФИКА (кнопка «График» над графиком)
• Линия — ряд цен закрытия
• Область — заполненный ряд
• Свечи — OHLC-свечи; зелёная — рост, красная — падение. Прогнозный участок после пунктирной линии тоже рисуется свечами.
• Кнопки Log и Сетка мгновенно меняют масштаб оси и сетку.

ТАБЛИЦА ПРОГНОЗА
Каждая строка — средний (mean) путь. Полоса q10–q90 строится по нескольким выборочным путям: узкая полоса — уверенный прогноз, широкая — высокая неопределённость. Δ% — изменение закрытия относительно последней цены. Щелчок по строке выводит детали полосы в журнал.

ИСТОЧНИКИ ДАННЫХ
Криптовалюта: ccxt (100+ бирж). Акции/форекс/индексы: Yahoo Finance. Данные могут быть с задержкой. Не является инвестиционной рекомендацией.

ИНТЕРФЕЙС
• Поле DİL/LANGUAGE вверху переключает интерфейс на турецкий/английский; в этом окне — свой выбор языка.
• Через ⚙ меняется акцентный цвет; все настройки (язык, цвет, символ, параметры) сохраняются при закрытии.

ТЕХНИКА
Движок использует ускорение KV-cache поверх базовой архитектуры (лицензия MIT): примерно в 3,8–5,4 раза быстрее исходного цикла на CPU, побитово точный результат.
""",
    },
    "es": {
        "title": "Guía de Candlecast",
        "body": """Candlecast es una aplicación de escritorio gratuita para predecir velas financieras.

CÓMO USARLA
1. Mercado: para cripto, un exchange ccxt (binance, bybit, okx…); para acciones/forex/índices, «yahoo».
2. Símbolo: cripto p. ej. BTC/USDT; yahoo p. ej. THYAO.IS (Bolsa de Estambul), AAPL, EURUSD=X. Los botones de símbolos rápidos seleccionan con un clic.
3. Intervalo: marco de la vela (cripto: 5m…1d; yahoo: 1h/1d/1wk).
4. Modelo: Light (rápido), Balanced (recomendado), Large (más fino, más lento).
5. Contexto: cuántas velas recientes ve el modelo. Horizonte: cuántas velas predecir. Rutas: número de muestras para la incertidumbre.
6. Pulsa «EJECUTAR PRONÓSTICO» (o Enter). La primera ejecución descarga los pesos (~100 MB) de Hugging Face; después usa caché.

AJUSTE DEL GRÁFICO (botón «Gráfico» sobre el gráfico)
• Línea — serie de cierres
• Área — serie de cierres rellena
• Velas — velas OHLC; verde = sube, roja = baja. La sección de pronóstico tras la línea discontinua también se dibuja como velas.
• Los botones Log y Cuadrícula cambian la escala del eje y la rejilla al instante.

TABLA DE PRONÓSTICO
Cada fila es la ruta media (mean). La banda q10–q90 sale de varias rutas de muestra: banda estrecha = pronóstico seguro, banda ancha = alta incertidumbre. Δ% es el cambio de ese cierre respecto al último precio. Haz clic en una fila para ver el detalle en el registro.

FUENTES DE DATOS
Cripto: ccxt (100+ exchanges). Acciones/forex/índices: Yahoo Finance. Los datos pueden tener retardo. No es asesoramiento de inversión.

INTERFAZ
• El cuadro DİL/LANGUAGE de arriba cambia la interfaz entre turco e inglés; esta ventana tiene su propio selector de idioma.
• Usa ⚙ para cambiar el color de acento; todas las preferencias (idioma, color, símbolo, ajustes) se guardan al cerrar la app.

TÉCNICA
El motor aplica una aceleración KV-cache sobre la arquitectura base (licencia MIT): unas 3,8–5,4 veces más rápido que el bucle original en CPU, con salida idéntica bit a bit.
""",
    },
    "ar": {
        "title": "دليل Candlecast",
        "body": """Candlecast تطبيق سطح مكتب مجاني للتنبؤ ببيانات الشموع المالية.

كيفية الاستخدام
1. المصدر: للعملات الرقمية اختر منصة ccxt (binance, bybit, okx…)؛ وللأسهم/العملات/المؤشرات اختر «yahoo».
2. الرمز: للعملات مثل BTC/USDT؛ وليyahoo مثل THYAO.IS (بورصة إسطنبول)، AAPL، EURUSD=X. تتيح أزرار الرموز السريعة الاختيار بنقرة واحدة.
3. الفاترة: إطار زمني للشموع (العملات: 5m…1d؛ yahoo: 1h/1d/1wk).
4. النموذج: Light (سريع)، Balanced (موصى به)، Large (أدق وأبطأ).
5. السياق: عدد الشموع الأخيرة التي يراها النموذج. الأفق: عدد الشموع المتوقعة. المسارات: عدد العينات لقياس عدم اليقين.
6. اضغط «تشغيل التنبؤ» (أو Enter). في أول تشغيل تُحمَّل أوزان النموذج (~100 ميجابايت) من Hugging Face، ثم تُستخدم الذاكرة المؤقتة.

إعداد الرسم (زر «Grafik» فوق الرسم)
• خط — سلسلة أسعار الإغلاق
• منطقة — سلسلة الإغلاق مع تعبئة
• شموع — شموع OHLC؛ الأخضر صعود، الأحمر هبوط. قسم التنبؤ بعد الخط المتقطع يُرسم أيضًا كشموع.
• زرا Log و Izgara يغيّران مقياس المحور والشبكة فورًا.

جدول التنبؤ
كل صف هو المسار المتوسط (mean). نطاق q10–q90 يُحسب من عدة مسارات عشوائية: نطاق ضيق = تنبؤ واثق، واسع = عدم يقين مرتفع. Δ% هو تغيير الإغلاق مقارنة بآخر سعر. انقر الصف لعرض تفاصيل النطاق في السجل.

مصادر البيانات
العملات: ccxt (100+ منصة). الأسهم/العملات/المؤشرات: Yahoo Finance. قد تتأخر البيانات. ليست نصيحة استثمارية.

الواجهة
• مربع DİL/LANGUAGE في الأعلى يبدّل الواجهة بين التركية والإنجليزية؛ ونافذة هذه تضم محدد لغة خاصًا بها.
• استخدم ⚙ لتغيير لون التمييز؛ تُحفظ جميع التفضيلات (اللغة، اللون، الرمز، الإعدادات) عند إغلاق التطبيق.

تقنية
يطبق المحرك تسريع KV-cache على البنية الأساسية (رخصة MIT): أسرع بنحو 3.8–5.4 مرة من الحلقة الأصلية على CPU، مع ناتج مطابق تمامًا.
""",
    },
}

_pred_cache: dict = {}


def get_predictor(key: str) -> CandlePredictor:
    if key not in _pred_cache:
        _pred_cache[key] = CandlePredictor.from_hub(key, project=True, continuity=0.3)
    return _pred_cache[key]


def _load_cfg() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def do_forecast(source, symbol, tf, model_key, lookback, pred_len, samples, log, t):
    t0 = time.time()
    log(t("log_fetching", src=source, sym=symbol, tf=tf))
    if source == "yahoo":
        period = "2y" if tf in ("1d", "1wk") else "60d"
        df = fetch_yfinance(symbol, period=period, interval=tf)
    else:
        df = fetch_ccxt(source, symbol, tf, limit=min(max(lookback + pred_len + 50, 300), 1000))
    log(t("log_candles", n=len(df), c=f"{df['close'].iloc[-1]:,.4f}"))

    hist = df.iloc[-lookback:]
    y_ts = next_timestamps(hist.index[-1], pred_len, TF_FREQ[tf])
    log(t("log_loading", m=model_key))
    pred = get_predictor(model_key)
    log(t("log_forecasting", n=pred_len, s=samples))
    out = pred.predict(hist, hist.index, y_ts, pred_len,
                       T=1.0, top_p=0.9, sample_count=samples, return_samples=True)
    dt = time.time() - t0
    log(t("log_completed", d=dt))
    return {"hist": hist, "out": out, "symbol": symbol, "tf": tf,
            "model": model_key, "secs": dt}


class App(tk.Tk):
    def __init__(self):
        self.cfg = _load_cfg()
        super().__init__()
        self.lang = self.cfg.get("lang", "tr")
        self.accent_name = self.cfg.get("accent", "amber")
        if self.accent_name not in ACCENTS:
            self.accent_name = "amber"
        self.accent = ACCENTS[self.accent_name]
        self.chart_type = self.cfg.get("chart_type", "candle")
        if self.chart_type not in ("line", "area", "candle"):
            self.chart_type = "candle"
        self.log_scale = bool(self.cfg.get("log_scale", False))
        self.grid_on = bool(self.cfg.get("grid", True))
        self.result = None
        self._busy = False
        self._state = "ready"
        self.q: queue.Queue = queue.Queue()
        self._texts = []
        self.style = None
        self.guide_win = None
        self.guide_box = None
        self.guide_combo = None
        self.guide_lang = self.lang if self.lang in GUIDE_STR else "tr"
        self.guide_lang_pinned = False
        self._style()
        self._build()
        self._apply_cfg_values()
        self._set_lang(self.lang)
        self._apply_accent(self.accent)
        self._rebuild_chips()
        self._set_state("ready", update_status=False)
        self.after(120, self._poll)
        self._set_chart_placeholder()
        self.log(self.t("ready", app=APP_NAME))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def t(self, key, **kw):
        return STR[self.lang][key].format(**kw)

    def _tv(self, key, fg=MUTED, font=(FONT, 8, "bold")):
        var = tk.StringVar(value=STR[self.lang].get(key, ""))
        self._texts.append((var, key))
        return var

    # ---------------------------------------------------------------- theme
    def _style(self):
        s = ttk.Style(self)
        self.style = s
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, font=(FONT, 10),
                    fieldbackground=PANEL2, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, troughcolor=PANEL2,
                    selectbackground=self.accent, selectforeground="#10131a",
                    insertcolor=TEXT)
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        s.configure("TButton", background=PANEL2, foreground=TEXT, font=(FONT, 10),
                    padding=(14, 7), borderwidth=1, relief="flat",
                    focuscolor=PANEL2)
        s.map("TButton", background=[("active", BORDER), ("disabled", PANEL)],
              foreground=[("disabled", MUTED)])
        s.configure("Accent.TButton", background=self.accent, foreground="#10131a",
                    font=(FONT, 10, "bold"), padding=(20, 8), borderwidth=0)
        s.map("Accent.TButton", background=[("active", self.accent),
                                            ("disabled", ACCENT_DIM)])
        s.configure("On.TButton", background=self.accent, foreground="#10131a",
                    font=(FONT, 9, "bold"), padding=(10, 5), borderwidth=0)
        s.configure("Off.TButton", background=PANEL2, foreground=MUTED,
                    font=(FONT, 9), padding=(10, 5), borderwidth=1,
                    focuscolor=PANEL2)
        s.map("Off.TButton", background=[("active", BORDER)])
        s.configure("Chip.TButton", font=(FONT, 9, "bold"), padding=(10, 3),
                    borderwidth=0, focuscolor=PANEL2)
        s.configure("TEntry", fieldbackground=PANEL2, borderwidth=1,
                    insertwidth=1, padding=(6, 5))
        s.configure("TSpinbox", fieldbackground=PANEL2, borderwidth=1,
                    padding=(6, 5))
        s.configure("TCombobox", fieldbackground=PANEL2, borderwidth=1,
                    padding=(6, 5), arrowcolor=MUTED)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL2)],
              foreground=[("readonly", TEXT)])
        self.option_add("*TCombobox*Listbox.background", PANEL2)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", self.accent)
        self.option_add("*TCombobox*Listbox.selectForeground", "#10131a")
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=TEXT, rowheight=25, borderwidth=0,
                    font=(FONT, 10))
        s.configure("Treeview.Heading", background=PANEL2, foreground=MUTED,
                    font=(FONT, 9, "bold"), relief="flat", padding=(6, 6))
        s.map("Treeview.Heading", background=[("active", BORDER)])
        s.map("Treeview", background=[("selected", BORDER)],
              foreground=[("selected", TEXT)])
        s.configure("Horizontal.TProgressbar", troughcolor=PANEL2,
                    background=self.accent, borderwidth=0, thickness=3)

    def _apply_accent(self, color):
        self.accent = color
        self.style.configure("Accent.TButton", background=color)
        self.style.map("Accent.TButton", background=[("active", color)])
        self.style.configure("On.TButton", background=color)
        self.style.configure("Horizontal.TProgressbar", background=color)
        self.style.configure("TCombobox", selectbackground=color)
        self.option_add("*TCombobox*Listbox.selectBackground", color)
        if self.result:
            self._show(self.result)

    # ---------------------------------------------------------------- build
    def _build(self):
        hdr = tk.Frame(self, bg=BG, pady=14, padx=18)
        hdr.pack(fill="x")
        tk.Label(hdr, text="◆", font=(FONT, 15), bg=BG,
                 fg=self.accent).pack(side="left")
        tk.Label(hdr, text=APP_NAME.upper(), font=(FONT, 16, "bold"),
                 bg=BG, fg=TEXT).pack(side="left", padx=(6, 0))
        self.tagline_var = self._tv("tagline")
        tk.Label(hdr, textvariable=self.tagline_var, font=(FONT, 9), bg=BG,
                 fg=MUTED).pack(side="left", padx=(12, 0), pady=(4, 0))

        self.gear_btn = tk.Label(hdr, text="⚙", font=(FONT, 13), bg=BG,
                                 fg=MUTED, cursor="hand2")
        self.gear_btn.pack(side="right", padx=(8, 0))
        self.gear_btn.bind("<Button-1>", lambda e: self._gear_menu())

        self.guide_btn = ttk.Button(hdr, textvariable=self._tv("btn_guide"),
                                    command=self._open_guide)
        self.guide_btn.pack(side="right", padx=(0, 6))

        self.pill_var = tk.StringVar(value="")
        self.pill = tk.Label(hdr, textvariable=self.pill_var, font=(FONT, 8, "bold"),
                             padx=10, pady=3, bg=UP, fg="#07130b")
        self.pill.pack(side="right", padx=(0, 10))

        lang_cell = tk.Frame(hdr, bg=BG)
        lang_cell.pack(side="right", padx=(0, 12))
        self.lang_cap_var = self._tv("lang_cap")
        tk.Label(lang_cell, textvariable=self.lang_cap_var,
                 font=(FONT, 8, "bold"), bg=BG, fg=MUTED).pack(anchor="e")
        self.lang_box = ttk.Combobox(lang_cell, width=9, state="readonly",
                                     values=list(LANG_LABEL))
        self.lang_box.set(LANG_CODE[self.lang])
        self.lang_box.pack(anchor="e")
        self.lang_box.bind("<<ComboboxSelected>>", lambda e: self._lang_changed())

        self.clock = tk.Label(hdr, text="", font=(MONO, 9), bg=BG, fg=MUTED)
        self.clock.pack(side="right", padx=(0, 14))
        self._tick()

        strip = tk.Frame(self, bg=BORDER, height=4)
        strip.pack(fill="x")
        for i, seg in enumerate(CHIP_COLORS):
            seg_f = tk.Frame(strip, bg=seg, height=4)
            seg_f.pack(side="left", fill="both", expand=True)

        ctl = tk.Frame(self, bg=PANEL, padx=14, pady=12)
        ctl.pack(fill="x", padx=14, pady=(12, 0))
        caps = [self._tv(k) for k in
                ("cap_market", "cap_symbol", "cap_interval", "cap_model",
                 "cap_context", "cap_horizon", "cap_paths")]
        makers = [self._venue_row, self._symbol_row, self._tf_row,
                  self._model_row, self._ctx_row, self._hor_row,
                  self._path_row]
        for col, (cap_var, maker) in enumerate(zip(caps, makers)):
            cell = tk.Frame(ctl, bg=PANEL)
            cell.grid(row=0, column=col, sticky="nsew", padx=(0, 12))
            tk.Label(cell, textvariable=cap_var, font=(FONT, 8, "bold"),
                     bg=PANEL, fg=FIELD_COLORS[col]).pack(anchor="w")
            maker(cell)
        btn_cell = tk.Frame(ctl, bg=PANEL)
        btn_cell.grid(row=0, column=7, sticky="nsew")
        tk.Label(btn_cell, text="‍", font=(FONT, 8), bg=PANEL,
                 fg=MUTED).pack(anchor="w")
        self.run_btn = ttk.Button(btn_cell, textvariable=self._tv("btn_run"),
                                  style="Accent.TButton", command=self._run)
        self.run_btn.pack(anchor="w")
        for i in range(8):
            ctl.columnconfigure(i, weight=1 if i < 7 else 0)

        quick = tk.Frame(ctl, bg=PANEL)
        quick.grid(row=1, column=0, columnspan=8, sticky="w", pady=(10, 0))
        self.quick_cap_var = self._tv("cap_quick")
        tk.Label(quick, textvariable=self.quick_cap_var, font=(FONT, 8, "bold"),
                 bg=PANEL, fg=MUTED).pack(side="left", padx=(0, 8))
        self.chips = tk.Frame(quick, bg=PANEL)
        self.chips.pack(side="left")

        kpis = tk.Frame(self, bg=BG)
        kpis.pack(fill="x", padx=14, pady=(10, 0))
        self.kpi_vars = {}
        kpi_caps = ("kpi_last", "kpi_hor", "kpi_chg", "kpi_band", "kpi_time")
        for i, cap in enumerate(kpi_caps):
            card = tk.Frame(kpis, bg=PANEL, highlightbackground=BORDER,
                            highlightthickness=1)
            card.pack(side="left", fill="x", expand=True,
                      padx=(0 if i == 0 else 8, 0))
            tk.Frame(card, bg=FIELD_COLORS[i % len(FIELD_COLORS)],
                     height=3).pack(fill="x")
            inner = tk.Frame(card, bg=PANEL, padx=12, pady=8)
            inner.pack(fill="x")
            tk.Label(inner, textvariable=self._tv(cap),
                     font=(FONT, 8, "bold"), bg=PANEL,
                     fg=MUTED).pack(anchor="w")
            var = tk.StringVar(value="—")
            lab = tk.Label(inner, textvariable=var, font=(FONT, 14, "bold"),
                           bg=PANEL, fg=TEXT)
            lab.pack(anchor="w")
            self.kpi_vars[cap] = (var, lab)

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(10, 8))

        left = tk.Frame(body, bg=PANEL, highlightbackground=BORDER,
                        highlightthickness=1)
        body.add(left, weight=3)
        bar = tk.Frame(left, bg=PANEL)
        bar.pack(fill="x")
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor=CHART_BG)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        NavigationToolbar2Tk(self.canvas, bar)
        for child in bar.winfo_children():
            try:
                child.configure(background=PANEL, activebackground=BORDER)
            except tk.TclError:
                pass
        self.chart_btn = ttk.Button(bar, textvariable=self._tv("chart_btn"),
                                    command=self._chart_menu)
        self.chart_btn.pack(side="right", padx=(6, 8), pady=2)
        self._chart_btn_var = self._texts[-1][0]
        self.grid_btn = ttk.Button(bar, textvariable=self._tv("t_grid"),
                                   command=self._toggle_grid)
        self.grid_btn.pack(side="right", padx=3, pady=2)
        self.log_btn = ttk.Button(bar, textvariable=self._tv("t_log"),
                                  command=self._toggle_log)
        self.log_btn.pack(side="right", padx=3, pady=2)
        self._log_btn_var = self._texts[-1][0]
        self._grid_btn_var = self._texts[-2][0]
        self._refresh_chart_btn()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        right = tk.Frame(body, bg=BG)
        body.add(right, weight=2)

        tk.Label(right, textvariable=self._tv("sec_forecast"),
                 font=(FONT, 9, "bold"), bg=BG, fg=MUTED).pack(anchor="w",
                                                               pady=(2, 4))
        cols = ("time", "open", "high", "low", "close", "chg")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=12)
        widths = {"time": 120, "open": 74, "high": 74, "low": 74,
                  "close": 74, "chg": 62}
        for c in cols:
            self.tree.column(c, width=widths[c],
                             anchor="w" if c == "time" else "e")
        self.tree.tag_configure("up", foreground=UP)
        self.tree.tag_configure("down", foreground=DOWN)
        self.tree.pack(fill="x")
        self.tree.bind("<ButtonRelease-1>", self._on_row_select)

        exp = tk.Frame(right, bg=BG)
        exp.pack(fill="x", pady=6)
        self.png_btn = ttk.Button(exp, textvariable=self._tv("btn_png"),
                                  state="disabled", command=self._save_fig)
        self.png_btn.pack(side="left")
        self.csv_btn = ttk.Button(exp, textvariable=self._tv("btn_csv"),
                                  state="disabled", command=self._save_csv)
        self.csv_btn.pack(side="left", padx=6)

        tk.Label(right, textvariable=self._tv("sec_activity"),
                 font=(FONT, 9, "bold"), bg=BG, fg=MUTED).pack(anchor="w",
                                                               pady=(4, 4))
        logwrap = tk.Frame(right, bg=BORDER, highlightbackground=BORDER,
                           highlightthickness=1)
        logwrap.pack(fill="both", expand=True)
        self.log_box = tk.Text(logwrap, bg=PANEL2, fg=MUTED,
                               insertbackground=TEXT, relief="flat",
                               font=(MONO, 9), wrap="word", state="disabled",
                               padx=8, pady=6)
        self.log_box.pack(fill="both", expand=True)
        self.log_box.tag_configure("ts", foreground="#5b6478")

        sb = tk.Frame(self, bg=PANEL, pady=5, padx=14)
        sb.pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(sb, mode="indeterminate", length=120)
        self.status = tk.StringVar(value="")
        tk.Label(sb, textvariable=self.status, font=(FONT, 9), bg=PANEL,
                 fg=TEXT).pack(side="left")
        self.progress.pack(side="right")
        self.data_note_var = self._tv("data_note")
        tk.Label(sb, textvariable=self.data_note_var, font=(FONT, 8),
                 bg=PANEL, fg=MUTED).pack(side="right", padx=14)

        self.bind("<Return>", lambda e: self._run())

    def _venue_row(self, parent):
        self.src = tk.StringVar(value="binance")
        box = ttk.Combobox(parent, textvariable=self.src, width=11,
                           state="normal", values=CRYPTO_SOURCES + ["yahoo"])
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", lambda e: self._src_changed())

    def _symbol_row(self, parent):
        self.symbol = tk.StringVar(value="BTC/USDT")
        self.sym_box = ttk.Combobox(parent, textvariable=self.symbol, width=14,
                                    values=CRYPTO_SYMBOLS)
        self.sym_box.pack(fill="x")

    def _tf_row(self, parent):
        self.tf = tk.StringVar(value="1h")
        self.tf_box = ttk.Combobox(parent, textvariable=self.tf, width=7,
                                   state="readonly", values=CRYPTO_TFS)
        self.tf_box.pack(fill="x")

    def _model_row(self, parent):
        self.model = tk.StringVar(value=LABEL_BY_KEY["balanced"])
        ttk.Combobox(parent, textvariable=self.model, state="readonly",
                     values=list(MODEL_LABELS)).pack(fill="x")

    def _ctx_row(self, parent):
        self.lookback = tk.IntVar(value=384)
        ttk.Spinbox(parent, from_=64, to=1024,
                    textvariable=self.lookback).pack(fill="x")

    def _hor_row(self, parent):
        self.pred_len = tk.IntVar(value=24)
        ttk.Spinbox(parent, from_=1, to=256,
                    textvariable=self.pred_len).pack(fill="x")

    def _path_row(self, parent):
        self.samples = tk.IntVar(value=8)
        ttk.Spinbox(parent, from_=1, to=64,
                    textvariable=self.samples).pack(fill="x")

    # -------------------------------------------------------------- config
    def _apply_cfg_values(self):
        venue = self.cfg.get("venue", "binance")
        if venue in CRYPTO_SOURCES + ["yahoo"]:
            self.src.set(venue)
        self._sync_venue_widgets(set_symbol=False)
        sym = self.cfg.get("symbol")
        if sym:
            self.symbol.set(sym)
        tf = self.cfg.get("tf")
        if tf:
            self.tf.set(tf)
        key = self.cfg.get("model")
        if key in LABEL_BY_KEY:
            self.model.set(LABEL_BY_KEY[key])
        for var, k in ((self.lookback, "lookback"), (self.pred_len, "pred_len"),
                       (self.samples, "samples")):
            v = self.cfg.get(k)
            if isinstance(v, int):
                var.set(v)

    def _save_cfg(self):
        data = {
            "lang": self.lang,
            "accent": self.accent_name,
            "chart_type": self.chart_type,
            "log_scale": self.log_scale,
            "grid": self.grid_on,
            "venue": self.src.get(),
            "symbol": self.symbol.get(),
            "tf": self.tf.get(),
            "model": MODEL_LABELS.get(self.model.get(), "balanced"),
            "lookback": int(self.lookback.get()),
            "pred_len": int(self.pred_len.get()),
            "samples": int(self.samples.get()),
        }
        try:
            CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        except Exception:
            pass

    def _on_close(self):
        self._save_cfg()
        self.destroy()

    # -------------------------------------------------------------- language
    def _lang_changed(self):
        code = LANG_LABEL.get(self.lang_box.get(), "tr")
        self._set_lang(code)

    def _set_lang(self, code):
        self.lang = code
        self.title(f"{APP_NAME} — {self.t('tagline')}")
        for var, key in self._texts:
            var.set(STR[code][key])
        for c in ("time", "open", "high", "low", "close", "chg"):
            self.tree.heading(c, text=self.t(f"col_{c}"))
        self._refresh_chart_btn()
        self._refresh_toggle_btns()
        self.pill_var.set(STR[code][STATE_KEYS[self._state]])
        if not self.guide_lang_pinned:
            if code in GUIDE_STR:
                self.guide_lang = code
            if self.guide_combo is not None and \
                    self.guide_combo.winfo_exists():
                self.guide_combo.set(GUIDE_CODE_LANG.get(self.guide_lang, "Türkçe"))
            self._refresh_guide()
        if self.result:
            self._show(self.result)
        elif not self._busy and self.ax is not None:
            self._set_chart_placeholder()
        if self.result:
            self.status.set(self.t("status_done", sym=self.result["symbol"],
                                   tf=self.result["tf"], s=self.result["secs"]))
        elif not self._busy:
            self.status.set(self.t("ready", app=APP_NAME))

    def _set_state(self, state, update_status=True):
        self._state = state
        bg, fg = STATE_STYLE[state]
        self.pill.configure(bg=bg, fg=fg)
        self.pill_var.set(STR[self.lang][STATE_KEYS[state]])
        if not update_status:
            return
        if state == "working":
            self.status.set(self.t("working"))
        elif state == "error":
            self.status.set(self.t("status_error"))
        elif state == "done" and self.result:
            self.status.set(self.t("status_done", sym=self.result["symbol"],
                                   tf=self.result["tf"], s=self.result["secs"]))
        elif state == "ready":
            self.status.set(self.t("ready", app=APP_NAME))

    def _tick(self):
        self.clock.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._tick)

    # ------------------------------------------------------- venue & chips
    def _sync_venue_widgets(self, set_symbol=True):
        src = self.src.get().strip().lower()
        if src == "yahoo":
            self.tf_box.configure(values=YF_TFS)
            if self.tf.get() not in YF_TFS:
                self.tf.set("1d")
            self.sym_box.configure(values=YF_SYMBOLS)
            if set_symbol:
                self.symbol.set("THYAO.IS")
        else:
            self.tf_box.configure(values=CRYPTO_TFS)
            if self.tf.get() not in CRYPTO_TFS:
                self.tf.set("1h")
            self.sym_box.configure(values=CRYPTO_SYMBOLS)
            if set_symbol:
                self.symbol.set("BTC/USDT")

    def _src_changed(self):
        src = self.src.get().strip().lower()
        current = self.symbol.get()
        pool = YF_SYMBOLS if src == "yahoo" else CRYPTO_SYMBOLS
        if current not in pool:
            self._sync_venue_widgets(set_symbol=True)
        else:
            self._sync_venue_widgets(set_symbol=False)
        self._rebuild_chips()

    def _rebuild_chips(self):
        for w in self.chips.winfo_children():
            w.destroy()
        src = self.src.get().strip().lower()
        syms = QUICK_YF if src == "yahoo" else QUICK_CRYPTO
        for i, sym in enumerate(syms):
            col = CHIP_COLORS[i % len(CHIP_COLORS)]
            btn = tk.Button(self.chips, text=sym, bg=col, fg="#0c0f14",
                            activebackground=col, activeforeground="#0c0f14",
                            relief="flat", font=(FONT, 9, "bold"),
                            padx=10, pady=2, cursor="hand2",
                            command=lambda s=sym: self._pick_symbol(s))
            btn.pack(side="left", padx=(0, 6))

    def _pick_symbol(self, sym):
        self.symbol.set(sym)

    # ----------------------------------------------------------- settings
    def _gear_menu(self):
        menu = tk.Menu(self, tearoff=0, bg=PANEL2, fg=TEXT, bd=0,
                       activebackground=self.accent, activeforeground="#10131a",
                       font=(FONT, 10))
        menu.add_command(label=self.t("acc_cap"), state="disabled")
        for name, color in ACCENTS.items():
            menu.add_radiobutton(label=self.t(f"acc_{name}"), value=name,
                                 variable=tk.StringVar(value=self.accent_name),
                                 background=PANEL2, foreground=TEXT,
                                 selectcolor=color,
                                 activebackground=self.accent,
                                 command=lambda n=name: self._set_accent(n))
        menu.post(self.gear_btn.winfo_rootx(),
                  self.gear_btn.winfo_rooty() + self.gear_btn.winfo_height())

    def _set_accent(self, name):
        self.accent_name = name
        self._apply_accent(ACCENTS[name])
        self.configure(bg=BG)

    # -------------------------------------------------------------- guide
    def _open_guide(self):
        if self.guide_win is not None and self.guide_win.winfo_exists():
            self.guide_win.lift()
            self.guide_win.focus_force()
            return
        win = tk.Toplevel(self)
        win.title(GUIDE_STR[self.guide_lang]["title"])
        win.geometry("780x660")
        win.minsize(560, 420)
        win.configure(bg=BG)
        win.transient(self)
        self.guide_win = win

        head = tk.Frame(win, bg=PANEL, padx=14, pady=10)
        head.pack(fill="x")
        self.guide_title_var = tk.StringVar(value=GUIDE_STR[self.guide_lang]["title"])
        tk.Label(head, textvariable=self.guide_title_var, font=(FONT, 14, "bold"),
                 bg=PANEL, fg=TEXT).pack(side="left")
        self.guide_combo = ttk.Combobox(head, width=11, state="readonly",
                                        values=GUIDE_LANGS)
        self.guide_combo.set(GUIDE_CODE_LANG.get(self.guide_lang, "Türkçe"))
        self.guide_combo.pack(side="right")
        tk.Label(head, text=STR[self.lang]["lang_cap"], font=(FONT, 8, "bold"),
                 bg=PANEL, fg=MUTED).pack(side="right", padx=(0, 8))
        self.guide_combo.bind("<<ComboboxSelected>>",
                              lambda e: self._guide_lang_selected())

        body = tk.Frame(win, bg=BG, padx=10, pady=10)
        body.pack(fill="both", expand=True)
        box = tk.Text(body, bg=PANEL2, fg=TEXT, relief="flat", font=(FONT, 10),
                      wrap="word", padx=14, pady=12, insertbackground=TEXT)
        sb = ttk.Scrollbar(body, command=box.yview)
        box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        box.pack(fill="both", expand=True)
        box.insert("1.0", GUIDE_STR[self.guide_lang]["body"])
        box.configure(state="disabled")
        self.guide_box = box

    def _guide_lang_selected(self):
        code = GUIDE_LANG_CODE.get(self.guide_combo.get(), "tr")
        self.guide_lang = code
        self.guide_lang_pinned = True
        self._refresh_guide()

    def _refresh_guide(self):
        if self.guide_win is None or not self.guide_win.winfo_exists():
            return
        title = GUIDE_STR[self.guide_lang]["title"]
        self.guide_win.title(title)
        self.guide_title_var.set(title)
        if self.guide_box is not None:
            self.guide_box.configure(state="normal")
            self.guide_box.delete("1.0", "end")
            self.guide_box.insert("1.0", GUIDE_STR[self.guide_lang]["body"])
            self.guide_box.configure(state="disabled")

    # -------------------------------------------------------------- chart
    def _refresh_chart_btn(self):
        self._chart_btn_var.set(self.t("chart_btn", t=self.t(f"t_{self.chart_type}")))

    def _refresh_toggle_btns(self):
        self.log_btn.configure(style="On.TButton" if self.log_scale
                               else "Off.TButton")
        self.grid_btn.configure(style="On.TButton" if self.grid_on
                                else "Off.TButton")

    def _toggle_log(self):
        self.log_scale = not self.log_scale
        self._refresh_toggle_btns()
        self._redraw()

    def _toggle_grid(self):
        self.grid_on = not self.grid_on
        self._refresh_toggle_btns()
        self._redraw()

    def _redraw(self):
        if self.result:
            self._show(self.result)
        else:
            self._set_chart_placeholder()

    def _chart_menu(self):
        menu = tk.Menu(self, tearoff=0, bg=PANEL2, fg=TEXT, bd=0,
                       activebackground=self.accent, activeforeground="#10131a",
                       font=(FONT, 10))
        for ct in ("line", "area", "candle"):
            menu.add_radiobutton(label=self.t(f"t_{ct}"), value=ct,
                                 variable=tk.StringVar(value=self.chart_type),
                                 selectcolor=PANEL,
                                 command=lambda c=ct: self._set_chart_type(c))
        menu.post(self.chart_btn.winfo_rootx(),
                  self.chart_btn.winfo_rooty() + self.chart_btn.winfo_height())

    def _set_chart_type(self, ct):
        self.chart_type = ct
        self._refresh_chart_btn()
        self._redraw()

    def _style_axes(self):
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.ax.tick_params(colors=MUTED, labelsize=8)
        self.ax.grid(self.grid_on, color=BORDER, alpha=0.35, lw=0.6)
        self.ax.set_axisbelow(True)

    def _set_chart_placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor(CHART_BG)
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.ax.text(0.5, 0.5, self.t("placeholder"), transform=self.ax.transAxes,
                     ha="center", va="center", color=MUTED, fontsize=12,
                     alpha=0.7)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.patch.set_facecolor(CHART_BG)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_candles(self, df, width_days):
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        up_mask = c >= o
        for x, hi, lo, op, cl, is_up in zip(list(df.index), h, l, o, c, up_mask):
            col = UP if is_up else DOWN
            self.ax.vlines(x, lo, hi, color=col, lw=0.9, zorder=3)
            self.ax.bar(x, max(cl - op, 0.0), bottom=op, width=width_days,
                        color=col, edgecolor=col, linewidth=0.3, zorder=4)

    def _fmt_price(self, v):
        return f"{v:,.4f}" if v < 100 else f"{v:,.2f}"

    # -------------------------------------------------------------- actions
    def _run(self):
        src = self.src.get().strip().lower()
        symbol = self.symbol.get().strip()
        tf = self.tf.get()
        model_key = MODEL_LABELS.get(self.model.get(), self.model.get())
        lb = min(max(int(self.lookback.get()), 64), MAX_LOOKBACK.get(model_key, 512))
        pl = min(max(int(self.pred_len.get()), 1), 256)
        n = min(max(int(self.samples.get()), 1), 64)
        if not symbol:
            messagebox.showwarning(self.t("err_title"), self.t("warn_symbol"))
            return
        self._busy = True
        self.run_btn.configure(state="disabled")
        self.progress.start(12)
        self._set_state("working")
        threading.Thread(target=self._worker, daemon=True,
                         args=(src, symbol, tf, model_key, lb, pl, n)).start()

    def _worker(self, src, symbol, tf, model_key, lb, pl, n):
        def log(msg):
            self.q.put(("log", msg))
        try:
            res = do_forecast(src, symbol, tf, model_key, lb, pl, n, log, self.t)
            self.q.put(("ok", res))
        except Exception as e:
            self.q.put(("err", f"{type(e).__name__}: {e}"))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    ts = datetime.now().strftime("%H:%M:%S")
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", f"{ts}  ", "ts")
                    self.log_box.insert("end", payload + "\n")
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                elif kind == "ok":
                    self.result = payload
                    self._busy = False
                    self._show(payload)
                    self.run_btn.configure(state="normal")
                    self.progress.stop()
                    self._set_state("done")
                elif kind == "err":
                    self.log(f"ERROR: {payload}")
                    self._busy = False
                    self.run_btn.configure(state="normal")
                    self.progress.stop()
                    self._set_state("error")
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def _show(self, res):
        hist, out = res["hist"], res["out"]
        m = out["mean"]

        self.ax.clear()
        self.ax.set_facecolor(CHART_BG)
        tail = hist.iloc[-160:]

        if self.chart_type == "line":
            self.ax.plot(tail.index, tail["close"].values, color=HIST_LINE,
                         lw=1.3, zorder=3)
            self.ax.plot(m.index, m["close"].values, color=self.accent, lw=1.8,
                         zorder=4)
            self.ax.fill_between(m.index, out["q10"]["close"],
                                 out["q90"]["close"], color=self.accent,
                                 alpha=0.15, zorder=2)
        elif self.chart_type == "area":
            self.ax.fill_between(tail.index, tail["close"].values, alpha=0.25,
                                 color=HIST_LINE, zorder=2)
            self.ax.plot(tail.index, tail["close"].values, color=HIST_LINE,
                         lw=1.2, zorder=3)
            self.ax.fill_between(m.index, m["close"], out["q90"]["close"],
                                 alpha=0.12, color=self.accent, zorder=2)
            self.ax.fill_between(m.index, out["q10"]["close"], m["close"],
                                 alpha=0.22, color=self.accent, zorder=3)
            self.ax.plot(m.index, m["close"].values, color=self.accent, lw=1.8,
                         zorder=4)
        else:
            combined = list(tail.index) + list(m.index)
            span = (combined[-1] - combined[0]).total_seconds() / 86400.0
            width = max(span / max(len(combined), 1) * 0.6, 1e-4)
            self._draw_candles(tail, width)
            self._draw_candles(m, width)
            self.ax.fill_between(m.index, out["q10"]["close"],
                                 out["q90"]["close"], color=self.accent,
                                 alpha=0.10, zorder=1)

        self.ax.axvline(hist.index[-1], color=BORDER, ls=(0, (4, 4)), lw=1,
                        zorder=5)
        if self.log_scale:
            self.ax.set_yscale("log")
        self._style_axes()
        handles = [
            Line2D([], [], color=HIST_LINE, lw=1.4,
                   label=self.t("hist_legend")),
            Line2D([], [], color=self.accent, lw=1.8,
                   label=self.t("fcst_legend")),
            Patch(facecolor=self.accent, alpha=0.2,
                  label=self.t("band_legend")),
        ]
        leg = self.ax.legend(handles=handles, loc="best", fontsize=9,
                             frameon=False)
        for txt in leg.get_texts():
            txt.set_color(TEXT)
        self.ax.set_title(self.t("chart_title", sym=res["symbol"],
                                 tf=res["tf"], app=APP_NAME),
                          color=TEXT, fontsize=11, pad=10)
        self.fig.patch.set_facecolor(CHART_BG)
        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.tree.delete(*self.tree.get_children())
        prev = float(hist["close"].iloc[-1])
        for ts, row in m.iterrows():
            chg = (row["close"] / prev - 1.0) * 100.0 if prev else 0.0
            tag = "up" if chg >= 0 else "down"
            self.tree.insert("", "end", tags=(tag,), values=(
                str(ts), self._fmt_price(row["open"]),
                self._fmt_price(row["high"]), self._fmt_price(row["low"]),
                self._fmt_price(row["close"]), f"{chg:+.2f}"))
            prev = row["close"]

        last = float(hist["close"].iloc[-1])
        end = float(m["close"].iloc[-1])
        chg = (end / last - 1.0) * 100.0
        lo = float(out["q10"]["close"].iloc[-1])
        hi = float(out["q90"]["close"].iloc[-1])
        self._set_kpi("kpi_last", self._fmt_price(last), TEXT)
        self._set_kpi("kpi_hor", self._fmt_price(end), UP if chg >= 0 else DOWN)
        self._set_kpi("kpi_chg", f"{chg:+.2f}%", UP if chg >= 0 else DOWN)
        self._set_kpi("kpi_band", f"{self._fmt_price(lo)} – {self._fmt_price(hi)}",
                      MUTED)
        self._set_kpi("kpi_time", f"{res['secs']:.1f}s", TEXT)

        self.png_btn.configure(state="normal")
        self.csv_btn.configure(state="normal")

    def _set_kpi(self, key, value, color):
        var, lab = self.kpi_vars[key]
        var.set(value)
        lab.configure(fg=color)

    def _on_row_select(self, _event):
        sel = self.tree.selection()
        if not sel or not self.result:
            return
        idx = self.tree.index(sel[0])
        m = self.result["out"]["mean"]
        lo = self.result["out"]["q10"].iloc[idx]
        hi = self.result["out"]["q90"].iloc[idx]
        self.log(self.t("log_band", t=str(m.index[idx]),
                        lo=f"{lo['close']:,.2f}", hi=f"{hi['close']:,.2f}"))

    def _save_fig(self):
        if not self.result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"candlecast_{self.result['symbol'].replace('/', '_')}.png",
            filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=130, facecolor=CHART_BG)
            self.log(self.t("log_chart_exported", p=path))

    def _save_csv(self):
        if not self.result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"candlecast_{self.result['symbol'].replace('/', '_')}.csv",
            filetypes=[("CSV", "*.csv")])
        if path:
            self.result["out"]["mean"].to_csv(path)
            self.log(self.t("log_csv_exported", p=path))

    def log(self, msg):
        self.q.put(("log", msg))


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
