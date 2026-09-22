"""Chart generation service -- renders visual charts from tabular data files."""
import os, io, base64
from typing import Any, Dict, Optional

os.environ.setdefault("MPLCONFIGDIR",
    os.path.join(os.path.dirname(__file__), "..", ".matplot_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import pandas as pd
import numpy as np

from config import config

# --- Chinese font support (Windows) ---
_CN_FONTS = ['Microsoft YaHei', 'SimHei', 'DengXian', 'YouYuan', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = _CN_FONTS
plt.rcParams['axes.unicode_minus'] = False

_PALETTE = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#E2A74B', '#8C8C8C']


def _style_ax(ax):
    """Apply clean, modern spine / grid styling."""
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#D0D0D0')
    ax.tick_params(colors='#555555', labelsize=9)
    ax.grid(axis='y', alpha=0.2, linestyle='--', color='#CCCCCC')
    ax.set_axisbelow(True)


class ChartService:
    """Load a dataset and generate PNG-base64 charts."""

    def load_dataset(self, path: str) -> Optional[pd.DataFrame]:
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        try:
            if ext == "csv":
                for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
                    try:
                        with open(path, "r", encoding=enc) as fh:
                            return pd.read_csv(fh)
                    except (UnicodeDecodeError, OSError): continue
            elif ext in ("xlsx", "xls"):
                for engine in [None, "openpyxl", "xlrd"]:
                    try:
                        with open(path, "rb") as fh:
                            return pd.read_excel(fh, engine=engine)
                    except Exception: continue
            elif ext == "json":
                with open(path, "r", encoding="utf-8") as fh:
                    return pd.read_json(fh)
            elif ext == "txt":
                for enc in ["utf-8", "gbk", "gb2312"]:
                    try:
                        with open(path, "r", encoding=enc) as fh:
                            lines = [fh.readline() for _ in range(5)]
                        cand = {"\t": sum(l.count("\t") for l in lines),
                                ",": sum(l.count(",") for l in lines),
                                ";": sum(l.count(";") for l in lines),
                                "|": sum(l.count("|") for l in lines)}
                        best = max(cand, key=cand.get)
                        sep = best if cand[best] > 0 else None
                        with open(path, "r", encoding=enc) as fh:
                            return pd.read_csv(fh, sep=sep, engine="python")
                    except Exception: continue
        except Exception as e:
            print(f"[ChartService] load error: {e}")
        return None

    @staticmethod
    def _fig_b64(fig) -> str:
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def generate_charts(self, file_path: str) -> Dict[str, Any]:
        df = self.load_dataset(file_path)
        if df is None:
            return {"success": False, "error": "无法读取文件"}

        charts = []
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # --- histograms (first 4 numeric) ---
        for col in num_cols[:4]:
            fig, ax = plt.subplots(figsize=(8, 3.8))
            data = df[col].dropna()
            _style_ax(ax)
            bins = min(30, max(5, len(data)//5))
            ax.hist(data, bins=bins, color=_PALETTE[0], edgecolor="white",
                    linewidth=0.8, alpha=0.85)
            mu = data.mean()
            ax.axvline(mu, color=_PALETTE[2], linestyle="--", linewidth=1.5, alpha=0.7,
                       label=f"均值 = {mu:.1f}")
            ax.legend(fontsize=9, loc="upper right", frameon=True,
                      facecolor="white", edgecolor="#E0E0E0")
            ax.set_title(f"{col}  分布", fontsize=14, fontweight="bold",
                         color="#2C3E50", pad=12)
            ax.set_xlabel(col, fontsize=11, color="#555555")
            ax.set_ylabel("频数", fontsize=11, color="#555555")
            charts.append({"type": "histogram", "column": col,
                           "image": self._fig_b64(fig)})

        # --- bar charts for categorical columns (first 4) ---
        for idx, col in enumerate(text_cols[:4]):
            fig, ax = plt.subplots(figsize=(8, 3.8))
            top = df[col].value_counts().head(10)
            _style_ax(ax)
            bar_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(top))]
            bars = ax.bar(range(len(top)), top.values, color=bar_colors,
                         edgecolor="white", linewidth=0.5, width=0.7)
            ax.set_xticks(range(len(top)))
            ax.set_xticklabels(top.index.astype(str), rotation=35, ha="right",
                              fontsize=9, color="#555555")
            ax.set_title(f"{col}  前 10（频数）", fontsize=14, fontweight="bold",
                         color="#2C3E50", pad=12)
            ax.set_ylabel("频数", fontsize=11, color="#555555")
            for b in bars:
                h = b.get_height()
                ax.annotate(str(int(h)), xy=(b.get_x() + b.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=9, fontweight="bold",
                            color="#444444")
            charts.append({"type": "bar", "column": col,
                           "image": self._fig_b64(fig)})

        # --- correlation heatmap (>=2 numeric cols) ---
        if len(num_cols) >= 2:
            ndf = df[num_cols].dropna()
            if ndf.shape[1] >= 2 and ndf.shape[0] >= 3:
                corr = ndf.corr()
                n = len(corr.columns)
                fig, ax = plt.subplots(figsize=(max(6, n*0.8), max(5, n*0.7)))
                im = ax.imshow(corr.values, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1)
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
                ax.set_yticklabels(corr.columns, fontsize=9)
                cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                cbar.ax.tick_params(labelsize=9)
                cbar.set_label("相关系数", fontsize=10, color="#555555")
                for i in range(n):
                    for j in range(n):
                        v = corr.values[i, j]
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8, fontweight="bold" if abs(v) > 0.5 else "normal",
                                color="white" if abs(v) > 0.5 else "#333333")
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_title("相关性热力图", fontsize=14, fontweight="bold",
                             color="#2C3E50", pad=14)
                charts.append({"type": "correlation",
                               "columns": list(corr.columns),
                               "image": self._fig_b64(fig)})

        # --- statistics summary ---
        stats = {}
        for col in num_cols[:8]:
            s = df[col].describe()
            stats[col] = {k: round(float(v), 2) if isinstance(v, (float, np.floating)) else int(v)
                          for k, v in s.items()}

        return {
            "success": True,
            "data": {
                "charts": charts,
                "stats": stats,
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "numeric_cols": len(num_cols),
                "text_cols": len(text_cols),
                "filename": os.path.basename(file_path),
            }
        }



