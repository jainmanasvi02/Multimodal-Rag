"""Generate three small sample PDFs (text + charts/diagrams) with known facts.

The PDFs are already included in data/samples/, so you only need this script if you
want to regenerate or change them.  Extra dependencies:
    pip install reportlab matplotlib
Run from the repo root:
    python eval/make_sample_pdfs.py

DOCS below is the ground truth: every question in eval/questions.json can be answered
from the page text, or (for image questions) from the "image_facts" of that page.
"""
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "data" / "samples"


# ---------------------------------------------------------------- figures
def _to_image(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def quarterly_revenue_chart():
    fig, ax = plt.subplots(figsize=(6, 4))
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    values = [9.8, 11.4, 12.9, 14.1]
    bars = ax.bar(quarters, values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v}", ha="center")
    ax.set_title("Quarterly Revenue 2025 ($ millions)")
    ax.set_ylabel("Revenue ($M)")
    ax.set_ylim(0, 16)
    return _to_image(fig)


def region_pie_chart():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        [45, 30, 25],
        labels=["North America", "Europe", "Asia"],
        autopct="%1.0f%%",
        colors=["#4c72b0", "#dd8452", "#55a868"],
        startangle=90,
    )
    ax.set_title("Revenue by Region, 2025")
    return _to_image(fig)


def brewmaster_diagram():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec="black"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, weight="bold")

    box(1, 7.5, 5, 1.5, "Control panel", "#cfe8ff")
    box(1, 4.5, 5, 2, "Brew head", "#ffe0b3")
    box(1, 1, 5, 1.5, "Drip tray", "#d9f2d9")
    box(7, 1, 2.5, 8, "Water tank", "#e6ccff")
    ax.annotate("", xy=(3.5, 6.6), xytext=(3.5, 7.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.5, 2.6), xytext=(3.5, 4.4), arrowprops=dict(arrowstyle="->"))
    ax.set_title("BrewMaster 3000 - main parts")
    return _to_image(fig)


def sst_line_chart():
    import numpy as np

    years = list(range(2000, 2025))
    rng = np.random.default_rng(7)
    vals = np.linspace(28.7, 29.5, len(years)) + rng.uniform(-0.1, 0.1, len(years))
    vals[0] = 28.7
    vals[years.index(2016)] = 29.9  # unique peak
    vals[-1] = 29.6
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(years, vals, marker="o", color="#c0392b")
    ax.set_xticks(years[::2])
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Annual Mean Sea Surface Temperature, 2000-2024")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea surface temperature (deg C)")
    ax.grid(alpha=0.3)
    return _to_image(fig)


# ---------------------------------------------------------------- content
DOCS = {
    "acme_annual_report.pdf": [
        {
            "title": "Acme Corp Annual Report 2025: Executive Summary",
            "paragraphs": [
                "Acme Corp closed 2025 with total revenue of $48.2 million, a 12% increase over the "
                "$43.0 million reported in 2024. Net profit was $6.1 million. The company employs "
                "310 people across four offices: Pune, Berlin, Austin and Singapore.",
                "This report covers financial results, customer metrics and the outlook for 2026.",
            ],
        },
        {
            "title": "Quarterly Revenue",
            "paragraphs": [
                "Revenue grew every quarter of 2025. Q1 revenue was $9.8 million, Q2 was $11.4 million, "
                "Q3 was $12.9 million and Q4 was $14.1 million. Growth in Q3 was driven mainly by the "
                "launch of the Atlas analytics product, which signed 85 new enterprise accounts.",
            ],
            "image": quarterly_revenue_chart,
            "caption": "Figure 1: Quarterly revenue.",
            "image_facts": "Bar chart: Q1 9.8, Q2 11.4, Q3 12.9, Q4 14.1 ($M); Q4 is the tallest bar.",
        },
        {
            "title": "Customer Metrics",
            "paragraphs": [
                "At the end of 2025 Acme served 1,240 customers. The annual churn rate was 4.3% and the "
                "Net Promoter Score (NPS) reached 52. Mid-size retailers remain the largest customer "
                "segment, accounting for 38% of revenue.",
                "Figure 2 below breaks down revenue by geography.",
            ],
            "image": region_pie_chart,
            "caption": "Figure 2: Revenue by geography.",
            "image_facts": "Pie chart: North America 45%, Europe 30%, Asia 25%.",
        },
        {
            "title": "Risks and Outlook",
            "paragraphs": [
                "The main risks for 2026 are supply chain delays, currency volatility and rising cloud "
                "infrastructure costs. Management has set a revenue target of $55 million for 2026. "
                "Planned investments include opening a Toronto office and hiring 60 additional engineers.",
            ],
        },
    ],
    "brewmaster_manual.pdf": [
        {
            "title": "BrewMaster 3000: Quick Start Guide",
            "paragraphs": [
                "Fill the water tank with fresh water up to the MAX line. The tank holds a maximum of "
                "1.8 liters. Insert a paper filter, add ground coffee and press the power button.",
                "Before first use, run one rinse cycle with water only. The rinse cycle takes 90 seconds. "
                "The BrewMaster 3000 is covered by a 2-year limited warranty.",
            ],
        },
        {
            "title": "Parts Overview",
            "paragraphs": [
                "Figure 1 shows the main parts of the BrewMaster 3000. Refer to this diagram when "
                "following the maintenance instructions in this manual.",
            ],
            "image": brewmaster_diagram,
            "caption": "Figure 1: Main parts of the BrewMaster 3000.",
            "image_facts": "Diagram labels: Control panel (top), Brew head (middle, directly above the "
            "Drip tray), Drip tray (bottom), Water tank (right side).",
        },
        {
            "title": "Troubleshooting and Maintenance",
            "paragraphs": [
                "Error E4 means the water tank is empty or not seated correctly. Remove the tank, refill "
                "it and push it firmly into place. Error E7 means the brew head is clogged. Run a "
                "descaling cycle to clear it.",
                "Descale the machine every 3 months. Use 100 ml of descaling solution mixed with 500 ml "
                "of water for each cycle.",
            ],
        },
    ],
    "climate_study_summary.pdf": [
        {
            "title": "Coastal Temperature Study 2000-2024",
            "paragraphs": [
                "This study examines sea surface temperature (SST) in the Bay of Bengal between 2000 and "
                "2024. Data were collected from 42 moored buoys. Over the study period the mean SST "
                "increased by 0.9 degrees Celsius.",
            ],
        },
        {
            "title": "Results",
            "paragraphs": [
                "Figure 3 shows the annual mean sea surface temperature for the study period. Warming was "
                "not uniform: the series shows year-to-year variation around a clear upward trend.",
            ],
            "image": sst_line_chart,
            "caption": "Figure 3: Annual mean sea surface temperature.",
            "image_facts": "Line chart of annual mean sea surface temperature in degrees C; the "
            "highest point is 2016 (about 29.9); y-axis is sea surface temperature.",
        },
        {
            "title": "Methods and Funding",
            "paragraphs": [
                "Buoys sampled temperature every hour. Readings more than 3 standard deviations from the "
                "mean were removed as outliers. Trends were estimated with linear regression.",
                "The study was funded by the National Ocean Research Fund.",
            ],
        },
    ],
}


# ---------------------------------------------------------------- PDF writer
def write_pdf(path: Path, pages: list[dict]) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin = 50
    for page in pages:
        y = height - margin
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, y, page["title"])
        y -= 32
        c.setFont("Helvetica", 11)
        for para in page["paragraphs"]:
            for line in simpleSplit(para, "Helvetica", 11, width - 2 * margin):
                c.drawString(margin, y, line)
                y -= 15
            y -= 8
        if "image" in page:
            img = page["image"]()
            iw, ih = img.size
            draw_w = 380
            draw_h = draw_w * ih / iw
            c.drawImage(ImageReader(img), margin, y - draw_h - 10, width=draw_w, height=draw_h)
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(margin, y - draw_h - 24, page.get("caption", ""))
        c.showPage()
    c.save()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCS.items():
        write_pdf(OUT / name, pages)
        print(f"wrote {OUT / name} ({len(pages)} pages)")
