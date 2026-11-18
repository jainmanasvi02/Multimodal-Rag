"""Generate a larger evaluation corpus (15 PDFs) and an extended question set.

Writes:
  data/corpus/*.pdf          the 3 original sample PDFs + 12 new ones
  eval/questions_large.json  the original 27 questions + new ones (unchanged originals,
                             so the same questions now face a harder corpus)

Hard on purpose: four near-identical company annual reports (Acme, Borealis, Cedarline,
Delphi), similar appliance manuals, several studies with the same structure, and policy
documents.  Retrieval must tell documents apart by entity name and detail.

Extra dependencies:  pip install reportlab matplotlib
Run from the repo root:  python eval/make_large_corpus.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from make_sample_pdfs import DOCS as ORIGINAL_DOCS
from make_sample_pdfs import _to_image, write_pdf

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "corpus"
PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]


# ------------------------------------------------------------------ chart helpers
def line_chart(title, x, y, ylabel, rotate=0):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, y, marker="o", color="#c0392b")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    ax.grid(alpha=0.3)
    return _to_image(fig)


def bar_chart(title, labels, values, ylabel, horizontal=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = PALETTE[: len(labels)]
    top = max(values)
    if horizontal:
        bars = ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_xlim(0, top * 1.15)
        for b, v in zip(bars, values):
            ax.text(v + top * 0.01, b.get_y() + b.get_height() / 2, f"{v}", va="center")
    else:
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, top * 1.15)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.01, f"{v}", ha="center")
    ax.set_title(title)
    return _to_image(fig)


def pie_chart(title, labels, values):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(values, labels=labels, autopct="%1.0f%%", colors=PALETTE[: len(labels)], startangle=90)
    ax.set_title(title)
    return _to_image(fig)


def stack_diagram(title, stack, side=None):
    """Vertical stack of labeled boxes (top to bottom) joined by arrows; optional tall side box."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fills = ["#cfe8ff", "#ffe0b3", "#d9f2d9", "#f9d6d5", "#e6ccff"]
    n, gap, top = len(stack), 0.8, 9.0
    h = (8.4 - gap * (n - 1)) / n
    for i, label in enumerate(stack):
        y = top - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((1, y), 5, h, boxstyle="round,pad=0.1", fc=fills[i % 5], ec="black"))
        ax.text(3.5, y + h / 2, label, ha="center", va="center", fontsize=11, weight="bold")
        if i < n - 1:
            ax.annotate("", xy=(3.5, y - gap + 0.15), xytext=(3.5, y - 0.15), arrowprops=dict(arrowstyle="->"))
    if side:
        ax.add_patch(FancyBboxPatch((7, 0.6), 2.5, 8.4, boxstyle="round,pad=0.1", fc="#e6ccff", ec="black"))
        ax.text(8.25, 4.8, side, ha="center", va="center", fontsize=11, weight="bold")
    ax.set_title(title)
    return _to_image(fig)


# ------------------------------------------------------------------ content builders
def join_and(items):
    return ", ".join(items[:-1]) + " and " + items[-1]


def company_pages(c, extra=None):
    """Four-page annual report built from the same template for every company.
    extra = {page_index: {"image": callable, "caption": str, "image_facts": str, "note": str}}"""
    extra = extra or {}
    n, q = c["name"], c["quarters"]
    assert round(sum(q), 1) == c["rev"], f"{n}: quarters do not sum to revenue"
    growth = round((c["rev"] / c["prev"] - 1) * 100)
    pages = [
        {
            "title": f"{n} Annual Report 2025: Executive Summary",
            "paragraphs": [
                f"{n} closed 2025 with total revenue of ${c['rev']:.1f} million, a {growth}% increase over "
                f"the ${c['prev']:.1f} million reported in 2024. Net profit was ${c['profit']:.1f} million. "
                f"The company employs {c['emp']:,} people across {len(c['sites'])} {c['sitekind']}: "
                f"{join_and(c['sites'])}.",
                f"{n} was founded in {c['founded']} and is headquartered in {c['hq']}. The chief executive "
                f"officer is {c['ceo']}. This report covers financial results, customer metrics and the "
                f"outlook for 2026.",
            ],
        },
        {
            "title": "Quarterly Performance",
            "paragraphs": [
                f"Revenue grew in every quarter of 2025. Q1 revenue was ${q[0]:.1f} million, Q2 was "
                f"${q[1]:.1f} million, Q3 was ${q[2]:.1f} million and Q4 was ${q[3]:.1f} million. Growth in "
                f"{c['growth_q']} was driven mainly by the launch of the {c['product']} {c['ptype']}, which "
                f"signed {c['newaccts']} new {c['acctkind']}.",
                f"Operating margin for the year was {c['margin']}%. Research and development spending "
                f"totaled ${c['rd']:.1f} million.",
            ],
        },
        {
            "title": "Customer Metrics",
            "paragraphs": [
                f"At the end of 2025 {n} served {c['customers']:,} customers. The annual churn rate was "
                f"{c['churn']}% and the Net Promoter Score (NPS) reached {c['nps']}. {c['segment']} remain "
                f"the largest customer segment, accounting for {c['segpct']}% of revenue.",
                f"The average contract length was {c['contract']} months, and {c['resolve']}% of support "
                f"tickets were resolved within one business day.",
            ],
        },
        {
            "title": "Risks and Outlook",
            "paragraphs": [
                f"The main risks for 2026 are {c['risks']}. Management has set a revenue target of "
                f"${c['target']} million for 2026. Planned investments include opening a {c['newplace']} "
                f"{c['newkind']} and hiring {c['hires']} additional {c['hirekind']}.",
                f"The board proposed a dividend of ${c['div']:.2f} per share, payable in {c['paymonth']} 2026.",
            ],
        },
    ]
    for idx, e in extra.items():
        e = dict(e)
        note = e.pop("note", None)
        if note:
            pages[idx]["paragraphs"].append(note)
        pages[idx].update(e)
    return pages


COMPANIES = {
    "borealis_annual_report.pdf": (
        dict(name="Borealis Systems", founded=2009, hq="London", ceo="Maria Lindqvist", rev=73.5, prev=64.9,
             profit=9.4, emp=520, sitekind="offices", sites=["London", "Toronto", "Sydney", "Dublin", "Mumbai"],
             quarters=[15.2, 17.8, 18.9, 21.6], growth_q="Q2", product="Helix", ptype="data platform",
             newaccts=112, acctkind="enterprise customers", margin=14, rd=11.3, customers=2030, churn=3.1,
             nps=61, segment="Financial services firms", segpct=44, contract=26, resolve=91,
             risks="regulatory changes, talent shortages and rising data center costs", target=84,
             newplace="Amsterdam", newkind="office", hires=90, hirekind="software engineers", div=0.40,
             paymonth="April"),
        {1: dict(
            image=lambda: line_chart(
                "Borealis Systems: Monthly Active Users 2025 (thousands)",
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                [310, 318, 327, 335, 348, 356, 361, 372, 385, 412, 398, 405], "Active users (thousands)"),
            caption="Figure 1: Borealis monthly active users.",
            image_facts="Line chart of Borealis monthly active users (thousands) for 2025; the highest "
                        "point is October (412).",
            note="Figure 1 below tracks user activity through the year.")},
    ),
    "cedarline_annual_report.pdf": (
        dict(name="Cedarline Foods", founded=1978, hq="Leeds", ceo="Tom Hargreaves", rev=212.8, prev=201.4,
             profit=17.3, emp=1850, sitekind="plants", sites=["Leeds", "Lyon", "Porto"],
             quarters=[49.6, 51.7, 54.2, 57.3], growth_q="Q4", product="Harvest Oat", ptype="snack line",
             newaccts=38, acctkind="retail partners", margin=8, rd=4.2, customers=3400, churn=5.8, nps=44,
             segment="Supermarket chains", segpct=52, contract=18, resolve=84,
             risks="crop price volatility, packaging costs and shifting consumer tastes", target=225,
             newplace="Rotterdam", newkind="distribution center", hires=120, hirekind="production staff",
             div=1.10, paymonth="May"),
        {1: dict(
            image=lambda: bar_chart(
                "Cedarline Foods: Sales by Product Line 2025 ($M)",
                ["Dairy", "Bakery", "Snacks", "Beverages"], [78.4, 61.2, 44.9, 28.3], "Sales ($M)",
                horizontal=True),
            caption="Figure 1: Cedarline sales by product line.",
            image_facts="Horizontal bar chart of Cedarline sales by product line ($M): Dairy 78.4, "
                        "Bakery 61.2, Snacks 44.9, Beverages 28.3; Dairy is the largest.",
            note="Figure 1 below shows how sales split across product lines."),
         2: dict(
            image=lambda: pie_chart("Cedarline Foods: Cost Breakdown 2025",
                                    ["Ingredients", "Labor", "Logistics", "Marketing"], [40, 25, 20, 15]),
            caption="Figure 2: Cedarline cost breakdown.",
            image_facts="Pie chart of Cedarline costs: Ingredients 40%, Labor 25%, Logistics 20%, "
                        "Marketing 15%.",
            note="Figure 2 below breaks down costs by category.")},
    ),
    "delphi_annual_report.pdf": (
        dict(name="Delphi Logistics", founded=1995, hq="Chicago", ceo="Priya Nair", rev=96.4, prev=90.1,
             profit=5.2, emp=1120, sitekind="warehouses", sites=["Chicago", "Dubai", "Hamburg", "Osaka"],
             quarters=[22.1, 23.4, 24.9, 26.0], growth_q="Q3", product="RouteIQ", ptype="fleet routing tool",
             newaccts=64, acctkind="freight clients", margin=6, rd=2.9, customers=860, churn=6.4, nps=38,
             segment="E-commerce retailers", segpct=47, contract=14, resolve=79,
             risks="fuel price swings, port congestion and driver shortages", target=105,
             newplace="Nairobi", newkind="hub", hires=150, hirekind="drivers and dispatchers", div=0.25,
             paymonth="June"),
        {1: dict(
            image=lambda: bar_chart(
                "Delphi Logistics: On-Time Delivery Rate by Region 2025 (%)",
                ["Europe", "North America", "Asia", "Middle East"], [94, 91, 89, 87], "On-time rate (%)"),
            caption="Figure 1: Delphi on-time delivery by region.",
            image_facts="Bar chart of Delphi on-time delivery rate (%): Europe 94, North America 91, "
                        "Asia 89, Middle East 87; Middle East is the lowest.",
            note="Figure 1 below compares delivery performance across regions.")},
    ),
}

NEW_DOCS = {name: company_pages(c, extra) for name, (c, extra) in COMPANIES.items()}

NEW_DOCS["airpure_200_manual.pdf"] = [
    {"title": "AirPure 200: Quick Start Guide",
     "paragraphs": [
         "Place the AirPure 200 at least 30 cm from walls and plug it in. Press the power button to start. "
         "The unit covers rooms up to 35 square meters and runs at 24 dB on its lowest setting.",
         "The HEPA filter must be replaced every 6 months. A new filter kit includes one HEPA filter and one "
         "carbon pre-filter. The AirPure 200 is covered by a 3-year limited warranty."]},
    {"title": "Parts Overview",
     "paragraphs": ["Figure 1 shows the main parts of the AirPure 200. Refer to this diagram when replacing "
                    "the filter."],
     "image": lambda: stack_diagram("AirPure 200 - main parts",
                                    ["Air outlet", "Fan unit", "HEPA filter", "Air intake"],
                                    side="Control panel"),
     "caption": "Figure 1: Main parts of the AirPure 200.",
     "image_facts": "Diagram labels top to bottom: Air outlet, Fan unit, HEPA filter, Air intake; Control "
                    "panel on the right side. The HEPA filter sits between the Fan unit and the Air intake."},
    {"title": "Troubleshooting and Maintenance",
     "paragraphs": [
         "Code F2 means the filter is missing or not seated correctly. Code F5 means the fan is blocked by "
         "debris. Switch the unit off and remove the obstruction before restarting.",
         "Wipe the air intake grille with a dry cloth every 2 weeks. Do not wash the HEPA filter. The "
         "replacement filter costs $39."]},
]

NEW_DOCS["kettlemax_manual.pdf"] = [
    {"title": "KettleMax 1.7: Quick Start Guide",
     "paragraphs": [
         "Fill the KettleMax with water between the MIN and MAX marks. The kettle holds a maximum of 1.7 "
         "liters. Press the switch down to boil; boiling a full kettle takes about 4 minutes and the kettle "
         "switches itself off automatically.",
         "Rinse the kettle by boiling fresh water twice before first use. The KettleMax 1.7 is covered by a "
         "1-year limited warranty."]},
    {"title": "Parts Overview",
     "paragraphs": ["Figure 1 shows the main parts of the KettleMax. Keep the base dry and away from the "
                    "sink."],
     "image": lambda: stack_diagram("KettleMax 1.7 - main parts",
                                    ["Lid", "Water chamber", "Heating base"], side="Handle"),
     "caption": "Figure 1: Main parts of the KettleMax.",
     "image_facts": "Diagram labels top to bottom: Lid, Water chamber, Heating base; Handle on the right "
                    "side. The Heating base sits directly below the Water chamber."},
    {"title": "Troubleshooting and Maintenance",
     "paragraphs": [
         "A light flashing three times means dry-boil protection was triggered. Let the kettle cool for 10 "
         "minutes, then add water. If the kettle leaks from the base, stop using it and contact support.",
         "Descale the kettle every 2 months. Use 50 g of citric acid dissolved in 1 liter of water, boil it, "
         "and rinse twice."]},
]

NEW_DOCS["soundbar_x_manual.pdf"] = [
    {"title": "SoundBar X: Quick Start Guide",
     "paragraphs": [
         "Connect the SoundBar X to your television using the HDMI ARC port. The soundbar delivers 120 watts "
         "of total output power and supports Bluetooth 5.3. Pairing mode starts when the Bluetooth button is "
         "held for 3 seconds.",
         "The remote uses two AAA batteries. The SoundBar X is covered by a 2-year limited warranty."]},
    {"title": "Ports and Connections",
     "paragraphs": ["Figure 1 shows the connection ports on the back of the SoundBar X. Switch the unit off "
                    "before connecting any cable."],
     "image": lambda: stack_diagram("SoundBar X - rear ports",
                                    ["HDMI ARC", "Optical input", "USB port", "Power input"]),
     "caption": "Figure 1: Rear ports of the SoundBar X.",
     "image_facts": "Diagram labels top to bottom: HDMI ARC, Optical input, USB port, Power input. The USB "
                    "port sits directly below the Optical input."},
    {"title": "Troubleshooting",
     "paragraphs": [
         "A red light blinking 3 times means the unit is overheating. Move it away from other devices and let "
         "it cool for 15 minutes. If there is no sound, check that the TV audio output is set to HDMI ARC.",
         "To reset the soundbar to factory settings, hold the volume down button for 10 seconds. Firmware "
         "updates are installed via USB and take about 5 minutes."]},
]

NEW_DOCS["urban_air_quality_study.pdf"] = [
    {"title": "Urban Air Quality Study 2018-2024",
     "paragraphs": [
         "This study measures nitrogen dioxide (NO2) concentrations in the city of Brightport between 2018 "
         "and 2024. Data were collected from 64 roadside sensors. The mean annual NO2 concentration fell by "
         "18 percent over the study period.",
         "The study covers 12 districts and 3 million residents."]},
    {"title": "Results",
     "paragraphs": ["Figure 2 shows the annual mean NO2 concentration for the study period. The "
                    "low-emission zone was introduced in 2021."],
     "image": lambda: line_chart("Annual Mean NO2 Concentration, Brightport 2018-2024",
                                 list(range(2018, 2025)), [41.0, 44.0, 36.0, 38.0, 35.0, 34.0, 33.6],
                                 "NO2 (ug/m3)"),
     "caption": "Figure 2: Annual mean NO2 concentration.",
     "image_facts": "Line chart of annual mean NO2 concentration (ug/m3), 2018-2024; the highest point is "
                    "2019 (44)."},
    {"title": "Methods and Funding",
     "paragraphs": [
         "Sensors recorded hourly readings. Readings from sensors with more than 20 percent missing data were "
         "excluded. Trends were estimated with a mixed-effects model.",
         "The study was funded by the Clean Cities Initiative."]},
]

NEW_DOCS["soil_moisture_trial.pdf"] = [
    {"title": "Soil Moisture and Crop Yield Trial",
     "paragraphs": [
         "This field trial compared three irrigation levels on maize yield across 120 plots in the Kaveri "
         "basin. The high irrigation level raised average yield by 22 percent compared with the low level.",
         "The trial ran for two growing seasons in 2022 and 2023."]},
    {"title": "Results",
     "paragraphs": ["Figure 3 shows the average maize yield for each irrigation level in tons per hectare."],
     "image": lambda: bar_chart("Maize Yield by Irrigation Level (tons/ha)", ["Low", "Medium", "High"],
                                [5.1, 5.8, 6.2], "Yield (tons/ha)"),
     "caption": "Figure 3: Average maize yield by irrigation level.",
     "image_facts": "Bar chart of maize yield (tons/ha): Low 5.1, Medium 5.8, High 6.2; High irrigation "
                    "gives the highest yield."},
    {"title": "Methods and Funding",
     "paragraphs": [
         "Plots were assigned to irrigation levels at random. Differences between levels were tested with a "
         "one-way ANOVA. Soil moisture was logged every 30 minutes by buried probes.",
         "The trial was funded by the Agri Research Council."]},
]

NEW_DOCS["sleep_memory_study.pdf"] = [
    {"title": "Sleep and Memory Clinical Study",
     "paragraphs": [
         "This randomized controlled trial enrolled 240 adult participants to test the effect of sleep "
         "duration on word recall. Participants who slept 8 hours improved their recall score by 14 percent "
         "compared with the control group.",
         "Participants were aged 18 to 35."]},
    {"title": "Results",
     "paragraphs": ["Figure 4 shows the mean recall score for each group after one week."],
     "image": lambda: bar_chart("Mean Word Recall Score by Sleep Group", ["Control", "6-hour", "8-hour"],
                                [62, 64, 71], "Mean recall score"),
     "caption": "Figure 4: Mean recall score by group.",
     "image_facts": "Bar chart of mean recall score: Control 62, 6-hour 64, 8-hour 71; the 8-hour group is "
                    "highest."},
    {"title": "Methods and Funding",
     "paragraphs": [
         "Recall was scored by two independent raters who did not know the group assignments. Group "
         "differences were analyzed with ANCOVA, adjusting for baseline scores.",
         "The study was funded by the Wellness Science Trust."]},
]

NEW_DOCS["employee_handbook.pdf"] = [
    {"title": "Employee Handbook: Leave Policy",
     "paragraphs": [
         "Full-time employees receive 24 days of paid annual leave per year. Unused leave of up to 5 days "
         "can be carried over to the next year. Sick leave is 10 days per year and requires a medical "
         "certificate after 3 consecutive days.",
         "Parental leave is 16 weeks for the primary caregiver and 4 weeks for the secondary caregiver."]},
    {"title": "Working Arrangements",
     "paragraphs": [
         "Employees may work remotely up to 3 days per week with manager approval. Core hours are 10:00 to "
         "16:00 local time. Overtime must be approved in advance and is compensated with time off in lieu.",
         "Company laptops must be returned within 7 days of leaving the company."]},
    {"title": "Hiring and Probation",
     "paragraphs": [
         "New employees serve a probation period of 3 months. Either side may end employment during "
         "probation with 7 days of notice. After probation the notice period is 60 days.",
         "Referral bonuses of $1,500 are paid after the referred employee completes 6 months."]},
    {"title": "Training and Development",
     "paragraphs": [
         "Each employee has an annual learning budget of $800. Certification exams are reimbursed in full "
         "after a passing result. Employees must complete the mandatory compliance course within 30 days of "
         "joining."]},
]

NEW_DOCS["travel_expense_policy.pdf"] = [
    {"title": "Travel and Expense Policy: Limits",
     "paragraphs": [
         "The daily meal allowance is $45. Hotel stays are reimbursed up to $180 per night, or $250 per "
         "night in New York, London and Tokyo.",
         "Economy class is required for flights shorter than 6 hours. Business class may be booked for "
         "flights of 6 hours or longer."]},
    {"title": "Receipts and Approvals",
     "paragraphs": [
         "Receipts are required for every expense above $25. Expenses above $500 need approval from a "
         "department manager before booking, and expenses above $5,000 need approval from the finance "
         "director.",
         "Use the company card for all travel bookings."]},
    {"title": "Reimbursement",
     "paragraphs": [
         "Expense reports must be submitted within 30 days of the trip. Approved claims are reimbursed "
         "within 14 days. Personal car use is reimbursed at $0.60 per kilometer.",
         "Lost receipts require a signed declaration form."]},
]

NEW_DOCS["it_security_guidelines.pdf"] = [
    {"title": "IT Security Guidelines: Passwords",
     "paragraphs": [
         "Passwords must be at least 14 characters long and must be changed every 180 days. Reusing any of "
         "the last 10 passwords is not allowed. Multi-factor authentication is mandatory for all company "
         "accounts.",
         "Password managers approved by IT may be used to store credentials."]},
    {"title": "Devices and Data",
     "paragraphs": [
         "All laptops must use full-disk encryption with AES-256. Company data may only be stored in "
         "approved cloud storage. USB storage devices are blocked by default.",
         "Lost or stolen devices must be reported to the security desk within 4 hours."]},
    {"title": "Incident Response",
     "paragraphs": [
         "Suspected security incidents must be reported within 24 hours using the security hotline. The "
         "security team acknowledges reports within 1 hour during business days. Phishing emails should be "
         "forwarded to the phishing mailbox and then deleted."]},
]

# ------------------------------------------------------------------ boilerplate
# Real documents carry plenty of generic text that no question asks about.  It is added after the
# facts on every NEW page (the 3 original PDFs stay untouched).  It contains no digits and no
# answer keywords (checked below), and the same sentences repeat across documents of one kind.
BOILERPLATE = {
    "report": [
        "Management remains committed to disciplined execution, transparent reporting and long-term value "
        "creation for shareholders. The leadership team reviews strategic priorities every quarter and "
        "adjusts investment plans as market conditions evolve.",
        "Employees across the organization contributed to these results, and the company thanks its "
        "customers and partners for their continued trust. Sustainability and responsible sourcing "
        "remain part of every major business decision.",
        "The company continues to invest in training, internal tools and customer support so that teams can "
        "respond quickly to changing needs. Governance practices are reviewed regularly by the board and "
        "its audit committee.",
        "Forward-looking statements in this report reflect current expectations and are subject to "
        "uncertainty. Actual results may differ materially from those described, and readers should not "
        "place undue reliance on them.",
    ],
    "manual": [
        "Read all safety instructions carefully before using this product and keep this manual for future "
        "reference. Use only original accessories and replacement parts supplied or approved by the "
        "manufacturer.",
        "Do not immerse the unit in water or any other liquid, and keep it out of reach of children. If the "
        "power cord is damaged, it must be replaced by the manufacturer or a qualified technician to avoid "
        "hazards.",
        "This appliance is intended for normal household use only. The manufacturer accepts no liability "
        "for damage caused by improper use, unauthorized repairs or failure to follow these instructions.",
    ],
    "study": [
        "The authors thank the field teams, laboratory staff and participants who made this work possible. "
        "All procedures followed institutional ethical guidelines and the relevant national regulations.",
        "The full dataset and analysis code are available from the corresponding author on reasonable "
        "request. Limitations include the relatively short observation period and the possibility of "
        "unmeasured confounding factors.",
        "Future work should replicate these findings in other settings and combine them with complementary "
        "data sources. The conclusions presented here are those of the authors and do not necessarily "
        "reflect the views of the funding body.",
    ],
    "policy": [
        "Questions about this policy should be directed to the relevant department. Exceptions require "
        "written approval and are reviewed on a case-by-case basis.",
        "This policy is reviewed annually and may be updated at any time with notice to all staff. Failure "
        "to comply may result in disciplinary action in accordance with company procedures.",
        "Managers are responsible for making sure their teams understand and follow these rules. Staff who "
        "are unsure how a rule applies to them should ask before acting.",
    ],
}


def _kind(doc_name):
    if doc_name.endswith("annual_report.pdf"):
        return "report"
    if doc_name.endswith("_manual.pdf"):
        return "manual"
    if doc_name in ("soil_moisture_trial.pdf", "sleep_memory_study.pdf", "urban_air_quality_study.pdf"):
        return "study"
    return "policy"


for _name, _pages in NEW_DOCS.items():
    _bank = BOILERPLATE[_kind(_name)]
    for _i, _page in enumerate(_pages):
        _page["paragraphs"].append(_bank[_i % len(_bank)])

ALL_DOCS = {**ORIGINAL_DOCS, **NEW_DOCS}


# ------------------------------------------------------------------ new questions
def T(id, q, src, page, ans):
    return {"id": id, "question": q, "kind": "text", "source": src, "page": page, "answer_contains": ans}


def I(id, q, src, page, ans):
    return {"id": id, "question": q, "kind": "image", "source": src, "page": page, "answer_contains": ans}


def O(id, q, diff):
    return {"id": id, "question": q, "kind": "offtopic", "difficulty": diff}


BO, CE, DE = "borealis_annual_report.pdf", "cedarline_annual_report.pdf", "delphi_annual_report.pdf"
AP, KE, SB = "airpure_200_manual.pdf", "kettlemax_manual.pdf", "soundbar_x_manual.pdf"
UR, SO, SL = "urban_air_quality_study.pdf", "soil_moisture_trial.pdf", "sleep_memory_study.pdf"
HB, TR, SE = "employee_handbook.pdf", "travel_expense_policy.pdf", "it_security_guidelines.pdf"

NEW_QUESTIONS = [
    T("t14", "What was Borealis Systems' total revenue in 2025?", BO, 0, ["73.5"]),
    T("t15", "Who is the chief executive officer of Borealis Systems?", BO, 0, ["lindqvist"]),
    T("t16", "Which product drove Borealis Systems' growth in Q2?", BO, 1, ["helix"]),
    T("t17", "What operating margin did Borealis Systems report for the year?", BO, 1, ["14%"]),
    T("t18", "What was Borealis Systems' annual customer churn rate?", BO, 2, ["3.1"]),
    T("t19", "How many employees does Cedarline Foods have?", CE, 0, ["1,850", "1850"]),
    T("t20", "In which year was Cedarline Foods founded?", CE, 0, ["1978"]),
    T("t21", "What revenue target has Cedarline Foods set for 2026?", CE, 3, ["225"]),
    T("t22", "What dividend per share did the Cedarline Foods board propose?", CE, 3, ["1.10"]),
    T("t23", "How many customers did Delphi Logistics serve at the end of 2025?", DE, 2, ["860"]),
    T("t24", "Which product drove Delphi Logistics' growth in Q3?", DE, 1, ["routeiq"]),
    T("t25", "Where is Delphi Logistics headquartered?", DE, 0, ["chicago"]),
    T("t26", "How large a room can the AirPure 200 cover?", AP, 0, ["35 square"]),
    T("t27", "How often must the AirPure 200 HEPA filter be replaced?", AP, 0, ["6 months", "six months"]),
    T("t28", "What does code F5 mean on the AirPure 200?", AP, 2, ["fan", "blocked"]),
    T("t29", "How much does the AirPure replacement filter cost?", AP, 2, ["$39"]),
    T("t30", "What is the maximum capacity of the KettleMax 1.7?", KE, 0, ["1.7"]),
    T("t31", "How often should the KettleMax be descaled?", KE, 2, ["2 months", "two months"]),
    T("t32", "What does it mean when the kettle light flashes three times?", KE, 2, ["dry-boil", "dry boil"]),
    T("t33", "How much total output power does the SoundBar X deliver?", SB, 0, ["120 watts", "120 w"]),
    T("t34", "Which Bluetooth version does the SoundBar X support?", SB, 0, ["5.3"]),
    T("t35", "How do you reset the SoundBar X to factory settings?", SB, 2, ["volume down"]),
    T("t36", "How many sensors were used in the urban air quality study?", UR, 0, ["64"]),
    T("t37", "Who funded the urban air quality study?", UR, 2, ["clean cities initiative"]),
    T("t38", "What statistical model was used in the urban air quality study?", UR, 2, ["mixed-effects"]),
    T("t39", "How many plots were used in the soil moisture trial?", SO, 0, ["120 plots"]),
    T("t40", "Which statistical test was used to compare irrigation levels?", SO, 2, ["anova"]),
    T("t41", "How many participants were enrolled in the sleep and memory study?", SL, 0, ["240"]),
    T("t42", "Who funded the sleep and memory study?", SL, 2, ["wellness science trust"]),
    T("t43", "How many days of paid annual leave do full-time employees receive?", HB, 0, ["24 days"]),
    T("t44", "How long is the probation period for new employees?", HB, 2, ["3 months", "three months"]),
    T("t45", "What is the annual learning budget per employee?", HB, 3, ["$800"]),
    T("t46", "What is the daily meal allowance in the travel policy?", TR, 0, ["$45"]),
    T("t47", "Above what amount are receipts required for expenses?", TR, 1, ["$25"]),
    T("t48", "Within how many days are approved expense claims reimbursed?", TR, 2, ["14 days"]),
    T("t49", "What is the minimum password length required by the IT security guidelines?", SE, 0, ["14 characters"]),
    T("t50", "How quickly must lost or stolen devices be reported?", SE, 1, ["4 hours"]),
    T("t51", "Which encryption standard must company laptops use?", SE, 1, ["aes-256"]),

    I("i08", "In the Borealis monthly active users chart, in which month did users peak?", BO, 1, ["october"]),
    I("i09", "Which product line has the largest sales in the Cedarline Foods sales chart?", CE, 1, ["dairy"]),
    I("i10", "What share of Cedarline Foods' costs goes to ingredients in the cost breakdown pie chart?", CE, 2, ["40%", "40 percent"]),
    I("i11", "Which region has the lowest on-time delivery rate in the Delphi Logistics chart?", DE, 1, ["middle east"]),
    I("i12", "In the AirPure 200 diagram, which component sits between the fan unit and the air intake?", AP, 1, ["hepa"]),
    I("i13", "In the KettleMax diagram, which part sits directly below the water chamber?", KE, 1, ["heating base"]),
    I("i14", "In the SoundBar X connection diagram, which port is directly below the optical input?", SB, 1, ["usb"]),
    I("i15", "In which year did the urban NO2 concentration chart reach its peak?", UR, 1, ["2019"]),
    I("i16", "Which irrigation level gives the highest maize yield in the soil trial chart?", SO, 1, ["high"]),
    I("i17", "Which group has the highest mean recall score in the sleep study chart?", SL, 1, ["8-hour", "8 hour"]),

    O("o08", "Who painted the Mona Lisa?", "easy"),
    O("o09", "What is the tallest mountain in the world?", "easy"),
    O("o10", "How many players are on a football team on the field?", "easy"),
    O("o11", "What is the chemical symbol for gold?", "easy"),
    O("o12", "What was Borealis Systems' total revenue in 2022?", "hard"),
    O("o13", "Who is the chief financial officer of Cedarline Foods?", "hard"),
    O("o14", "Which airline is the preferred carrier under the travel policy?", "hard"),
    O("o15", "What is the noise level of the AirPure 300?", "hard"),
    O("o16", "How long is the warranty on the SoundBar Y?", "hard"),
]


# ------------------------------------------------------------------ consistency checks
def page_text(src, page):
    p = ALL_DOCS[src][page]
    return (p["title"] + " " + " ".join(p["paragraphs"])).lower()


def check(questions):
    problems = []
    for q in questions:
        if q["kind"] == "offtopic":
            continue
        src, page = q["source"], q["page"]
        if src not in ALL_DOCS or page >= len(ALL_DOCS[src]):
            problems.append(f"{q['id']}: no such page {src} p.{page}")
            continue
        p = ALL_DOCS[src][page]
        text = page_text(src, page)
        if q["kind"] == "text":
            hay = text
        else:
            if "image" not in p:
                problems.append(f"{q['id']}: page has no image")
                continue
            hay = (p["image_facts"] + " " + text).lower()
            # new image questions must be answerable only from the image, not from page text
            if q["id"] >= "i08" and any(k.lower() in text for k in q["answer_contains"]):
                problems.append(f"{q['id']}: answer keyword is also in the page text")
        if not any(k.lower() in hay for k in q["answer_contains"]):
            problems.append(f"{q['id']}: expected answer not found in ground truth {q['answer_contains']}")

    # hard off-topic questions must really be unanswerable from the corpus
    everything = " ".join(page_text(s, i) for s in ALL_DOCS for i in range(len(ALL_DOCS[s])))
    for term in ["chief financial", "airline", "carrier", "airpure 300", "soundbar y"]:
        if term in everything:
            problems.append(f"hard negative term present in corpus: {term!r}")
    if "2022" in " ".join(page_text(BO, i) for i in range(4)):
        problems.append("2022 appears in the Borealis report")
    boiler = " ".join(t for bank in BOILERPLATE.values() for t in bank).lower()
    if any(ch.isdigit() for ch in boiler):
        problems.append("boilerplate contains digits")
    for q in questions:
        for k in q.get("answer_contains", []):
            if k.lower() in boiler:
                problems.append(f"{q['id']}: answer keyword {k!r} appears in boilerplate")
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        problems.append("duplicate question ids")
    return problems


if __name__ == "__main__":
    original_questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    questions = original_questions + NEW_QUESTIONS
    problems = check(questions)
    if problems:
        sys.exit("Consistency problems:\n  " + "\n  ".join(problems))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, pages in ALL_DOCS.items():
        write_pdf(OUT_DIR / name, pages)
    n_pages = sum(len(p) for p in ALL_DOCS.values())
    n_imgs = sum("image" in pg for p in ALL_DOCS.values() for pg in p)
    print(f"Wrote {len(ALL_DOCS)} PDFs ({n_pages} pages, {n_imgs} images) to {OUT_DIR}")

    out = ROOT / "eval" / "questions_large.json"
    out.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    kinds = {k: sum(q["kind"] == k for q in questions) for k in ("text", "image", "offtopic")}
    print(f"Wrote {len(questions)} questions to {out}: {kinds}")
