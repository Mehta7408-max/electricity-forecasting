"""
Build FINAL_REPORT.pdf from the report content + rendered figures using reportlab
(pure-Python, no system deps). Self-contained layout: title page, sections,
tables, and embedded figures scaled to the page width.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, HRFlowable, ListFlowable, ListItem,
)
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).parent
FIG = ROOT / "artifacts" / "figures"
OUT = ROOT / "FINAL_REPORT.pdf"

USABLE_W = A4[0] - 4 * cm  # left+right margins of 2cm each

# ---- styles ---------------------------------------------------------------
ss = getSampleStyleSheet()
NAVY = colors.HexColor("#1f3b66")
ACCENT = colors.HexColor("#2ca02c")
GREY = colors.HexColor("#555555")

styles = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontSize=22, leading=27,
                            textColor=NAVY, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", parent=ss["Normal"], fontSize=13,
                               leading=17, textColor=GREY, alignment=TA_CENTER,
                               spaceAfter=4),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=19,
                         textColor=NAVY, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, leading=16,
                         textColor=NAVY, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14.5,
                          alignment=TA_LEFT, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", parent=ss["BodyText"], fontSize=10, leading=14),
    "caption": ParagraphStyle("caption", parent=ss["Normal"], fontSize=8.5,
                              leading=11, textColor=GREY, alignment=TA_CENTER,
                              spaceBefore=3, spaceAfter=10),
    "quote": ParagraphStyle("quote", parent=ss["BodyText"], fontSize=9.5, leading=14,
                            leftIndent=10, textColor=colors.HexColor("#7a4a00"),
                            backColor=colors.HexColor("#fff6e6"), borderPadding=6,
                            spaceBefore=4, spaceAfter=8),
    "cell": ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.8, leading=11.5),
    "cellh": ParagraphStyle("cellh", parent=ss["Normal"], fontSize=8.8, leading=11.5,
                            textColor=colors.white, fontName="Helvetica-Bold"),
}

story = []


def P(text, style="body"):
    story.append(Paragraph(text, styles[style]))


def H1(text):
    story.append(Spacer(1, 4))
    story.append(Paragraph(text, styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=6))


def H2(text):
    story.append(Paragraph(text, styles["h2"]))


def quote(text):
    story.append(Paragraph(text, styles["quote"]))


def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(t, styles["bullet"]), leftIndent=12) for t in items],
        bulletType="bullet", start="•", leftIndent=14, spaceAfter=6))


def figure(name, caption):
    path = FIG / name
    iw, ih = ImageReader(str(path)).getSize()
    w = USABLE_W
    h = w * ih / iw
    max_h = 10.5 * cm
    if h > max_h:
        h = max_h
        w = h * iw / ih
    img = Image(str(path), width=w, height=h)
    img.hAlign = "CENTER"
    story.append(Spacer(1, 4))
    story.append(img)
    story.append(Paragraph(caption, styles["caption"]))


def table(data, col_widths=None, header=True):
    rows = []
    for r_i, row in enumerate(data):
        style = "cellh" if (header and r_i == 0) else "cell"
        rows.append([Paragraph(str(c), styles[style]) for c in row])
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    ts = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
               ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                [colors.white, colors.HexColor("#f3f6fa")])]
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(Spacer(1, 8))


# ===========================================================================
# TITLE
# ===========================================================================
story.append(Spacer(1, 5 * cm))
P("Heterogeneous Spatio-Temporal GNNs<br/>for Nordic Electricity Price Forecasting", "title")
story.append(HRFlowable(width="60%", thickness=1.4, color=ACCENT, spaceBefore=8, spaceAfter=14))
P("Final Project Report", "subtitle")
P("Multi-area day-ahead electricity price forecasting (DK1, DK2, DE, HYDRO)", "subtitle")
P("ST-HeteroSAGE &mdash; benchmarked against XGBoost and three GNN baselines", "subtitle")
story.append(Spacer(1, 1.5 * cm))
quote("<b>Headline result:</b> ST-HeteroSAGE reaches MAE = 151.1 DKK / "
      "R&sup2; = 0.696 on the DK1+DK2 test set &mdash; 26.5% better than an "
      "XGBoost baseline and 6.7% better than a plain GraphSAGE.")
story.append(PageBreak())

# ===========================================================================
# 1. EXECUTIVE SUMMARY
# ===========================================================================
H1("1. Executive Summary")
P("This project tackles <b>multi-area day-ahead electricity price forecasting</b> "
  "for the Nordic power market. The central idea is to treat the market not as a "
  "set of independent time series, but as a <b>graph</b> &mdash; where prices in "
  "different zones, different hours, and different fuels all influence one another "
  "&mdash; and to learn on that graph with a neural network.")
P("The flagship model, <b>ST-HeteroSAGE</b>, combines a heterogeneous graph (typed "
  "nodes and edges) with a causal temporal convolution. It is the most accurate "
  "model in the study. This report walks through the data, the graph construction, "
  "the model, and then the evidence pillars: <b>accuracy, forecast behaviour, "
  "ablation, interpretability, and robustness</b>.")

# 2. PROBLEM
H1("2. Problem &amp; Research Question")
P("Day-ahead prices are set once per day at gate closure, so a forecaster only has "
  "information known <i>before</i> the auction. Prices across Nordic zones move "
  "together (shared weather, shared transmission), and they are driven by "
  "fundamentals like gas price, carbon price, load, and renewable generation.")
quote("<b>Research question:</b> How can a heterogeneous graph be effectively "
      "constructed and integrated into GNN models to improve the <b>accuracy, "
      "interpretability, and robustness</b> of multi-area day-ahead electricity "
      "price forecasting in the Nordic power market?")

# 3. DATA
H1("3. Data")
table([
    ["Source", "Data", "Cost"],
    ["Energinet / Nord Pool", "DK1, DK2 spot prices, load, renewables", "free"],
    ["Energy-Charts (Fraunhofer ISE)", "DE spot prices", "free"],
    ["Open-Meteo", "temperature, wind, cloud, humidity", "free"],
], col_widths=[5.5 * cm, 8 * cm, 2.5 * cm])
bullets([
    "<b>Coverage:</b> hourly, 2019-12-31 &rarr; 2025-09-30 &mdash; 50,399 hours per zone.",
    "<b>Split:</b> strictly chronological 80 / 10 / 10; test window &asymp; Mar&ndash;Sep 2025.",
    "<b>No leakage:</b> every price lag is &ge; 24 h (known at gate closure); scalers "
    "fit on the training split only.",
])

# 4. GRAPH
H1("4. The Heterogeneous Graph")
P("The graph has <b>2 node types</b> and <b>5 edge relation types</b> spanning four "
  "market zones (DK1, DK2, HYDRO, DE).")
H2("Nodes")
table([
    ["Type", "Count", "Meaning"],
    ["hour", "201,596", "one node per hour per zone (50,399 h &times; 4 zones)"],
    ["market", "4", "one node per area (NordPool, DK1, DK2, DE)"],
], col_widths=[3 * cm, 3 * cm, 10 * cm])
P("Each <b>hour</b> node carries <b>17 features</b>: 5 price-history "
  "(lag_24h/48h/168h, roll24_mean/std), 4 weather (temperature, wind, cloud, "
  "humidity), 4 fundamentals (load_mwh, renewable_mwh, gas_dkk, co2_dkk), and "
  "4 calendar (hour_sin/cos, week_sin/cos).")
H2("Edges")
table([
    ["Relation", "Type", "Meaning"],
    ["lag_to", "temporal", "links an hour to a later hour (24 / 48 / 168 h)"],
    ["co_occurs_with", "spatial", "same-hour price co-movement across zones"],
    ["belongs_to / rev", "membership", "hour &harr; its market node"],
    ["interconnects", "spatial", "market &harr; market (transmission capacity)"],
], col_widths=[4 * cm, 3 * cm, 9 * cm])
quote("<b>Honest design caveat:</b> German prices end 2024-12-31. To stop stale "
      "forward-filled 2025 values leaking into test predictions, DE is deliberately "
      "excluded from hour-level co_occurs_with edges. This is why DE (and the "
      "synthetic HYDRO zone) score poorly later &mdash; they are structurally "
      "de-weighted, not genuinely forecast.")

# 5. ARCHITECTURE
H1("5. ST-HeteroSAGE Architecture")
P("Each spatio-temporal block does <b>spatial message passing first, then a "
  "temporal convolution</b>:")
P("<font face='Courier' size=9>x_dict &rarr; HeteroConv (spatial) &rarr; per-zone "
  "CausalTCN (dilations 1,4,24; kernel 7; RF&asymp;174h) &rarr; BatchNorm + "
  "residual &nbsp;(&times;2 blocks) &rarr; regression head &rarr; price</font>")
bullets([
    "The <b>CausalTCN</b> replaces explicit lag edges with a strictly causal "
    "~174-hour receptive field &mdash; a much richer temporal signal than three "
    "discrete lags.",
    "<b>BatchNorm, no Dropout</b> &mdash; the two interact badly (dropout masking "
    "corrupts BN statistics), so BatchNorm alone does the regularising.",
])

# 6. RESULTS
H1("6. Headline Results &mdash; Model Comparison")
table([
    ["Model", "MAE &darr;", "RMSE &darr;", "R&sup2; &uarr;", "Notes"],
    ["XGBoost (tabular)", "205.62", "296.41", "0.520", "13 engineered features"],
    ["Homogeneous GraphSAGE", "161.91", "212.53", "0.671", "single node type"],
    ["GAT (heterogeneous)", "179.20", "236.41", "0.591", "attention"],
    ["HeteroSAGE", "162.78", "213.32", "0.668", "typed edges, served by API"],
    ["<b>ST-HeteroSAGE &#9733;</b>", "<b>151.08</b>", "<b>204.36</b>", "<b>0.696</b>",
     "CausalTCN + HeteroConv"],
], col_widths=[4.7 * cm, 2 * cm, 2 * cm, 1.8 * cm, 5.5 * cm])
figure("02_model_comparison.png", "Figure 1 — Model comparison across MAE, RMSE, R² and sMAPE (DK1+DK2 test set).")
P("<b>Reading the chart.</b> Lower is better for MAE/RMSE/sMAPE; higher for R&sup2;. "
  "The story is consistent across every panel: the graph models beat the tabular "
  "baseline, and adding the temporal convolution (ST-HeteroSAGE) gives the final "
  "edge. The jump from XGBoost &rarr; graph is large; the jump from plain GraphSAGE "
  "&rarr; ST-HeteroSAGE is smaller but real.")

# 7. TRAINING
H1("7. Training Behaviour")
figure("01_training_history.png", "Figure 2 — Training/validation loss and train MAE over epochs.")
P("Train and validation loss fall together and flatten without diverging &mdash; no "
  "obvious overfitting, consistent with the BatchNorm-only regularisation. The "
  "stored history saved validation MAE/RMSE/R&sup2; as final scalars rather than "
  "per-epoch curves, so the right panel shows per-epoch train MAE with the final "
  "validation MAE as a reference line (final val R&sup2; &asymp; 0.873).")

# 8. FORECAST BEHAVIOUR
H1("8. Day-Ahead Forecast Behaviour")
H2("8.1 Average daily price shape")
figure("03_dayahead_profiles.png", "Figure 3 — Actual vs predicted average price by hour, DK1 and DK2.")
P("Averaged over the test window, the model reproduces the characteristic "
  "<b>twin-peak day</b>: a morning ramp around 05:00&ndash;06:00, the midday "
  "<b>solar dip</b>, and the strong <b>evening peak around 18:00</b>. The main "
  "visible bias is a slight over-prediction during the midday dip.")
H2("8.2 Where error concentrates across the horizon")
figure("04_horizon_mae.png", "Figure 4 — MAE by forecast horizon step (1–24 h ahead).")
P("Error is fairly flat across most of the horizon but <b>spikes at the "
  "17:00&ndash;19:00 steps</b> &mdash; exactly the evening peak, the most volatile "
  "and highest-stakes part of the day.")

# 9. ABLATION
H1("9. Ablation &mdash; What Actually Matters")
figure("05_ablation.png", "Figure 5 — ST-HeteroSAGE ablation: MAE and R² as each component is removed.")
table([
    ["Variant", "MAE", "R&sup2;", "Verdict"],
    ["<b>Full model</b>", "<b>151.1</b>", "<b>0.696</b>", "baseline"],
    ["No spatial (HeteroConv)", "309.0", "&minus;0.255", "catastrophic"],
    ["No TCN (temporal)", "257.3", "0.148", "catastrophic"],
    ["No co-occurrence edges", "175.6", "0.606", "meaningful hit"],
    ["No market context", "163.1", "0.658", "small hit"],
    ["No hydro", "155.5", "0.682", "negligible"],
], col_widths=[6 * cm, 2.5 * cm, 2.5 * cm, 5 * cm])
P("<b>Headline finding:</b> removing either the spatial graph or the temporal "
  "convolution collapses the model &mdash; R&sup2; even goes <i>negative</i> without "
  "spatial message passing. This is the strongest evidence that both halves of "
  "&lsquo;spatio-temporal&rsquo; are essential. Market and hydro components "
  "contribute little and could be trimmed.")

# 10. INTERPRETABILITY
H1("10. Interpretability &mdash; Which Features Drive Price")
figure("06_feature_importance.png", "Figure 6 — ST-HeteroSAGE feature importance; fuel/market features in red.")
table([
    ["Rank", "Feature", "Importance", "Category"],
    ["1", "gas_dkk", "100", "fuel"],
    ["2", "price_lag_24h", "~82", "price history"],
    ["3", "renewable_mwh", "~78", "generation"],
    ["4", "price_lag_48h", "~50", "price history"],
    ["5", "load_mwh", "~46", "load"],
    ["6", "co2_dkk", "~38", "carbon"],
], col_widths=[1.8 * cm, 5 * cm, 3 * cm, 4 * cm])
P("<b>Economic sense check:</b> the single most important input is the "
  "<b>natural gas price</b> &mdash; exactly right for this market, where gas-fired "
  "plants are frequently the marginal (price-setting) generator. Renewable "
  "generation and load follow. The model learned genuine power-market economics, "
  "not just autocorrelation. (There is no oil feature in this market &mdash; gas, "
  "carbon, wind and hydro are the drivers.)")
quote("<b>Caveat on magnitudes:</b> the raw gradient-sensitivity values are all "
      "tiny (~1e-5) and close together, so treat the ranking as the signal, not the "
      "absolute gaps. Error analysis confirms peaks at 17:00&ndash;18:00 and on Mondays.")

# 11. ROBUSTNESS
H1("11. Robustness &mdash; Does It Hold Up Under Stress")
figure("07_robustness.png", "Figure 7 — MAE degradation under noise, price-spike, and feature-dropout perturbations.")
table([
    ["Perturbation", "Worst case tested", "MAE increase", "Verdict"],
    ["Gaussian noise", "30%", "+6.5%", "very robust"],
    ["Price spikes", "5&times;", "+8.0%", "robust"],
    ["Feature dropout", "30%", "+30.3%", "sensitive"],
], col_widths=[4 * cm, 4 * cm, 3.5 * cm, 4 * cm])
P("<b>Interpretation:</b> the model is highly tolerant of input noise and price "
  "spikes &mdash; small, graceful degradation. Its real vulnerability is "
  "<b>missing features</b>: losing 30% of inputs roughly doubles the error. In "
  "deployment terms, noisy data is fine; missing data is the risk &mdash; arguing "
  "for solid upstream data-quality monitoring.")

# 12. LIMITATIONS
H1("12. Limitations &amp; Honest Caveats")
bullets([
    "<b>DE and HYDRO zones are not genuinely forecast</b> (R&sup2; &asymp; 0). DE is "
    "intentionally de-weighted (data ends 2024); HYDRO is synthetic/sparse. All "
    "headline metrics are reported on DK1+DK2 only.",
    "<b>MAPE is unreliable</b> &mdash; prices cross/near zero, so MAPE explodes. "
    "sMAPE and MAE are the meaningful error metrics.",
    "<b>Feature-importance magnitudes are tiny and close</b> &mdash; the ranking is "
    "trustworthy; the absolute spacing is not.",
    "<b>Single random seed (42)</b> &mdash; results aren't averaged over seeds, so "
    "small gaps between close models shouldn't be over-interpreted.",
])

# 13. MLOPS
H1("13. Reproducibility &amp; MLOps")
bullets([
    "<b>Deterministic:</b> torch/np seed 42; XGBoost random_state=42.",
    "<b>Environment:</b> Python 3.11, PyTorch 2.2 (CPU), PyTorch-Geometric &ge; 2.3.",
    "<b>Git-versioned artifacts</b> &mdash; checkpoints, metrics, and scalers committed.",
    "<b>Serving:</b> docker compose brings up MLflow, a FastAPI prediction service, "
    "and a 7-page Streamlit dashboard.",
    "<b>Monitoring:</b> rolling MAE and feature-drift checks via the API.",
])

# 14. CONCLUSION
H1("14. Conclusion")
P("The project answers its research question affirmatively on all three axes:")
bullets([
    "<b>Accuracy</b> &mdash; ST-HeteroSAGE is the best model, &minus;26.5% MAE vs "
    "XGBoost, by combining a heterogeneous graph with a causal temporal convolution.",
    "<b>Interpretability</b> &mdash; feature importance recovers real market "
    "economics (gas price is the top driver), and error analysis localises the hard "
    "cases (evening peak, Mondays).",
    "<b>Robustness</b> &mdash; graceful degradation under noise and price spikes; the "
    "one real sensitivity is to missing features, a data-quality concern rather than "
    "a modelling flaw.",
])
P("The ablation study is the strongest single result: stripping out either the "
  "spatial or the temporal component collapses performance, proving that the "
  "&lsquo;spatio-temporal&rsquo; framing is the source of the gain, not incidental "
  "complexity.")
story.append(Spacer(1, 8))
story.append(HRFlowable(width="100%", thickness=0.6, color=GREY))
P("<font size=8 color='#777777'>All figures generated from the committed JSON "
  "artifacts via render_graphs.py (artifacts/figures/).</font>")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Final Report — Nordic Electricity Price Forecasting")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Wrote", OUT, "-", OUT.stat().st_size, "bytes")
