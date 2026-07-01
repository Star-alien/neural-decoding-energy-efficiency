#!/usr/bin/env python3
"""
generate_report.py — Neural Decoding Energy Efficiency PDF Report
Run after pipeline.py has populated output/.
"""

import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

ROOT   = Path(__file__).parent
OUT    = ROOT / "output"
PDF    = OUT / "neural_decoding_energy_efficiency_report.pdf"

# ── Colours ────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1B2A4A")
TEAL   = colors.HexColor("#2E86AB")
SILVER = colors.HexColor("#E8ECF0")
GOLD   = colors.HexColor("#D4A017")
WHITE  = colors.white
BLACK  = colors.black
DKGRAY = colors.HexColor("#444444")
LTGRAY = colors.HexColor("#888888")

# ── Styles ─────────────────────────────────────────────────────
def make_styles():
    S = {}
    S['cover_title'] = ParagraphStyle('CvT',
        fontName='Helvetica-Bold', fontSize=20,
        textColor=WHITE, alignment=TA_CENTER, leading=26, spaceAfter=8)
    S['cover_sub'] = ParagraphStyle('CvS',
        fontName='Helvetica', fontSize=11,
        textColor=colors.HexColor("#BDD5EA"), alignment=TA_CENTER, spaceAfter=4)
    S['cover_meta'] = ParagraphStyle('CvM',
        fontName='Helvetica', fontSize=9,
        textColor=colors.HexColor("#9BBAD4"), alignment=TA_CENTER, spaceAfter=3)
    S['cover_body'] = ParagraphStyle('CvB',
        fontName='Helvetica', fontSize=9.5,
        textColor=DKGRAY, leading=14, alignment=TA_JUSTIFY, spaceAfter=0)
    S['h1'] = ParagraphStyle('H1',
        fontName='Helvetica-Bold', fontSize=14,
        textColor=NAVY, spaceBefore=14, spaceAfter=6)
    S['h2'] = ParagraphStyle('H2',
        fontName='Helvetica-Bold', fontSize=11,
        textColor=TEAL, spaceBefore=10, spaceAfter=4)
    S['body'] = ParagraphStyle('Body',
        fontName='Helvetica', fontSize=9,
        textColor=BLACK, leading=13, spaceAfter=5, alignment=TA_JUSTIFY)
    S['bullet'] = ParagraphStyle('Bul',
        fontName='Helvetica', fontSize=9,
        textColor=BLACK, leading=13, spaceAfter=4,
        leftIndent=14, firstLineIndent=-10)
    S['caption'] = ParagraphStyle('Cap',
        fontName='Helvetica-Oblique', fontSize=8,
        textColor=LTGRAY, alignment=TA_CENTER, spaceAfter=6)
    S['small'] = ParagraphStyle('Sm',
        fontName='Helvetica', fontSize=8,
        textColor=BLACK, leading=11, spaceAfter=2)
    S['tbl_hdr'] = ParagraphStyle('TH',
        fontName='Helvetica-Bold', fontSize=8.5,
        textColor=WHITE, alignment=TA_CENTER)
    S['tbl_cell'] = ParagraphStyle('TC',
        fontName='Helvetica', fontSize=8.5,
        textColor=BLACK, alignment=TA_LEFT)
    S['tbl_cell_c'] = ParagraphStyle('TCC',
        fontName='Helvetica', fontSize=8.5,
        textColor=BLACK, alignment=TA_CENTER)
    S['ref'] = ParagraphStyle('Ref',
        fontName='Helvetica', fontSize=8,
        textColor=DKGRAY, leading=12, spaceAfter=5,
        leftIndent=18, firstLineIndent=-18)
    return S


class ColorBox(Flowable):
    def __init__(self, w, h, c):
        Flowable.__init__(self)
        self.bw, self.bh, self.bc = w, h, c
    def draw(self):
        self.canv.setFillColor(self.bc)
        self.canv.rect(0, 0, self.bw, self.bh, fill=1, stroke=0)
    def wrap(self, *a):
        return self.bw, self.bh


def embed(path, width=6.2*inch):
    p = Path(path)
    if not p.exists():
        return None
    img = Image(str(p))
    return Image(str(p), width=width, height=width * img.imageHeight / img.imageWidth)


def make_table(rows, cws, hdr_color=NAVY):
    tbl = Table(rows, colWidths=cws)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), hdr_color),
        ('TEXTCOLOR',     (0,0), (-1,0), WHITE),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,0), 8.5),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, SILVER]),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,1), (-1,-1), 8.5),
        ('ALIGN',         (0,0), (-1,-1), 'LEFT'),
        ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('RIGHTPADDING',  (0,0), (-1,-1), 5),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('GRID',          (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ('LINEBELOW',     (0,0), (-1,0),  0.8, NAVY),
    ]))
    return tbl


# ══════════════════════════════════════════════════════════════
# PAGE 1 — Cover
# ══════════════════════════════════════════════════════════════

def page_cover(S, story):
    # Navy banner
    story.append(ColorBox(7.5*inch, 3.1*inch, NAVY))
    story.append(Spacer(1, -3.1*inch))
    story.append(Spacer(1, 0.40*inch))

    story.append(Paragraph(
        "Spike-Efficient Neural Decoding:", S['cover_title']))
    story.append(Paragraph(
        "Applying Cauwenberghs 2026 Thermodynamic Bounds", S['cover_title']))
    story.append(Paragraph(
        "to Motor Cortex Data", S['cover_title']))
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph(
        "Extending Chen et al. 2026, Phys Rev E — Cauwenberghs Lab, UCSD",
        S['cover_sub']))
    story.append(Spacer(1, 0.10*inch))
    story.append(Paragraph("Avani Agarwal · UCSD · July 2026", S['cover_meta']))
    story.append(Paragraph(
        "github.com/Star-alien/neural-decoding-energy-efficiency",
        S['cover_meta']))
    story.append(Paragraph(
        "Anchor paper: DOI 10.1103/PhysRevE.113.035311",
        S['cover_meta']))
    story.append(Spacer(1, 1.48*inch))

    story.append(HRFlowable(width="100%", thickness=2, color=GOLD, spaceAfter=14))

    story.append(Paragraph(
        "Neuromorphic hardware promises orders-of-magnitude energy savings over "
        "conventional AI accelerators, but how close do practical spiking neural "
        "decoders actually get to the fundamental thermodynamic limits? "
        "Chen et al. 2026 (Phys Rev E 113, 035311) established lower bounds on "
        "energy dissipation in neuromorphic learning systems from first principles "
        "of stochastic thermodynamics. This analysis applies those bounds to motor "
        "cortex neural decoding, comparing rate coding and leaky integrate-and-fire "
        "(LIF) temporal coding against the Cauwenberghs thermodynamic minimum to "
        "quantify the remaining efficiency gap and identify which decoder architecture "
        "comes closer to the Landauer limit.",
        S['cover_body']))

    story.append(Spacer(1, 0.22*inch))

    # Summary metric box
    summary = [
        [Paragraph("<b>Decoder</b>", S['tbl_hdr']),
         Paragraph("<b>R²</b>",      S['tbl_hdr']),
         Paragraph("<b>Energy</b>",  S['tbl_hdr']),
         Paragraph("<b>Efficiency (E<sub>min</sub>/E)</b>", S['tbl_hdr'])],
        ["Rate coding (Ridge)", "0.990", "10,000 pJ", "1.15 × 10⁻¹²"],
        ["LIF temporal (Brian2)", "0.937", "264 pJ",    "4.36 × 10⁻¹¹"],
        ["Cauwenberghs 2026 bound", "—", "1.2 × 10⁻⁵ fJ", "1.0 (perfect)"],
    ]
    story.append(make_table(summary, [2.2*inch, 0.9*inch, 1.5*inch, 2.0*inch]))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "LIF temporal coding is 38× closer to the thermodynamic limit than rate "
        "coding. Both remain 10¹⁰–10¹² above the Cauwenberghs bound.",
        S['small']))

    story.append(PageBreak())


# ══════════════════════════════════════════════════════════════
# PAGE 2 — Method Summary
# ══════════════════════════════════════════════════════════════

def page_method(S, story):
    story.append(Paragraph("2. Method Summary", S['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    # 2.1 Data
    story.append(Paragraph("2.1 Data", S['h2']))
    story.append(Paragraph(
        "The intended dataset was the Neural Latents Benchmark MC_Maze recording "
        "(multi-electrode array in macaque motor cortex during reaching tasks; "
        "DANDI Archive dandiset/000128). Download requires account registration; "
        "synthetic spike trains were used as a documented fallback and are clearly "
        "labelled as such throughout.",
        S['body']))
    bullets = [
        "100 neurons, Poisson spike process with cosine directional tuning",
        "Mean firing rates 10–50 Hz (population range); baseline 15 Hz + 25 Hz amplitude",
        "200 trials, 1 second each; bell-shaped speed profile peaking at t=0.5s",
        "2D finger velocity (vx, vy) generated as smoothed population activity",
        "80/20 train/test split (160 train, 40 test trials)",
        "Random seed fixed at 42 for reproducibility",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", S['bullet']))

    # 2.2 Rate coding
    story.append(Paragraph("2.2 Rate Coding Decoder", S['h2']))
    story.append(Paragraph(
        "Spike trains were binned into 20 ms windows, producing a 100-neuron × 50-bin "
        "feature matrix per trial (5,000 features total). Ridge regression (α=1.0, "
        "sklearn) was trained on 160 trials and evaluated on 40 held-out trials. "
        "The target was the mean 2D velocity in the peak-speed window (middle 50% "
        "of each trial).",
        S['body']))
    story.append(Paragraph(
        "Energy estimate: n<sub>features</sub> × n<sub>outputs</sub> × E<sub>MAC</sub> "
        "= 5,000 × 2 × 1 pJ = <b>10,000 pJ per inference</b>, where E<sub>MAC</sub> = "
        "1 pJ is the standard 32-bit multiply-accumulate energy in 45 nm CMOS "
        "(Horowitz 2014 ISSCC keynote).",
        S['body']))

    # 2.3 LIF
    story.append(Paragraph("2.3 LIF Temporal Coding Decoder", S['h2']))
    story.append(Paragraph(
        "Spike trains were fed into a Brian2 leaky integrate-and-fire reservoir network "
        "with 30 hidden neurons. The reservoir uses fixed random weights (reservoir "
        "computing / liquid state machine architecture); only the linear readout layer "
        "is trained. Brian2 was run with the numpy codegen backend (no C++ compiler "
        "required on Windows).",
        S['body']))

    param_data = [
        [Paragraph("<b>Parameter</b>", S['tbl_hdr']),
         Paragraph("<b>Value</b>",     S['tbl_hdr']),
         Paragraph("<b>Parameter</b>", S['tbl_hdr']),
         Paragraph("<b>Value</b>",     S['tbl_hdr'])],
        ["tau_m (membrane time constant)", "20 ms",    "V_thresh (threshold)",  "−50 mV"],
        ["V_reset",                        "−70 mV",   "V_rest (resting)",      "−65 mV"],
        ["t_ref (refractory)",             "2 ms",     "Hidden neurons",        "30"],
        ["Integration timestep",           "0.5 ms",   "Readout",               "Ridge (α=1.0)"],
    ]
    story.append(make_table(param_data, [2.1*inch, 1.1*inch, 2.1*inch, 1.1*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Energy estimate: mean synaptic events per trial × E<sub>syn</sub> "
        "= 26,428 × 10 fJ = <b>264 pJ per inference</b>, where E<sub>syn</sub> = "
        "10 fJ per event is the IBM TrueNorth neuromorphic hardware benchmark "
        "(Merolla et al. 2014).",
        S['body']))

    # 2.4 Bound
    story.append(Paragraph("2.4 Cauwenberghs 2026 Bound Application", S['h2']))
    story.append(Paragraph(
        "The central result of Chen et al. 2026 gives the thermodynamic minimum energy "
        "dissipated per inference as:",
        S['body']))
    story.append(Paragraph(
        "&nbsp;&nbsp;&nbsp;&nbsp;"
        "E<sub>min</sub> = k<sub>B</sub> × T × ln(2) × I",
        S['body']))
    story.append(Paragraph(
        "where k<sub>B</sub> = 1.38 × 10<sup>−23</sup> J/K is Boltzmann's constant, "
        "T = 310 K (physiological body temperature), and I is the mutual information "
        "in bits between the neural input and the decoded output. This bound follows "
        "from the Landauer principle applied to the information-processing operations "
        "in a neuromorphic learning system.",
        S['body']))

    bound_data = [
        [Paragraph("<b>Quantity</b>",  S['tbl_hdr']),
         Paragraph("<b>Value</b>",     S['tbl_hdr']),
         Paragraph("<b>Method</b>",    S['tbl_hdr'])],
        ["Mutual information I", "3.885 bits",
         "sklearn mutual_info_regression (k-NN, top 10 PCA components, n=80 sub-sample)"],
        ["k_B × T × ln(2)", "2.97 × 10⁻²¹ J/bit",
         "Physical constants at T=310 K"],
        ["E_min (Cauwenberghs bound)", "1.2 × 10⁻⁵ fJ",
         "= k_B T ln(2) × I"],
    ]
    story.append(make_table(bound_data, [2.0*inch, 1.5*inch, 3.0*inch]))
    story.append(PageBreak())


# ══════════════════════════════════════════════════════════════
# PAGE 3 — Results
# ══════════════════════════════════════════════════════════════

def page_results(S, story, r):
    story.append(Paragraph("3. Results", S['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    # 3.1 Table
    story.append(Paragraph("3.1 Decoding Performance and Energy Summary", S['h2']))
    rc  = r["rate_coding"]
    lif = r["lif_temporal"]
    cb  = r["cauwenberghs_2026_bound"]

    perf_data = [
        [Paragraph("<b>Metric</b>",    S['tbl_hdr']),
         Paragraph("<b>Rate Coding</b>", S['tbl_hdr']),
         Paragraph("<b>LIF Temporal</b>", S['tbl_hdr']),
         Paragraph("<b>Bound / Notes</b>", S['tbl_hdr'])],
        ["R² (velocity decoding)",
         f"{rc['r_squared']:.3f}", f"{lif['r_squared']:.3f}",
         "Higher = better"],
        ["Pearson r",
         f"{rc['pearson_r']:.3f}", f"{lif['pearson_r']:.3f}",
         "Higher = better"],
        ["Energy per inference",
         f"{rc['energy_pJ_per_inference']:,.0f} pJ",
         f"{lif['energy_fJ_per_inference']:.0f} fJ",
         f"E_min = {cb['E_min_fJ']:.2e} fJ"],
        ["Efficiency (E_min / E)",
         f"{rc['efficiency_vs_cauwenberghs_bound']:.2e}",
         f"{lif['efficiency_vs_cauwenberghs_bound']:.2e}",
         "1.0 = thermodynamic limit"],
        ["Bits per spike",
         f"{rc['bits_per_spike']:.4f}",
         f"{lif['bits_per_spike']:.4f}",
         "MI / mean spike count"],
        ["Mean synaptic events",
         "N/A (dense MAC)",
         f"{lif['n_synaptic_events_mean']:,.0f}",
         "Per trial average"],
        ["Efficiency vs rate coding",
         "1× (baseline)",
         f"{r['comparison']['lif_advantage_factor']:.0f}× closer to bound",
         "LIF wins"],
    ]
    story.append(make_table(perf_data, [2.1*inch, 1.35*inch, 1.35*inch, 1.8*inch]))
    story.append(Spacer(1, 6))

    # 3.2 Figure 2
    story.append(Paragraph("3.2 Decoding Accuracy Comparison", S['h2']))
    fig2 = embed(OUT / "figure2_decoding_comparison.png", width=6.5*inch)
    if fig2:
        story.append(fig2)
        story.append(Paragraph(
            "Figure 1. Predicted vs actual finger velocity for rate coding (left) "
            "and LIF temporal coding (right). Both decoders accurately predict "
            "2D velocity direction. Rate coding achieves slightly higher R² (0.990 "
            "vs 0.937) due to richer feature dimensionality (5,000 binned counts vs "
            "sparse spike timing), but at 38× greater energy cost relative to the "
            "thermodynamic minimum.",
            S['caption']))

    # 3.3 Figure 3
    story.append(Paragraph("3.3 Energy Efficiency Analysis", S['h2']))
    fig3 = embed(OUT / "figure3_energy_comparison.png", width=5.5*inch)
    if fig3:
        story.append(fig3)
        story.append(Paragraph(
            "Figure 2. Energy per inference on a log scale. Three orders of magnitude "
            "separate rate coding (10,000 pJ) from LIF temporal coding (264 pJ). "
            "Both remain 10¹⁰–10¹² above the Cauwenberghs "
            "2026 thermodynamic lower bound (1.2 × 10⁻⁵ fJ), "
            "revealing vast efficiency headroom in future neuromorphic hardware.",
            S['caption']))

    story.append(PageBreak())


# ══════════════════════════════════════════════════════════════
# PAGE 4 — Figure 1 full page + Figure 4
# ══════════════════════════════════════════════════════════════

def page_figures(S, story):
    story.append(Paragraph("3.4 Spike Raster — Representative Trials", S['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    fig1 = embed(OUT / "figure1_spike_raster.png", width=7.0*inch)
    if fig1:
        story.append(fig1)
        story.append(Paragraph(
            "Figure 3. Spike raster for 10 representative neurons across 5 trials. "
            "Coloured rectangles represent 20 ms rate-coding bins (opacity scales with "
            "spike count in each bin and neuron subset). Tick marks show individual "
            "spike times at millisecond resolution. Red/blue traces on secondary axes "
            "show the corresponding finger velocity components (vx, vy). "
            "Bell-shaped speed profiles visible mid-trial reflect stereotyped reaching "
            "movements with cosine directional tuning. The LIF temporal decoder "
            "uses exact spike timing rather than bin counts.",
            S['caption']))

    story.append(Spacer(1, 10))
    story.append(Paragraph("3.5 Efficiency Gap to Thermodynamic Limit", S['h2']))
    fig4 = embed(OUT / "figure4_efficiency_gap.png", width=6.5*inch)
    if fig4:
        story.append(fig4)
        story.append(Paragraph(
            "Figure 4. Efficiency ratio E_min/E_actual as a function of mutual "
            "information for both decoders. The green dashed line at 1.0 is "
            "thermodynamically perfect efficiency. At our operating MI of 3.88 bits "
            "(dotted vertical), LIF temporal coding (red triangle) is 38× "
            "closer to the bound than rate coding (blue square). The mechanism is "
            "architectural: LIF uses sparse 10 fJ synaptic events; rate coding "
            "uses dense 1 pJ multiply-accumulates.",
            S['caption']))

    story.append(PageBreak())


# ══════════════════════════════════════════════════════════════
# PAGE 5 — Limitations and Next Steps
# ══════════════════════════════════════════════════════════════

def page_limitations(S, story):
    story.append(Paragraph("4. Honest Limitations", S['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    lims = [
        ("<b>Synthetic data.</b> Real MC_Maze recordings from the Neural Latents "
         "Benchmark (DANDI dandiset/000128) were the intended dataset. Synthetic "
         "Poisson spike trains were used as a documented fallback. Real motor cortex "
         "data exhibits burst firing, pairwise noise correlations, non-stationary "
         "firing rates across the trial, and population structure (e.g. rotational "
         "dynamics in latent space) that independent Poisson processes do not capture. "
         "Whether LIF maintains its 38x efficiency advantage on real data is the "
         "primary open question — it is possible that temporal coding's advantage "
         "is partly an artefact of the Poisson assumption."),
        ("<b>Software LIF is not hardware neuromorphic.</b> Brian2 simulates LIF "
         "dynamics in 64-bit floating-point arithmetic on a CPU. The 10 fJ per "
         "synaptic-event energy figure comes from published IBM TrueNorth benchmarks "
         "(Merolla et al. 2014), not from direct measurement on the Cauwenberghs "
         "lab's own hardware. The lab's 1024-channel mixed-signal neural interface "
         "SoC (Wang et al. 2026, IEEE TBCAS) would have a different and likely "
         "lower energy per event due to analog sub-threshold circuit operation."),
        ("<b>Mutual information estimate is approximate.</b> sklearn's "
         "mutual_info_regression uses a k-nearest-neighbours estimator (Kraskov "
         "2004) that converges slowly and requires many samples for accuracy. "
         "With 200 trials and a 80-sample sub-sample, the MI estimate of 3.88 bits "
         "carries uncertainty of roughly ±1 bit; this directly propagates into "
         "uncertainty on E_min. A full analysis should report bootstrap confidence "
         "intervals on both MI and E_min."),
        ("<b>STDP training not implemented for the readout.</b> The LIF reservoir "
         "uses fixed random weights; only the linear readout was trained. "
         "Reward-modulated STDP as used in the Cauwenberghs lab's neuromorphic "
         "learning chips would adapt the reservoir weights on-chip, potentially "
         "improving decoding accuracy and altering the energy profile. The current "
         "results represent a fixed-reservoir lower bound on LIF decoder performance."),
        ("<b>Single MI estimate applied to both decoders.</b> Mutual information "
         "was estimated from rate-coded features only (for computational tractability). "
         "The LIF decoder may extract different information from the spike trains; "
         "a per-decoder MI estimate would give a more accurate E_min comparison."),
    ]
    for l in lims:
        story.append(Paragraph(f"• {l}", S['bullet']))

    story.append(Paragraph("4.1 Suggested Next Steps", S['h2']))
    nexts = [
        ("<b>Priority 1 — Replicate on real NLB MC_Maze data.</b> Register at "
         "DANDI Archive (dandiarchive.org/dandiset/000128) and re-run pipeline.py "
         "with real multi-electrode recordings. The key question: does the 38x "
         "LIF efficiency advantage survive under realistic firing statistics?"),
        ("<b>Priority 2 — Measure energy on actual Cauwenberghs lab hardware.</b> "
         "Run the same decoding task on the 1024-channel SoC described in Wang et "
         "al. 2026 (IEEE TBCAS DOI: 10.1109/TBCAS.2026.3524051) and measure "
         "on-chip energy per inference directly rather than using the TrueNorth "
         "benchmark as a proxy."),
        ("<b>Priority 3 — Apply bounds to the full pipeline.</b> This analysis "
         "covers only the decoding stage. Spike sorting, amplification, and "
         "analog-to-digital conversion are the dominant energy consumers in a real "
         "implanted BMI. Applying Cauwenberghs 2026 bounds to the full signal chain "
         "would give a system-level efficiency picture."),
        ("<b>Priority 4 — Bootstrap CI on MI and E_min.</b> Run bootstrap resampling "
         "on the MI estimate to get 95% CIs on both MI and the thermodynamic bound, "
         "so the efficiency gap claim is statistically grounded."),
    ]
    for n in nexts:
        story.append(Paragraph(f"• {n}", S['bullet']))

    story.append(PageBreak())


# ══════════════════════════════════════════════════════════════
# PAGE 6 — References
# ══════════════════════════════════════════════════════════════

def page_references(S, story):
    story.append(Paragraph("5. References", S['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    refs = [
        ("1.", "Chen Z, Ahsan F, Chakrabartty S, Leugering J, Cauwenberghs G. "
               "Estimation of energy-dissipation lower bounds for neuromorphic "
               "learning in memory. "
               "<i>Physical Review E</i>. 2026;113(3-2):035311. "
               "DOI: 10.1103/PhysRevE.113.035311"),
        ("2.", "Wang J, Olajide O, Paul A, et al. A 1024-Channel Hybrid "
               "Voltage/Current-Clamp Neural Interface System-on-Chip with "
               "Dynamic Incremental SAR Acquisition. "
               "<i>IEEE Trans Biomed Circuits Syst</i>. 2026. "
               "DOI: 10.1109/TBCAS.2026.3524051"),
        ("3.", "Gao S, ..., Cauwenberghs G, Farina D, Zhao H. "
               "Wearable technologies for assisted mobility in the real world. "
               "<i>Nature Communications</i>. 2025;16(1):10988."),
        ("4.", "Stimberg M, Brette R, Goodman DFM. Brian 2, an intuitive and "
               "efficient neural simulator. <i>eLife</i>. 2019;8:e47314. "
               "DOI: 10.7554/eLife.47314"),
        ("5.", "Karpowicz BM, et al. Stabilizing brain-computer interfaces through "
               "alignment of latent dynamics. <i>bioRxiv</i>. 2022. "
               "[Neural Latents Benchmark MC_Maze dataset]"),
        ("6.", "Merolla PA, Arthur JV, Alvarez-Icaza R, et al. A million "
               "spiking-neuron integrated circuit with a scalable communication "
               "network and interface. <i>Science</i>. 2014;345(6197):668-673. "
               "[TrueNorth energy benchmark: 10 fJ/synaptic event]"),
        ("7.", "Landauer R. Irreversibility and heat generation in the computing "
               "process. <i>IBM Journal of Research and Development</i>. "
               "1961;5(3):183-191. [Landauer limit underpinning Cauwenberghs bound]"),
        ("8.", "Horowitz M. Computing's energy problem (and what we can do about it). "
               "<i>IEEE ISSCC Digest of Technical Papers</i>. 2014. "
               "[1 pJ/multiply-accumulate CMOS energy estimate]"),
        ("9.", "Kraskov A, Stogbauer H, Grassberger P. Estimating mutual information. "
               "<i>Physical Review E</i>. 2004;69(6):066138. "
               "[k-NN MI estimator used in pipeline]"),
        ("10.","Pedregosa F, et al. Scikit-learn: Machine learning in Python. "
               "<i>Journal of Machine Learning Research</i>. 2011;12:2825-2830."),
    ]
    for num, text in refs:
        story.append(Paragraph(
            f"<b>{num}</b>&nbsp;&nbsp;{text}", S['ref']))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("Generating Neural Decoding Energy Efficiency PDF Report...")

    # Load results
    with open(OUT / "results_summary.json") as f:
        r = json.load(f)

    S = make_styles()

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch,   bottomMargin=0.75*inch,
        title="Spike-Efficient Neural Decoding: Cauwenberghs 2026 Energy Bounds",
        author="Avani Agarwal",
        subject="Neuromorphic decoding energy efficiency — Cauwenberghs Lab UCSD",
    )

    story = []
    page_cover(S, story)
    page_method(S, story)
    page_results(S, story, r)
    page_figures(S, story)
    page_limitations(S, story)
    page_references(S, story)

    def on_page(canvas, doc):
        canvas.saveState()
        w, h = letter
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(LTGRAY)
        canvas.drawString(0.75*inch, 0.45*inch,
            "Spike-Efficient Neural Decoding · Avani Agarwal · UCSD · "
            "DOI: 10.1103/PhysRevE.113.035311")
        canvas.drawRightString(w - 0.75*inch, 0.45*inch,
            f"Page {doc.page}")
        if doc.page > 1:
            canvas.setFont('Helvetica-Bold', 7)
            canvas.setFillColor(NAVY)
            canvas.drawString(0.75*inch, h - 0.50*inch,
                "Neural Decoding Energy Efficiency — Cauwenberghs 2026")
            canvas.setFont('Helvetica', 7)
            canvas.setFillColor(LTGRAY)
            canvas.drawRightString(w - 0.75*inch, h - 0.50*inch,
                "Avani Agarwal · UCSD")
            canvas.setStrokeColor(TEAL)
            canvas.setLineWidth(0.5)
            canvas.line(0.75*inch, h - 0.55*inch, w - 0.75*inch, h - 0.55*inch)
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF written -> {PDF}")


if __name__ == "__main__":
    main()
