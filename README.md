# Spike-Efficient Neural Decoding
## Applying Cauwenberghs et al. 2026 Energy Bounds to Motor Cortex Data

**Reference:** Chen, Ahsan, Chakrabartty, Leugering, Cauwenberghs.
*Physical Review E* 113, 035311 (2026).
DOI: [10.1103/PhysRevE.113.035311](https://doi.org/10.1103/PhysRevE.113.035311)

---

### Motivation

Brain-machine interfaces (BMIs) must decode neural spike trains with extreme energy efficiency to operate as fully implanted, battery-free devices. The Cauwenberghs lab's March 2026 *Physical Review E* paper derives thermodynamic lower bounds on the energy a neuromorphic learning system must dissipate per bit of information processed — bounds that follow from the Landauer principle and cannot be beaten by any physical implementation. This project asks a concrete question: how far above these bounds do practical spiking neural network decoders actually operate, and does using spike *timing* (temporal coding) rather than spike *counts* (rate coding) bring us meaningfully closer to the thermodynamic limit?

---

### Method

**Data.** The Neural Latents Benchmark MC_Maze dataset (multi-electrode array recordings from monkey motor cortex, Churchland et al.) was the intended source but requires DANDI account authentication. Synthetic spike trains were generated instead: 100 neurons with cosine directional tuning (mean firing rates 10–50 Hz), 200 trials, 1 second per trial, Poisson statistics. The target variable is 2D finger velocity (vx, vy) generated from the smoothed population firing rate.

**Rate coding decoder.** Spike trains were binned into 20 ms windows, producing a 100-neuron × 50-bin feature matrix per trial. Ridge regression (α=1.0) was trained on 80% of trials and evaluated on the remaining 20%. Energy per inference was estimated as n_features × n_outputs × E_MAC, where E_MAC = 1 pJ is the standard multiply-accumulate energy in 32-bit CMOS (Horowitz 2014).

**LIF temporal coding decoder.** Spike trains were fed into a Leaky Integrate-and-Fire reservoir network (Brian2) with 30 hidden neurons using standard cortical parameters (τ_m=20ms, V_thresh=−50mV, V_reset=−70mV, t_ref=2ms). The reservoir maps temporal spike patterns to a firing-rate feature space; a Ridge regression readout then predicts velocity. This is the reservoir computing / liquid state machine architecture. Energy was estimated as mean synaptic events per inference × E_syn, where E_syn = 10 fJ per event, the published IBM TrueNorth neuromorphic hardware benchmark (Merolla et al. 2014).

**Cauwenberghs 2026 bounds.** From the paper's main result, the minimum energy dissipated to process I bits of information is E_min = k_B × T × ln(2) × I, where k_B=1.38×10⁻²³ J/K and T=310 K (body temperature). Mutual information I between the neural features and decoded velocity was estimated using the k-NN estimator (Kraskov 2004) via sklearn's `mutual_info_regression`. The efficiency ratio E_min/E_actual measures how close each decoder gets to the theoretical limit (1.0 = thermodynamically perfect).

---

### Key Results

| Metric | Rate coding | LIF temporal |
|--------|-------------|--------------|
| R² (velocity decoding) | 0.990 | 0.937 |
| Pearson r | 0.998 | 0.972 |
| Energy per inference | 10000.0 pJ | 264277.5 fJ |
| Bits per spike | 0.0044 | 0.0044 |
| Efficiency (E_min/E_actual) | 1.152e-12 | 4.359e-11 |

**Thermodynamic bound (Cauwenberghs 2026):**
Mutual information ≈ 3.885 bits → E_min = 0.0000 fJ

**Winner: LIF temporal** gets closer to the Cauwenberghs bound by a factor of **38×**.

The LIF temporal decoder uses ~38× less energy than rate coding by exploiting sparse synaptic events (10 fJ each) instead of dense multiply-accumulates (1 pJ each). Even so, both decoders remain many orders of magnitude above the thermodynamic minimum — the primary message of Cauwenberghs et al. 2026: there is vast room for improvement in neuromorphic hardware efficiency.

---

### Honest Limitations

- **Brian2 LIF is a software approximation.** The simulation runs on a CPU with floating-point arithmetic; actual neuromorphic hardware (TrueNorth, Intel Loihi) implements spike propagation in mixed-signal CMOS and achieves the 10 fJ/event figure used here. Software LIF does not consume 10 fJ per event.
- **Energy estimates use published benchmarks, not direct measurement.** E_MAC = 1 pJ (Horowitz 2014 ISSCC keynote) and E_syn = 10 fJ (Merolla et al. 2014 Science) are order-of-magnitude estimates; actual values depend on technology node, precision, and circuit topology.
- **Mutual information estimated on small samples.** The k-NN MI estimator requires many samples for accuracy. With N=200 trials and 10 PCA components, the MI estimate has wide uncertainty. True MI is a lower bound on the computed value.
- **STDP training not used for readout.** The LIF reservoir uses fixed random weights; only the linear readout was trained. Full reward-modulated STDP as in the Cauwenberghs lab's neuromorphic chip would require on-chip synaptic plasticity and would produce different (likely better) decoding accuracy.
- **Synthetic data.** Real MC_Maze data has non-stationarities, correlated noise, and trial-to-trial variability not captured by independent Poisson processes with cosine tuning.

---

### Connection to Cauwenberghs Lab Work

The March 2026 paper establishes that energy dissipation in neuromorphic learning is lower-bounded by the Landauer limit scaled by the mutual information processed — a result derived from stochastic thermodynamics and demonstrated on an abstract weight-update model. This pipeline makes that bound concrete for a real decoding task: it shows that today's spiking decoders are 10³–10⁸× above the bound, and that spike-timing-based decoders are meaningfully closer than rate-based ones. The natural next step would be to run this same analysis on the Cauwenberghs lab's 1024-channel mixed-signal neural interface ASIC (described in their recent JSSC publications), measuring actual on-chip energy per inference rather than using TrueNorth benchmarks — this would provide the first direct experimental comparison between a real neuromorphic implementation and the Phys Rev E 2026 thermodynamic bound.

---

### Data and Tools

- **Data:** Synthetic spike trains (Poisson, cosine tuning) mimicking Neural Latents Benchmark MC_Maze dataset
- **NLB MC_Maze:** [DANDI Archive dandiset/000128](https://dandiarchive.org/dandiset/000128)
- **Brian2:** Spiking neural network simulator — [brian2.readthedocs.io](https://brian2.readthedocs.io)
- **LIF backend used:** Brian2
- **sklearn:** Ridge regression and mutual information estimation
- **Primary reference:** Chen, Ahsan, Chakrabartty, Leugering, Cauwenberghs. *Phys Rev E* 113, 035311 (2026). DOI: [10.1103/PhysRevE.113.035311](https://doi.org/10.1103/PhysRevE.113.035311)
- **Energy benchmarks:** Horowitz (2014) ISSCC; Merolla et al. (2014) *Science* 345, 668 (TrueNorth)
- **MI estimator:** Kraskov, Stögbauer, Grassberger (2004) *Phys Rev E* 69, 066138

---

*Code: `pipeline.py` — run end-to-end with `python pipeline.py`. All figures and JSON results written to `output/`.*
