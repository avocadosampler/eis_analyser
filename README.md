# ⚡ EIS Diagnostic Tool

A browser-based tool that uses AI to classify solar cell health from Electrochemical Impedance Spectroscopy (EIS) Nyquist plots. Upload an image, import a CSV, or use your device camera — the tool identifies the degradation type and provides actionable diagnostics for field technicians.

## Getting Started

The app runs entirely in the browser. No installation, no backend, no dependencies.

1. Serve the project folder over HTTP (browsers block model loading from `file://` URLs):
   ```bash
   python -m http.server 8000
   ```
2. Open `http://localhost:8000`

For public deployment, upload the project to any static host (GitHub Pages, Netlify, Vercel, etc.).

## How to Use

### Image Upload
Drag and drop a Nyquist plot screenshot into the upload area, or click "browse files" to select one. Works with screenshots from any potentiostat software (Gamry, Biologic, Metrohm, etc.).

### CSV Import
Upload a CSV file containing impedance data. The tool auto-detects common column names:

| Real Part | Imaginary Part |
|-----------|---------------|
| `Z'`      | `Z''`         |
| `Zreal`   | `Zimag`       |
| `Re(Z)`   | `Im(Z)`       |
| `real`    | `imag`        |

If no recognized headers are found, it assumes positional columns (frequency, real, imaginary for 3 columns; real, imaginary for 2). Comma and tab delimiters are both supported. The tool renders the data as a Nyquist plot and classifies it automatically.

### Camera Capture
Tap "📷 Use Camera" to open your device camera (rear-facing by default on mobile). Point it at a Nyquist plot on a screen or printout and tap the capture button. Tap ✕ to cancel and return to the main menu.

## Diagnosis Classes

The AI classifies plots into four states:

| Classification | What It Means | Typical Action |
|---|---|---|
| ✅ Normal | Healthy cell — efficient charge transport | No intervention required |
| 🔌 Contact Resistance | High-frequency arc growth — busbar or connector failure | Check soldering, cables, interconnect ribbons |
| ⚠️ Recombination Loss | Mid-frequency arc collapse — internal recombination | Inspect encapsulation, check for moisture ingress |
| 🌊 Diffusion Limited | Low-frequency Warburg tail — transport bottleneck | Inspect active layer, check transport layers |

Each result includes a confidence score, a runner-up classification, a physics-based explanation of the cause, and a recommended field action. Low-confidence results trigger "re-test" recommendations rather than definitive diagnoses.

## Additional Features

- **Reference Comparison** — Toggle the "🔍 Compare" button to overlay a healthy reference plot on your uploaded image. Use the slider to adjust opacity and visually spot arc deviations.
- **Thermal Advisory** — For resistance-related faults, a "🌡️ Check Hotspots" button appears, advising the use of a thermal camera to scan for localized heating on the cell surface.
- **PDF Report** — Click "📄 PDF Report" to download a `Cell_Diagnostic_Report.pdf` containing the plot image, classification, confidence, physical cause, and recommended action. Useful for maintenance records and audit trails.

## Project Structure

```
├── index.html          # App interface
├── app.js              # Classification logic, diagnostics engine, CSV/camera support
└── model/              # TensorFlow.js model (MobileNetV2, fine-tuned)
    ├── model.json
    └── group1-shard*.bin
```

## Limitations

- The model is trained on synthetic data. Accuracy on real-world instrument output will vary depending on visual similarity to the training set.
- Diagnostics are rule-based, mapped from the AI classification. The model does not extract circuit parameters directly from the impedance data.
- For production use, validate results against known cell conditions and real EIS measurements.

---

## The Physics Behind the Training Data

The model was trained on 4,000 synthetic Nyquist plots generated from equivalent circuit models that represent the electrochemical behaviour of a solar cell.

### Equivalent Circuit

Each simulated cell uses a Randles-type circuit:

```
R₀ — p(R₁, CPE₁) — p(R₂, CPE₂)
```

- **R₀** (series resistance): Represents ohmic losses in wiring, contacts, and the bulk material. Appears as the leftmost intercept on the real axis.
- **R₁ ∥ CPE₁** (high-frequency arc): Models the contact interface — metal-semiconductor junctions at busbars and fingers. A growing arc here indicates contact degradation.
- **R₂ ∥ CPE₂** (mid-frequency arc): Models recombination in the active absorber layer. A collapsing arc indicates increased internal recombination, typically from moisture ingress or UV-induced material fatigue.

For the diffusion-limited class, a Warburg element is appended:

```
R₀ — p(R₁, CPE₁) — p(R₂, CPE₂) — W₁
```

The Warburg impedance produces a characteristic 45° tail at low frequencies, indicating that charge carrier transport is limited by diffusion rather than drift — a sign of degraded ion mobility in the active layer.

### Constant Phase Elements (CPE)

Real electrochemical interfaces rarely behave as ideal capacitors. A CPE accounts for surface roughness, porosity, and non-uniform current distribution:

```
Z_CPE = 1 / (Q × (jω)^α)
```

where Q is the pseudo-capacitance, ω is the angular frequency, and α (0 < α ≤ 1) is the ideality factor. When α = 1, the CPE reduces to a pure capacitor.

### How Classes Are Defined

Each class is generated by sampling circuit parameters from overlapping Gaussian distributions:

| Parameter | Normal | Contact Resistance | Recombination Loss | Diffusion Limited |
|---|---|---|---|---|
| R₀ (Ω) | 10 ± 2 | 14 ± 3 | 12 ± 2.5 | 12 ± 3 |
| R₁ (Ω) | 22 ± 6 | 55 ± 15 | 30 ± 8 | 25 ± 7 |
| R₂ (Ω) | 140 ± 25 | 130 ± 30 | 40 ± 15 | 80 ± 20 |
| Warburg Aᵥ | — | — | — | 50 ± 20 |

The distributions deliberately overlap at their boundaries (e.g., a healthy cell's R₁ can reach ~45Ω while a contact-degraded cell's R₁ starts at ~30Ω). This forces the model to learn subtle shape differences rather than relying on simple thresholds, and produces realistic low-confidence predictions in ambiguous cases.

### Visual Augmentation

To improve robustness against real-world visual variation, each training image is randomly augmented with:

- Line colour variation (black, blue, green, red, grey, cyan)
- Background shade variation (white to light grey)
- Grid lines (40% of images, random style and opacity)
- Axis labels (35% of images, random font size)
- Scatter point markers (50% of images, simulating raw data)
- Sensor noise (60% of images, Gaussian jitter on data points)
- Aspect ratio jitter (equal vs auto)
- Scale padding and DPI variation

This simulates the visual diversity of screenshots from different potentiostat software, printed plots, and photos taken of screens in the field.
