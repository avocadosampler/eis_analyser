(async function () {
    const IMAGE_SIZE = 224;
    const LABELS = ["Diffusion_Limited", "High_Contact_Resistance", "Normal", "Recombination_Loss"];
    const MODEL_URL = "model/model.json";

    // DOM refs
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const previewImg = document.getElementById("previewImg");
    const result = document.getElementById("result");
    const resultLabel = document.getElementById("resultLabel");
    const confidenceBar = document.getElementById("confidenceBar");
    const confidenceText = document.getElementById("confidenceText");
    const detailText = document.getElementById("detailText");
    const loading = document.getElementById("loading");
    const resetBtn = document.getElementById("resetBtn");
    const modelStatus = document.getElementById("modelStatus");

    // Feature DOM refs
    const compareSection = document.getElementById("compareSection");
    const compareSlider = document.getElementById("compareSlider");
    const compareVal = document.getElementById("compareVal");
    const compareRefLayer = document.getElementById("compareRefLayer");
    const compareUploaded = document.getElementById("compareUploaded");
    const compareToggle = document.getElementById("compareToggle");
    const actionBtns = document.getElementById("actionBtns");
    const thermalBtn = document.getElementById("thermalBtn");
    const thermalHint = document.getElementById("thermalHint");
    const pdfBtn = document.getElementById("pdfBtn");
    const csvCanvas = document.getElementById("csvCanvas");
    const csvHint = document.getElementById("csvHint");
    const cameraBtn = document.getElementById("cameraBtn");
    const cameraView = document.getElementById("cameraView");
    const cameraFeed = document.getElementById("cameraFeed");
    const captureBtn = document.getElementById("captureBtn");
    const cancelCameraBtn = document.getElementById("cancelCameraBtn");
    const captureCanvas = document.getElementById("captureCanvas");

    let cameraStream = null;

    // Load model
    let model;
    try {
        model = await tf.loadLayersModel(MODEL_URL);
        modelStatus.textContent = "Model loaded — ready to diagnose";
    } catch (e) {
        console.error("Model load failed:", e.message || e);
        console.error("Stack:", e.stack);
        modelStatus.textContent = "Failed to load model. Check the model/ folder.";
        return;
    }

    // Drag & drop
    dropZone.addEventListener("click", (e) => {
        if (e.target.closest("#cameraBtn")) return;
        fileInput.click();
    });
    dropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") fileInput.click();
    });
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (!file) return;
        if (file.name.toLowerCase().endsWith(".csv")) {
            handleCSV(file);
        } else if (file.type.startsWith("image/")) {
            handleFile(file);
        }
    });
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (!file) return;
        if (file.name.toLowerCase().endsWith(".csv")) {
            handleCSV(file);
        } else {
            handleFile(file);
        }
    });
    resetBtn.addEventListener("click", reset);

    // Compare slider opacity control
    compareSlider.addEventListener("input", () => {
        const v = compareSlider.value;
        compareRefLayer.style.opacity = v / 100;
        compareVal.textContent = v + "%";
    });

    // Compare toggle
    compareToggle.addEventListener("click", () => {
        const showing = !compareSection.hidden;
        compareSection.hidden = showing;
        compareToggle.classList.toggle("active", !showing);
    });

    // Thermal hint toggle
    thermalBtn.addEventListener("click", () => {
        thermalHint.hidden = !thermalHint.hidden;
        thermalBtn.classList.toggle("active", !thermalHint.hidden);
    });

    // PDF report generation
    pdfBtn.addEventListener("click", generatePDF);

    // Camera
    cameraBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        startCamera();
    });
    captureBtn.addEventListener("click", capturePhoto);
    cancelCameraBtn.addEventListener("click", stopCamera);

    // Track last diagnosis for PDF
    let lastDiagnosis = null;

    function reset() {
        preview.hidden = true;
        result.hidden = true;
        compareSection.hidden = true;
        actionBtns.hidden = true;
        thermalHint.hidden = true;
        thermalBtn.hidden = true;
        csvHint.hidden = true;
        compareToggle.classList.remove("active");
        thermalBtn.classList.remove("active");
        dropZone.style.display = "";
        fileInput.value = "";
        lastDiagnosis = null;
        stopCamera();
    }

    function handleFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.onload = () => classify(previewImg);
        };
        reader.readAsDataURL(file);

        dropZone.style.display = "none";
        preview.hidden = false;
        result.hidden = true;
        loading.hidden = false;
    }

    // ----- CSV support -----

    function handleCSV(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = parseCSV(e.target.result);
                renderNyquistToCanvas(data.real, data.imag);
                // Use the canvas as the preview image
                previewImg.src = csvCanvas.toDataURL("image/png");
                previewImg.onload = () => {
                    csvHint.hidden = false;
                    classify(previewImg);
                };
            } catch (err) {
                loading.hidden = true;
                modelStatus.textContent = "CSV error: " + err.message;
                console.error(err);
            }
        };
        reader.readAsText(file);

        dropZone.style.display = "none";
        preview.hidden = false;
        result.hidden = true;
        loading.hidden = false;
    }

    /**
     * Parse CSV with flexible column detection.
     * Accepts common EIS formats:
     *   - Columns named: Z', Z'', Zreal, Zimag, Re(Z), Im(Z), real, imag, etc.
     *   - Or just 2-3 numeric columns (freq, real, imag) or (real, imag)
     * Returns { real: Float64Array, imag: Float64Array } where imag is raw (negative).
     */
    function parseCSV(text) {
        const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
        if (lines.length < 3) throw new Error("CSV too short — need a header + at least 2 data rows.");

        // Detect delimiter
        const delim = lines[0].includes("\t") ? "\t" : ",";
        const header = lines[0].split(delim).map(h => h.trim().toLowerCase().replace(/['"()]/g, ""));
        const rows = lines.slice(1).map(l => l.split(delim).map(v => parseFloat(v.trim())));

        // Find real and imaginary columns by name
        const realAliases = ["z'", "zreal", "z_real", "rez", "re z", "rez", "real", "z_re", "re"];
        const imagAliases = ["z''", "z\"", "zimag", "z_imag", "imz", "im z", "imz", "imag", "z_im", "im", "-z''", "-zimag"];

        let rIdx = header.findIndex(h => realAliases.includes(h));
        let iIdx = header.findIndex(h => imagAliases.includes(h));
        let negateImag = false;

        // Check if the header indicates already-negated imaginary
        if (iIdx >= 0 && header[iIdx].startsWith("-")) {
            negateImag = true; // data is already -Z'', keep as positive for plotting
        }

        // Fallback: if no named columns, assume positional
        if (rIdx < 0 || iIdx < 0) {
            if (header.length >= 3) {
                // Assume: freq, real, imag
                rIdx = 1;
                iIdx = 2;
            } else if (header.length === 2) {
                // Assume: real, imag
                rIdx = 0;
                iIdx = 1;
            } else {
                throw new Error("Cannot detect Z' and Z'' columns. Use headers: Z', Z''");
            }
        }

        const real = new Float64Array(rows.length);
        const imag = new Float64Array(rows.length);
        let validCount = 0;

        for (const row of rows) {
            const r = row[rIdx];
            const im = row[iIdx];
            if (isNaN(r) || isNaN(im)) continue;
            real[validCount] = r;
            imag[validCount] = negateImag ? im : -im; // store as -Z'' (positive for Nyquist)
            validCount++;
        }

        if (validCount < 2) throw new Error("Not enough valid data points in CSV.");

        return {
            real: real.subarray(0, validCount),
            imag: imag.subarray(0, validCount)
        };
    }

    /**
     * Render Nyquist plot to hidden canvas — matches training data style.
     * Black line on white background, no axes, tight crop.
     */
    function renderNyquistToCanvas(real, imag) {
        const size = 400;
        csvCanvas.width = size;
        csvCanvas.height = size;
        const ctx = csvCanvas.getContext("2d");

        // White background
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, size, size);

        // Compute bounds with padding
        let rMin = Infinity, rMax = -Infinity, iMin = Infinity, iMax = -Infinity;
        for (let i = 0; i < real.length; i++) {
            if (real[i] < rMin) rMin = real[i];
            if (real[i] > rMax) rMax = real[i];
            if (imag[i] < iMin) iMin = imag[i];
            if (imag[i] > iMax) iMax = imag[i];
        }

        const rRange = rMax - rMin || 1;
        const iRange = iMax - iMin || 1;
        const pad = 0.08;
        const padR = rRange * pad;
        const padI = iRange * pad;

        function toX(r) { return ((r - rMin + padR) / (rRange + 2 * padR)) * size; }
        function toY(i) { return size - ((i - iMin + padI) / (iRange + 2 * padI)) * size; }

        // Draw line
        ctx.strokeStyle = "black";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(toX(real[0]), toY(imag[0]));
        for (let i = 1; i < real.length; i++) {
            ctx.lineTo(toX(real[i]), toY(imag[i]));
        }
        ctx.stroke();
    }

    // ----- Camera support -----

    async function startCamera() {
        try {
            // Prefer rear camera on mobile
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment" }
            });
            cameraFeed.srcObject = cameraStream;
            dropZone.style.display = "none";
            cameraView.hidden = false;
        } catch (err) {
            modelStatus.textContent = "Camera access denied or unavailable.";
            console.error(err);
        }
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
        }
        cameraFeed.srcObject = null;
        cameraView.hidden = true;
        dropZone.style.display = "";
    }

    function capturePhoto() {
        // Draw current video frame to canvas
        captureCanvas.width = cameraFeed.videoWidth;
        captureCanvas.height = cameraFeed.videoHeight;
        const ctx = captureCanvas.getContext("2d");
        ctx.drawImage(cameraFeed, 0, 0);

        // Stop camera, show preview, classify
        stopCamera();
        previewImg.src = captureCanvas.toDataURL("image/png");
        previewImg.onload = () => classify(previewImg);
        preview.hidden = false;
        result.hidden = true;
        loading.hidden = false;
    }

    async function classify(imgEl) {
        const tensor = tf.tidy(() => {
            return tf.browser
                .fromPixels(imgEl)
                .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
                .toFloat()
                .expandDims(0);
        });

        const predictions = await model.predict(tensor).data();
        tensor.dispose();

        // Find top class and confidence
        let topIdx = 0;
        for (let i = 1; i < predictions.length; i++) {
            if (predictions[i] > predictions[topIdx]) topIdx = i;
        }
        const topClass = LABELS[topIdx];
        const topConf = predictions[topIdx];
        const pct = (topConf * 100).toFixed(1);

        // Build sorted runner-up list for detail
        const ranked = LABELS.map((l, i) => ({ label: l, conf: predictions[i] }))
            .sort((a, b) => b.conf - a.conf);

        loading.hidden = true;
        result.hidden = false;

        const isNormal = topClass === "Normal";
        const icons = {
            Normal: "✅", High_Contact_Resistance: "🔌",
            Recombination_Loss: "⚠️", Diffusion_Limited: "🌊"
        };
        const displayNames = {
            Normal: "Normal", High_Contact_Resistance: "Contact Resistance",
            Recombination_Loss: "Recombination Loss", Diffusion_Limited: "Diffusion Limited"
        };

        resultLabel.textContent = `${icons[topClass]} ${displayNames[topClass]}`;
        resultLabel.className = "result-label " + (isNormal ? "healthy" : "degraded");

        confidenceBar.style.width = pct + "%";
        const barColors = {
            Normal: "#4ade80", High_Contact_Resistance: "#f87171",
            Recombination_Loss: "#fb923c", Diffusion_Limited: "#38bdf8"
        };
        confidenceBar.style.background = barColors[topClass];
        confidenceText.textContent = `Confidence: ${pct}% — Runner-up: ${displayNames[ranked[1].label]} (${(ranked[1].conf * 100).toFixed(1)}%)`;

        // Diagnostic engine
        const diagnosis = getDiagnosis(topClass, topConf);
        lastDiagnosis = { ...diagnosis, pct, topClass };
        renderDiagnosis(diagnosis);

        // Show action buttons
        actionBtns.hidden = false;
        compareUploaded.src = previewImg.src;

        // Show thermal button for Rₛ-related diagnoses
        const rsRelated = topClass === "High_Contact_Resistance" || topClass === "Diffusion_Limited"
            || (topClass === "Normal" && topConf < 0.8);
        thermalBtn.hidden = !rsRelated;
        thermalHint.hidden = true;
    }

    /**
     * Diagnostic Engine — 4-class
     * Maps AI classification to physics-based EIS insights.
     */
    function getDiagnosis(topClass, confidence) {
        const highConf = confidence > 0.7;

        const diagnostics = {
            Normal: {
                high: {
                    label: "NORMAL",
                    color: "#4ade80",
                    severity: "ok",
                    cause: "Low series resistance (Rₛ); high recombination resistance (R_rec). Two well-defined arcs indicate efficient charge transport with no layer separation or internal shorts.",
                    action: "System Optimal. No intervention required."
                },
                low: {
                    label: "LIKELY NORMAL",
                    color: "#facc15",
                    severity: "watch",
                    cause: "Impedance arcs appear normal, but confidence is below threshold. Possible early-stage parameter drift.",
                    action: "Schedule Re-test. Monitor for emerging trends in follow-up measurements."
                }
            },
            High_Contact_Resistance: {
                high: {
                    label: "CONTACT RESISTANCE FAULT",
                    color: "#f87171",
                    severity: "critical",
                    cause: "High-frequency arc growth indicates increased contact resistance (R₁). Likely busbar soldering failure, finger breakage, or cable connector degradation.",
                    action: "Check busbar soldering and cable connections. Inspect cell interconnect ribbons for micro-cracks. Measure contact resistance with a 4-wire probe."
                },
                low: {
                    label: "POSSIBLE CONTACT ISSUE",
                    color: "#fb923c",
                    severity: "warning",
                    cause: "High-frequency arc appears enlarged but classification confidence is moderate. Could indicate early-stage contact degradation or measurement artifact.",
                    action: "Re-test with clean probe contacts. If pattern persists, inspect busbar and cable connections."
                }
            },
            Recombination_Loss: {
                high: {
                    label: "RECOMBINATION LOSS",
                    color: "#f87171",
                    severity: "critical",
                    cause: "Mid-frequency arc collapse — recombination resistance (R_rec) has dropped significantly. Indicates a leaky cell with increased internal recombination, typically from moisture ingress or UV-induced absorber degradation.",
                    action: "Inspect cell encapsulation for delamination or browning. Check for moisture ingress at edge seals. This cell may need replacement if IV curve confirms power loss."
                },
                low: {
                    label: "POSSIBLE RECOMBINATION ISSUE",
                    color: "#fb923c",
                    severity: "warning",
                    cause: "Mid-frequency arc appears reduced but confidence is moderate. Could indicate early material fatigue or a borderline measurement.",
                    action: "Re-test under controlled conditions. Compare with baseline EIS from commissioning. Monitor for progressive R_rec decline."
                }
            },
            Diffusion_Limited: {
                high: {
                    label: "DIFFUSION LIMITATION",
                    color: "#38bdf8",
                    severity: "critical",
                    cause: "Low-frequency Warburg tail detected — charge carrier transport is diffusion-limited. Indicates degraded ion mobility in the active layer, possibly from absorber decomposition or blocked charge extraction pathways.",
                    action: "Inspect active layer for discoloration or delamination. Check hole/electron transport layers for degradation. This pattern often correlates with prolonged thermal or UV stress."
                },
                low: {
                    label: "POSSIBLE DIFFUSION ISSUE",
                    color: "#818cf8",
                    severity: "warning",
                    cause: "Low-frequency tail suggests diffusion limitation but confidence is moderate. Could be a temperature artifact or incomplete low-frequency sweep.",
                    action: "Re-test at controlled temperature. Ensure frequency sweep extends below 0.1 Hz. If tail persists, investigate mass transport path."
                }
            }
        };

        const classEntry = diagnostics[topClass];
        return classEntry ? (highConf ? classEntry.high : classEntry.low) : diagnostics.Normal.low;
    }

    function renderDiagnosis(d) {
        detailText.innerHTML = `
            <div class="diag-card diag-${d.severity}">
                <div class="diag-section">
                    <span class="diag-tag">Physical Cause</span>
                    <p>${d.cause}</p>
                </div>
                <div class="diag-section">
                    <span class="diag-tag">Recommended Action</span>
                    <p>${d.action}</p>
                </div>
            </div>`;
    }

    async function generatePDF() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF({ unit: "mm", format: "a4" });
        const w = doc.internal.pageSize.getWidth();
        const margin = 15;
        let y = 20;

        // Title
        doc.setFontSize(18);
        doc.setTextColor(30, 41, 59);
        doc.text("Cell Diagnostic Report", margin, y);
        y += 8;

        // Timestamp
        doc.setFontSize(9);
        doc.setTextColor(100, 116, 139);
        doc.text("Generated: " + new Date().toLocaleString(), margin, y);
        y += 10;

        // Separator
        doc.setDrawColor(200);
        doc.line(margin, y, w - margin, y);
        y += 8;

        // Embed the uploaded plot image
        try {
            const imgData = previewImg.src;
            const imgW = w - margin * 2;
            const imgH = imgW * (previewImg.naturalHeight / previewImg.naturalWidth);
            doc.addImage(imgData, "PNG", margin, y, imgW, Math.min(imgH, 100));
            y += Math.min(imgH, 100) + 8;
        } catch (_) {
            // skip image if it can't be embedded
            y += 4;
        }

        if (lastDiagnosis) {
            // Classification
            doc.setFontSize(14);
            const pdfColors = {
                Normal: [74, 222, 128], High_Contact_Resistance: [248, 113, 113],
                Recombination_Loss: [251, 146, 60], Diffusion_Limited: [56, 189, 248]
            };
            const c = pdfColors[lastDiagnosis.topClass] || [148, 163, 184];
            doc.setTextColor(c[0], c[1], c[2]);
            doc.text(lastDiagnosis.label + " — " + lastDiagnosis.pct + "% confidence", margin, y);
            y += 10;

            // Cause
            doc.setFontSize(10);
            doc.setTextColor(30, 41, 59);
            doc.setFont(undefined, "bold");
            doc.text("Physical Cause:", margin, y);
            y += 5;
            doc.setFont(undefined, "normal");
            const causeLines = doc.splitTextToSize(lastDiagnosis.cause, w - margin * 2);
            doc.text(causeLines, margin, y);
            y += causeLines.length * 5 + 6;

            // Action
            doc.setFont(undefined, "bold");
            doc.text("Recommended Action:", margin, y);
            y += 5;
            doc.setFont(undefined, "normal");
            const actionLines = doc.splitTextToSize(lastDiagnosis.action, w - margin * 2);
            doc.text(actionLines, margin, y);
            y += actionLines.length * 5 + 10;
        }

        // Footer
        doc.setDrawColor(200);
        doc.line(margin, y, w - margin, y);
        y += 6;
        doc.setFontSize(8);
        doc.setTextColor(148, 163, 184);
        doc.text("EIS Diagnostic Tool — AI-assisted analysis. Verify findings with manual inspection.", margin, y);

        doc.save("Cell_Diagnostic_Report.pdf");
    }
})();
