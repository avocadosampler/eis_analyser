(async function () {
    const IMAGE_SIZE = 224;
    const LABELS = ["Healthy", "Degraded"];
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

    // Load model
    let model;
    try {
        model = await tf.loadLayersModel(MODEL_URL);
        modelStatus.textContent = "Model loaded — ready to diagnose";
    } catch (e) {
        modelStatus.textContent = "Failed to load model. Check the model/ folder.";
        console.error(e);
        return;
    }

    // Drag & drop
    dropZone.addEventListener("click", () => fileInput.click());
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
        if (file && file.type.startsWith("image/")) handleFile(file);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) handleFile(fileInput.files[0]);
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

    // Track last diagnosis for PDF
    let lastDiagnosis = null;

    function reset() {
        preview.hidden = true;
        result.hidden = true;
        compareSection.hidden = true;
        actionBtns.hidden = true;
        thermalHint.hidden = true;
        thermalBtn.hidden = true;
        compareToggle.classList.remove("active");
        thermalBtn.classList.remove("active");
        dropZone.style.display = "";
        fileInput.value = "";
        lastDiagnosis = null;
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

    async function classify(imgEl) {
        const tensor = tf.tidy(() => {
            return tf.browser
                .fromPixels(imgEl)
                .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
                .toFloat()
                .div(255.0)
                .expandDims(0);
        });

        const predictions = await model.predict(tensor).data();
        tensor.dispose();

        const healthyConf = predictions[0];
        const degradedConf = predictions[1];
        const isHealthy = healthyConf >= degradedConf;
        const topConf = Math.max(healthyConf, degradedConf);
        const pct = (topConf * 100).toFixed(1);

        loading.hidden = true;
        result.hidden = false;

        resultLabel.textContent = isHealthy ? "✅ Healthy" : "⚠️ Degraded";
        resultLabel.className = "result-label " + (isHealthy ? "healthy" : "degraded");

        confidenceBar.style.width = pct + "%";
        confidenceBar.style.background = isHealthy ? "#4ade80" : "#f87171";
        confidenceText.textContent = `Confidence: ${pct}%`;

        // Diagnostic engine — map prediction to physics-based insight
        const diagnosis = getDiagnosis(isHealthy, topConf);
        lastDiagnosis = { ...diagnosis, pct, isHealthy };
        renderDiagnosis(diagnosis);

        // Show action buttons
        actionBtns.hidden = false;

        // Set compare overlay source to the uploaded image
        compareUploaded.src = previewImg.src;

        // Show thermal button for Rₛ-related diagnoses (degraded or low-conf healthy)
        const rsRelated = diagnosis.severity === "critical" || diagnosis.severity === "warning" || diagnosis.severity === "watch";
        thermalBtn.hidden = !rsRelated;
        thermalHint.hidden = true;
    }

    /**
     * Diagnostic Engine
     * Maps AI classification to physics-based EIS insights.
     * Returns { label, color, cause, action, severity }
     */
    function getDiagnosis(isHealthy, confidence) {
        const highConf = confidence > 0.8;

        if (isHealthy && highConf) {
            return {
                label: "HEALTHY",
                color: "#4ade80",
                severity: "ok",
                cause: "Low series resistance (Rₛ); high recombination resistance (R_rec). Charge transport is efficient with no signs of layer separation or internal shorts.",
                action: "System Optimal. No intervention required."
            };
        }

        if (isHealthy && !highConf) {
            return {
                label: "HEALTHY (Low Confidence)",
                color: "#facc15",
                severity: "watch",
                cause: "Impedance arcs appear normal, but confidence is below threshold. Possible early-stage right-shift indicating rising series resistance (Rₛ).",
                action: "Schedule Re-test. Monitor for wiring losses or junction box corrosion."
            };
        }

        if (!isHealthy && highConf) {
            return {
                label: "DEGRADATION DETECTED",
                color: "#f87171",
                severity: "critical",
                cause: "High recombination / mid-frequency arc collapse. Elevated contact resistance (Arc 1) and reduced recombination resistance (Arc 2) detected.",
                action: "Inspect cell encapsulation. This pattern usually indicates moisture intrusion in the absorber layer. Check busbar soldering and cable connections."
            };
        }

        // Degraded, low confidence
        return {
            label: "POSSIBLE DEGRADATION",
            color: "#fb923c",
            severity: "warning",
            cause: "Ambiguous impedance signature. Possible global increase in series resistance (Rₛ) or early contact finger failure.",
            action: "Wiring Loss Suspected. Inspect for corrosion in the junction box. Re-test with a cleaner image if available."
        };
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
            doc.setTextColor(lastDiagnosis.isHealthy ? 74 : 248, lastDiagnosis.isHealthy ? 222 : 113, lastDiagnosis.isHealthy ? 128 : 113);
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
