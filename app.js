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

    function reset() {
        preview.hidden = true;
        result.hidden = true;
        dropZone.style.display = "";
        fileInput.value = "";
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

        detailText.textContent = isHealthy
            ? "Impedance arcs indicate normal contact resistance and recombination behavior."
            : "Elevated contact resistance (Arc 1) and reduced recombination resistance (Arc 2) detected.";
    }
})();
