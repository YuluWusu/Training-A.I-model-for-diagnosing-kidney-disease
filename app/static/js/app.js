
(function () {
    "use strict";

    // Chan khoi tao trung neu script bi nap lai
    if (window.__CKD_SCREENING_APP_INITIALIZED) {
        return;
    }

    window.__CKD_SCREENING_APP_INITIALIZED = true;


    // ==========================================
    // CAU HINH
    // ==========================================

    const NUMERIC_FIELDS = new Set([
        "age", "bp", "bgr", "bu", "sc",
        "sod", "pot", "hemo", "pcv", "wc", "rc"
    ]);

    const PAGE_NAMES = {
        dashboard: "Tổng quan",
        analysis: "Dự đoán bệnh thận",
        history: "Lịch sử phân tích"
    };

    const FIELD_LABELS = {
        age: "Tuổi",
        bp: "Huyết áp",
        sg: "Tỷ trọng nước tiểu",
        al: "Albumin nước tiểu",
        su: "Đường trong nước tiểu",
        rbc: "Hồng cầu trong nước tiểu",
        pc: "Tế bào mủ",
        pcc: "Cụm tế bào mủ",
        ba: "Vi khuẩn",
        bgr: "Đường huyết ngẫu nhiên",
        bu: "Ure máu",
        sc: "Creatinine huyết thanh",
        sod: "Natri huyết thanh",
        pot: "Kali huyết thanh",
        hemo: "Hemoglobin",
        pcv: "Thể tích hồng cầu",
        wc: "Bạch cầu",
        rc: "Hồng cầu",
        htn: "Tăng huyết áp",
        dm: "Đái tháo đường",
        cad: "Bệnh động mạch vành",
        appet: "Tình trạng ăn uống",
        pe: "Phù chân",
        ane: "Thiếu máu"
    };

    const WIZARD_HINTS = [
        "Nhập thông tin cơ bản và tiền sử bệnh lý.",
        "Nhập các chỉ số xét nghiệm nước tiểu.",
        "Nhập các chỉ số xét nghiệm máu trước khi phân tích."
    ];


    // ==========================================
    // TRANG THAI UNG DUNG
    // ==========================================

    let currentWizardStep = 0;
    let pendingDeleteId = null;
    let predictionRunning = false;

    let predictionSamples = [];
    let predictionSamplesLoaded = false;
    let predictionSamplesPromise = null;


    const form = document.getElementById("predictionForm");
    const resultSection = document.getElementById("resultSection");
    const resultCard = document.getElementById("resultCard");

    const predictButton = document.getElementById("predictBtn");
    const predictButtonText = document.getElementById("predictBtnText");

    const historyTableBody = document.getElementById("historyTableBody");

    const deleteModal = document.getElementById("deleteModal");
    const confirmDeleteButton = document.getElementById("confirmDeleteBtn");

    const wizardSections = Array.from(
        document.querySelectorAll(".wizard-step")
    );

    const wizardIndicators = document.querySelectorAll(
        "[data-step-indicator]"
    );

    const wizardProgressFill = document.getElementById(
        "wizardProgressFill"
    );

    const wizardStepCounter = document.getElementById(
        "wizardStepCounter"
    );

    const wizardStepHint = document.getElementById(
        "wizardStepHint"
    );

    const prevStepButton = document.getElementById("prevStepBtn");
    const nextStepButton = document.getElementById("nextStepBtn");


    // ==========================================
    // HAM HO TRO
    // ==========================================

    function getElement(id) {
        return document.getElementById(id);
    }

    function setText(id, value) {
        const element = getElement(id);

        if (element) {
            element.textContent = value;
        }
    }

    function setHTML(id, value) {
        const element = getElement(id);

        if (element) {
            element.innerHTML = value;
        }
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatNumber(value, digits = 2) {
        if (
            value === null ||
            value === undefined ||
            value === "" ||
            !Number.isFinite(Number(value))
        ) {
            return "—";
        }

        return new Intl.NumberFormat("vi-VN", {
            minimumFractionDigits: 0,
            maximumFractionDigits: digits
        }).format(Number(value));
    }

    function formatPercent(value, digits = 2) {
        return `${formatNumber(value, digits)}%`;
    }

    function toPercentage(value) {
        if (
            value === undefined ||
            value === null ||
            value === ""
        ) {
            return null;
        }

        const number = Number(
            String(value).replace("%", "").trim()
        );

        if (!Number.isFinite(number)) {
            return null;
        }

        const percentage = number <= 1
            ? number * 100
            : number;

        return Math.min(100, Math.max(0, percentage));
    }

    function formatHistoryProbability(value) {
        const percentage = toPercentage(value);

        return percentage === null
            ? "--"
            : `${percentage.toFixed(2)}%`;
    }

    function formatHistoryDate(value) {
        if (!value) {
            return "--";
        }

        const date = new Date(value);

        if (Number.isNaN(date.getTime())) {
            return escapeHtml(value);
        }

        return date.toLocaleString("vi-VN");
    }

    async function requestJSON(url, options = {}) {
        const response = await fetch(url, options);

        const responseText = await response.text();

        let data;

        try {
            data = responseText
                ? JSON.parse(responseText)
                : {};
        } catch {
            throw new Error(
                `Máy chủ trả về dữ liệu không hợp lệ. HTTP ${response.status}.`
            );
        }

        if (!response.ok) {
            throw new Error(getErrorMessage(data));
        }

        return data;
    }

    function getErrorMessage(data) {
        if (!data) {
            return "Không nhận được phản hồi từ máy chủ.";
        }

        if (Array.isArray(data.detail)) {
            return data.detail.map(item => {
                const field = Array.isArray(item.loc)
                    ? item.loc[item.loc.length - 1]
                    : "Dữ liệu";

                return `${field}: ${item.msg}`;
            }).join("; ");
        }

        if (typeof data.detail === "string") {
            return data.detail;
        }

        if (data.message) {
            return data.message;
        }

        return "Đã xảy ra lỗi khi xử lý dữ liệu.";
    }


    // ==========================================
    // THONG BAO TOAST
    // ==========================================

    function showToast(
        message,
        type = "success",
        title = ""
    ) {
        const container = getElement("toastContainer");

        if (!container) {
            console.log(message);
            return;
        }

        const toast = document.createElement("div");

        toast.className = `toast ${type}`;

        const titles = {
            success: "Thao tác thành công",
            error: "Đã xảy ra lỗi",
            warning: "Thông báo"
        };

        const icons = {
            success: "✓",
            error: "!",
            warning: "!"
        };

        toast.innerHTML = `
            <div class="toast-icon">
                ${icons[type] || "i"}
            </div>

            <div class="toast-content">
                <strong>
                    ${escapeHtml(title || titles[type] || "Thông báo")}
                </strong>

                <p>${escapeHtml(message)}</p>
            </div>

            <button
                type="button"
                class="toast-close"
                aria-label="Đóng thông báo">
                ×
            </button>
        `;

        container.appendChild(toast);

        let timer;

        function removeToast() {
            clearTimeout(timer);
            toast.classList.add("removing");

            setTimeout(() => toast.remove(), 200);
        }

        toast.querySelector(".toast-close").addEventListener(
            "click",
            removeToast
        );

        timer = setTimeout(removeToast, 4500);
    }


    // ==========================================
    // DIEU HUONG TRANG
    // ==========================================

    function showView(viewName) {
        document.querySelectorAll(".view").forEach(view => {
            view.classList.toggle(
                "active",
                view.id === viewName
            );
        });

        document.querySelectorAll("[data-view-target]").forEach(button => {
            button.classList.toggle(
                "active",
                button.dataset.viewTarget === viewName
            );
        });

        const pageName = PAGE_NAMES[viewName] || "Tổng quan";

        setText(
            "pageHeading",
            viewName === "dashboard"
                ? "Tổng quan hệ thống"
                : pageName
        );

        setText("breadcrumbCurrent", pageName);

        if (viewName === "history") {
            loadHistory();
        }

        if (viewName === "analysis") {
            showWizardStep(currentWizardStep);
            loadPredictionSamples();
        }

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    }

    document.querySelectorAll("[data-view-target]").forEach(button => {
        button.addEventListener("click", () => {
            showView(button.dataset.viewTarget);
        });
    });

    const startAnalysisButton = getElement("startAnalysisBtn");

    if (startAnalysisButton) {
        startAnalysisButton.addEventListener("click", () => {
            showView("analysis");
        });
    }

    const modelAnalysisButton = getElement("modelAnalysisBtn");

    if (modelAnalysisButton) {
        modelAnalysisButton.addEventListener("click", () => {
            showView("analysis");
        });
    }


    // ==========================================
    // WIZARD 3 BUOC
    // ==========================================

    function showWizardStep(stepIndex) {
        if (
            stepIndex < 0 ||
            stepIndex >= wizardSections.length
        ) {
            return;
        }

        currentWizardStep = stepIndex;

        wizardSections.forEach((section, index) => {
            section.classList.toggle(
                "active",
                index === currentWizardStep
            );

            section.setAttribute(
                "aria-hidden",
                index !== currentWizardStep
            );
        });

        wizardIndicators.forEach((indicator, index) => {
            indicator.classList.toggle(
                "active",
                index === currentWizardStep
            );

            indicator.classList.toggle(
                "completed",
                index < currentWizardStep
            );
        });

        const progress = wizardSections.length > 1
            ? (
                currentWizardStep /
                (wizardSections.length - 1)
            ) * 100
            : 0;

        if (wizardProgressFill) {
            wizardProgressFill.style.width = `${progress}%`;
        }

        if (wizardStepCounter) {
            wizardStepCounter.textContent =
                `Bước ${currentWizardStep + 1} / ${wizardSections.length}`;
        }

        if (wizardStepHint) {
            wizardStepHint.textContent =
                WIZARD_HINTS[currentWizardStep] || "";
        }

        if (prevStepButton) {
            prevStepButton.classList.toggle(
                "hidden",
                currentWizardStep === 0
            );
        }

        if (nextStepButton) {
            nextStepButton.classList.toggle(
                "hidden",
                currentWizardStep === wizardSections.length - 1
            );
        }

        if (predictButton) {
            predictButton.classList.toggle(
                "hidden",
                currentWizardStep !== wizardSections.length - 1
            );
        }
    }

    function validateCurrentWizardStep() {
        const section = wizardSections[currentWizardStep];

        if (!section) {
            return false;
        }

        const controls = section.querySelectorAll(
            "input, select"
        );

        for (const input of controls) {
            if (!input.checkValidity()) {
                input.reportValidity();
                return false;
            }
        }

        return true;
    }

    function validateAllWizardSteps() {
        for (
            let index = 0;
            index < wizardSections.length;
            index++
        ) {
            const controls = wizardSections[index]
                .querySelectorAll("input, select");

            for (const input of controls) {
                if (!input.checkValidity()) {
                    showWizardStep(index);
                    input.reportValidity();
                    return false;
                }
            }
        }

        return true;
    }

    if (nextStepButton) {
        nextStepButton.addEventListener("click", () => {
            if (!validateCurrentWizardStep()) {
                return;
            }

            showWizardStep(currentWizardStep + 1);

            const progress = document.querySelector(
                ".wizard-progress"
            );

            if (progress) {
                progress.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        });
    }

    if (prevStepButton) {
        prevStepButton.addEventListener("click", () => {
            showWizardStep(currentWizardStep - 1);

            const progress = document.querySelector(
                ".wizard-progress"
            );

            if (progress) {
                progress.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        });
    }


    // ==========================================
    // KIEM TRA BACKEND
    // ==========================================

    async function checkBackend() {
        try {
            const data = await requestJSON("/health", {
                cache: "no-store"
            });

            const modelLoaded =
                data.model_loaded ??
                data.loaded;

            if (modelLoaded === false) {
                throw new Error("Mô hình chưa sẵn sàng.");
            }

            setText("dashboardModelStatus", "Sẵn sàng");
            setText(
                "dashboardModelDescription",
                "Mô hình đã được tải"
            );

            const dot = getElement("sidebarStatusDot");

            if (dot) {
                dot.classList.add("online");
                dot.classList.remove("offline");
            }

            setText(
                "sidebarStatusText",
                "Backend đang hoạt động"
            );

        } catch (error) {
            setText("dashboardModelStatus", "Chưa kết nối");

            setText(
                "dashboardModelDescription",
                "Kiểm tra Backend FastAPI"
            );

            const dot = getElement("sidebarStatusDot");

            if (dot) {
                dot.classList.add("offline");
                dot.classList.remove("online");
            }

            setText(
                "sidebarStatusText",
                "Backend chưa sẵn sàng"
            );

            console.error("Lỗi kiểm tra Backend:", error);
        }
    }


    // ==========================================
    // NAP 10 MAU TU TAP TEST
    // ==========================================

    async function fetchPredictionSamples(force = false) {
        if (predictionSamplesLoaded && !force) {
            return predictionSamples;
        }

        if (predictionSamplesPromise && !force) {
            return predictionSamplesPromise;
        }

        predictionSamplesPromise = (async () => {
            const data = await requestJSON(
                "/dataset/samples",
                {
                    cache: "no-store"
                }
            );

            if (
                data.success !== true ||
                !Array.isArray(data.items)
            ) {
                throw new Error(
                    "API dữ liệu mẫu không trả về danh sách hợp lệ."
                );
            }

            const internalSamples = data.items;
            const externalSamples = Array.isArray(data.external_items)
                ? data.external_items
                : [];

            const ckdCount = internalSamples.filter(
                item => Number(item.actual_class) === 1
            ).length;

            const notCkdCount = internalSamples.filter(
                item => Number(item.actual_class) === 0
            ).length;

            if (
                internalSamples.length !== 10 ||
                ckdCount !== 5 ||
                notCkdCount !== 5
            ) {
                throw new Error(
                    "Số lượng dữ liệu mẫu không đúng. " +
                    `Có bệnh: ${ckdCount}/5; ` +
                    `Không bệnh: ${notCkdCount}/5.`
                );
            }

            const samples = [...externalSamples, ...internalSamples];

            // Kiểm tra đủ và đúng tên 24 thuộc tính theo Feature Contract.
            const expectedFeatures = Object.keys(FIELD_LABELS);

            for (const sample of samples) {
                const features = sample.features;
                const actualFeatures = features && typeof features === "object"
                    ? Object.keys(features)
                    : [];

                const hasAllFeatures = expectedFeatures.every(
                    feature => Object.prototype.hasOwnProperty.call(
                        features || {},
                        feature
                    )
                );

                if (
                    actualFeatures.length !== expectedFeatures.length ||
                    !hasAllFeatures
                ) {
                    throw new Error(
                        `Mẫu ${sample.sample_code || "không tên"} không khớp 24 thuộc tính của mô hình.`
                    );
                }
            }

            predictionSamples = samples;
            predictionSamplesLoaded = true;

            return samples;
        })();

        try {
            return await predictionSamplesPromise;
        } finally {
            predictionSamplesPromise = null;
        }
    }


    // ==========================================
    // HIEN THI HOP CHON 10 MAU
    // ==========================================

    function renderPredictionSampleSelect(samples) {
        const select = getElement("predictionSampleSelect");

        if (!select) {
            return;
        }

        select.innerHTML = "";

        const defaultOption = document.createElement("option");

        defaultOption.value = "";
        defaultOption.textContent =
            "Chọn mẫu (Ngoại vi UCI #857 · Lâm sàng KDIGO · Test nội bộ)...";

        select.appendChild(defaultOption);

        const extGroup = document.createElement("optgroup");
        extGroup.label = "NGOẠI VI — BV ENAM BANGLADESH (UCI #857)";

        const clinicalGroup = document.createElement("optgroup");
        clinicalGroup.label = "LÂM SÀNG & CA CẬN BIÊN (KDIGO / DEMO τ)";

        const ckdGroup = document.createElement("optgroup");
        ckdGroup.label = "TẬP TEST NỘI BỘ — CÓ BỆNH (UCI #336)";

        const notCkdGroup = document.createElement("optgroup");
        notCkdGroup.label = "TẬP TEST NỘI BỘ — KHÔNG BỆNH (UCI #336)";

        samples.forEach((sample, index) => {
            const option = document.createElement("option");

            option.value = String(index);

            option.textContent =
                `${sample.sample_code} - ${sample.actual_label}`;

            if (sample.group === "external_uci857") {
                extGroup.appendChild(option);
            } else if (sample.group === "clinical_stress") {
                clinicalGroup.appendChild(option);
            } else if (Number(sample.actual_class) === 1) {
                ckdGroup.appendChild(option);
            } else {
                notCkdGroup.appendChild(option);
            }
        });

        if (extGroup.children.length > 0) select.appendChild(extGroup);
        if (clinicalGroup.children.length > 0) select.appendChild(clinicalGroup);
        select.appendChild(ckdGroup);
        select.appendChild(notCkdGroup);

        select.disabled = false;
    }


    async function loadPredictionSamples(force = false) {
        const select = getElement("predictionSampleSelect");

        if (!select) {
            console.warn(
                "HTML chưa có predictionSampleSelect."
            );
            return;
        }

        select.disabled = true;

        select.innerHTML = `
            <option value="">Đang tải 10 dữ liệu mẫu...</option>
        `;

        try {
            const samples = await fetchPredictionSamples(force);

            renderPredictionSampleSelect(samples);

            console.log(
                "Đã tải 10 mẫu:",
                samples
            );

        } catch (error) {
            select.innerHTML = `
                <option value="">Không tải được dữ liệu mẫu</option>
            `;

            console.error(
                "Lỗi tải dữ liệu mẫu:",
                error
            );

            showToast(
                error.message,
                "error",
                "Lỗi tải dữ liệu mẫu"
            );

        } finally {
            // Chỉ cho phép chọn khi danh sách mẫu đã tải thành công.
            select.disabled = !predictionSamplesLoaded;
        }
    }


    // ==========================================
    // DIEN DU LIEU MAU VAO FORM
    // ==========================================

    function fillSampleData(sample) {
        if (!sample || typeof sample !== "object") {
            showToast(
                "Dữ liệu mẫu không hợp lệ.",
                "error"
            );
            return;
        }

        Object.entries(sample).forEach(([fieldName, value]) => {
            const input = getElement(fieldName);

            if (!input) {
                return;
            }

            input.value = (
                value === null ||
                value === undefined
            )
                ? ""
                : String(value);
        });

        if (resultSection) {
            resultSection.classList.add("hidden");
        }

        showWizardStep(0);

        const progress = document.querySelector(
            ".wizard-progress"
        );

        if (progress) {
            progress.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }
    }


    // Nut nap mau tren trang du doan
    const loadPredictionSampleButton = getElement(
        "loadPredictionSampleBtn"
    );

    if (loadPredictionSampleButton) {
        loadPredictionSampleButton.addEventListener(
            "click",
            async () => {
                const select = getElement(
                    "predictionSampleSelect"
                );

                if (!select || select.value === "") {
                    showToast(
                        "Bạn hãy chọn một trong 10 bộ dữ liệu mẫu.",
                        "warning",
                        "Chưa chọn dữ liệu"
                    );
                    return;
                }

                try {
                    const samples = await fetchPredictionSamples();

                    const index = Number(select.value);
                    const sample = samples[index];

                    if (!sample) {
                        throw new Error(
                            "Không tìm thấy dữ liệu mẫu đã chọn."
                        );
                    }

                    fillSampleData(sample.features);

                    const actualLabel =
                        Number(sample.actual_class) === 1
                            ? "Có bệnh (CKD)"
                            : "Không bệnh (Not CKD)";

                    const metaBanner = getElement("selectedSampleMeta");
                    if (metaBanner) {
                        const isPos = Number(sample.actual_class) === 1;
                        metaBanner.className = `selected-sample-meta ${isPos ? "meta-ckd" : "meta-not-ckd"}`;
                        metaBanner.innerHTML = `
                            <div class="meta-top">
                                <strong>${escapeHtml(sample.sample_code)}</strong>
                                <span class="meta-badge">${escapeHtml(sample.source_name || "Tập Test độc lập (UCI #336)")}</span>
                                <span class="meta-truth">Nhãn thực tế: <b>${escapeHtml(actualLabel)}</b></span>
                            </div>
                            ${sample.clinical_note ? `<p>${escapeHtml(sample.clinical_note)}</p>` : ""}
                        `;
                    }

                    showToast(
                        `Đã nạp ${sample.sample_code}. Nhãn thực tế: ${actualLabel}.`,
                        "success",
                        "Đã tải dữ liệu mẫu"
                    );

                } catch (error) {
                    showToast(
                        error.message,
                        "error",
                        "Không thể nạp dữ liệu"
                    );
                }
            }
        );
    }


    // ==========================================
    // THU THAP 24 THUOC TINH DAU VAO
    // ==========================================

    function collectPatientData() {
        if (!form) {
            throw new Error(
                "Không tìm thấy biểu mẫu nhập dữ liệu."
            );
        }

        const payload = {};
        let enteredCount = 0;

        form.querySelectorAll("[name]").forEach(input => {
            const name = input.name;
            const rawValue = input.value.trim();

            if (rawValue === "") {
                payload[name] = null;
                return;
            }

            enteredCount++;

            if (NUMERIC_FIELDS.has(name)) {
                const numericValue = Number(rawValue);

                if (
                    !Number.isFinite(numericValue) ||
                    numericValue < 0
                ) {
                    throw new Error(
                        `Giá trị của trường ${name} không hợp lệ.`
                    );
                }

                payload[name] = numericValue;

            } else {
                payload[name] = rawValue;
            }
        });

        if (enteredCount === 0) {
            throw new Error(
                "Bạn cần nhập ít nhất một thông tin để phân tích."
            );
        }

        return payload;
    }


    // ==========================================
    // XAC DINH NHAN DU DOAN
    // ==========================================

    function determinePrediction(data) {
        const predictionValue = data.prediction;

        if (
            predictionValue !== undefined &&
            predictionValue !== null &&
            (
                Number(predictionValue) === 0 ||
                Number(predictionValue) === 1
            )
        ) {
            return Number(predictionValue) === 1;
        }

        const label = String(
            data.prediction_label || ""
        ).toLowerCase().replace(/[\s_-]/g, "");

        if (
            label.includes("notckd") ||
            label.includes("notchronic") ||
            label === "0"
        ) {
            return false;
        }

        if (
            label === "ckd" ||
            label === "1"
        ) {
            return true;
        }

        throw new Error(
            "Không xác định được nhãn dự đoán từ phản hồi API."
        );
    }


    // ==========================================
    // LAY XAC SUAT DU DOAN
    // ==========================================

    function extractProbabilities(data) {
        const probabilities = data.probabilities || {};

        let ckdValue;
        let notCkdValue;

        if (Array.isArray(probabilities)) {
            notCkdValue = probabilities[0];
            ckdValue = probabilities[1];

        } else {
            ckdValue =
                probabilities.ckd ??
                probabilities.CKD ??
                probabilities["1"] ??
                probabilities.ckd_probability ??
                data.ckd_probability;

            notCkdValue =
                probabilities.not_ckd ??
                probabilities.notckd ??
                probabilities.Not_CKD ??
                probabilities["0"] ??
                probabilities.not_ckd_probability ??
                data.not_ckd_probability;
        }

        return {
            ckd: toPercentage(ckdValue),
            notCkd: toPercentage(notCkdValue)
        };
    }


    // ==========================================
    // HIEN THI THANH XAC SUAT
    // ==========================================

    function displayPercentage(value, textId, barId) {
        const textElement = getElement(textId);
        const barElement = getElement(barId);

        if (!textElement || !barElement) {
            return;
        }

        if (value === null) {
            textElement.textContent = "Không có dữ liệu";
            barElement.style.width = "0%";
            return;
        }

        textElement.textContent = `${value.toFixed(2)}%`;
        barElement.style.width = `${value}%`;
    }


    // ==========================================
    // HIEN THI KET QUA DU DOAN
    // ==========================================

    function displayResult(data) {
        const isCkd = determinePrediction(data);
        const probabilities = extractProbabilities(data);

        if (resultCard) {
            resultCard.classList.remove(
                "result-ckd",
                "result-not-ckd"
            );

            resultCard.classList.add(
                isCkd ? "result-ckd" : "result-not-ckd"
            );
        }

        setText(
            "resultIcon",
            isCkd ? "CÓ" : "KHÔNG"
        );

        setText(
            "resultTitle",
            isCkd
                ? "Có dấu hiệu bệnh thận mạn tính"
                : "Không phát hiện dấu hiệu bệnh thận mạn tính"
        );

        setText(
            "resultDescription",
            isCkd
                ? "Mô hình xếp dữ liệu đầu vào vào nhóm có nhãn bệnh trong bộ dữ liệu."
                : "Mô hình xếp dữ liệu đầu vào vào nhóm không có nhãn bệnh trong bộ dữ liệu."
        );

        displayPercentage(
            probabilities.ckd,
            "ckdProbabilityText",
            "ckdProbabilityBar"
        );

        displayPercentage(
            probabilities.notCkd,
            "notCkdProbabilityText",
            "notCkdProbabilityBar"
        );

        if (
            probabilities.ckd === null ||
            probabilities.notCkd === null
        ) {
            setText(
                "probabilityNote",
                "API không cung cấp đầy đủ xác suất cho hai lớp."
            );
        } else {
            setText(
                "probabilityNote",
                "Đây là xác suất dự đoán của mô hình, không phải xác suất chẩn đoán lâm sàng."
            );
        }

        displayClinicalInsight(data.clinical_insight);

        displayCalculation(data.calculation);

        setText(
            "resultTime",
            "Thời gian phân tích: " +
            formatHistoryDate(
                data.created_at || new Date().toISOString()
            )
        );

        if (resultSection) {
            resultSection.classList.remove("hidden");

            resultSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }
    }


    // ==========================================
    // GUI DU LIEU DEN BACKEND
    // ==========================================

    if (form) {
        form.addEventListener("submit", async event => {
            event.preventDefault();

            if (predictionRunning) {
                return;
            }

            if (!validateAllWizardSteps()) {
                return;
            }

            let patientData;

            try {
                patientData = collectPatientData();

            } catch (error) {
                showToast(
                    error.message,
                    "warning",
                    "Kiểm tra dữ liệu"
                );
                return;
            }

            predictionRunning = true;

            if (predictButton) {
                predictButton.disabled = true;
            }

            if (predictButtonText) {
                predictButtonText.textContent =
                    "Đang phân tích...";
            }

            try {
                const data = await requestJSON("/predict", {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify(patientData)
                });

                displayResult(data);

                if (data.history_id !== undefined) {
                    showToast(
                        `Kết quả đã được lưu với mã lượt #${data.history_id}.`,
                        "success",
                        "Dự đoán hoàn tất"
                    );
                }

            } catch (error) {
                showToast(
                    error.message,
                    "error",
                    "Không thể dự đoán"
                );

                console.error("Lỗi dự đoán:", error);

            } finally {
                predictionRunning = false;

                if (predictButton) {
                    predictButton.disabled = false;
                }

                if (predictButtonText) {
                    predictButtonText.textContent =
                        "Dự đoán bệnh thận";
                }
            }
        });
    }


    // ==========================================
    // RESET FORM
    // ==========================================

    if (form) {
        form.addEventListener("reset", () => {
            setTimeout(() => {
                if (resultSection) {
                    resultSection.classList.add("hidden");
                }

                showWizardStep(0);

                const select = getElement(
                    "predictionSampleSelect"
                );

                if (select) {
                    select.value = "";
                }

                const metaBanner = getElement("selectedSampleMeta");
                if (metaBanner) {
                    metaBanner.classList.add("hidden");
                    metaBanner.innerHTML = "";
                }
            }, 0);
        });
    }


    // ==========================================
    // LICH SU PHAN TICH
    // ==========================================

    async function loadHistory() {
        if (!historyTableBody) {
            return;
        }

        const loading = getElement("historyLoading");
        const errorBox = getElement("historyError");

        if (loading) {
            loading.classList.remove("hidden");
        }

        if (errorBox) {
            errorBox.classList.add("hidden");
            errorBox.textContent = "";
        }

        historyTableBody.innerHTML = `
            <tr>
                <td colspan="7" class="history-empty">
                    Đang tải dữ liệu...
                </td>
            </tr>
        `;

        try {
            const data = await requestJSON(
                "/history",
                {
                    cache: "no-store"
                }
            );

            const records = data.items || [];

            setText(
                "historyTotal",
                data.total ?? records.length
            );

            renderHistory(records);

        } catch (error) {
            historyTableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="history-empty">
                        Không thể tải dữ liệu lịch sử.
                    </td>
                </tr>
            `;

            if (errorBox) {
                errorBox.textContent = error.message;
                errorBox.classList.remove("hidden");
            }

        } finally {
            if (loading) {
                loading.classList.add("hidden");
            }
        }
    }


    function renderHistory(records) {
        if (!historyTableBody) {
            return;
        }

        if (!records || records.length === 0) {
            historyTableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="history-empty">
                        Chưa có lần phân tích nào được lưu.
                        Hãy thực hiện dự đoán để tạo lịch sử.
                    </td>
                </tr>
            `;

            return;
        }

        historyTableBody.innerHTML = records.map(record => {
            const isCkd = Number(record.prediction) === 1;

            const resultClass = isCkd
                ? "ckd"
                : "not-ckd";

            const resultLabel = isCkd
                ? "Có bệnh"
                : "Không bệnh";

            return `
                <tr>
                    <td class="history-id">
                        #${escapeHtml(record.id)}
                    </td>

                    <td>
                        ${formatHistoryDate(record.created_at)}
                    </td>

                    <td>
                        <span class="history-result ${resultClass}">
                            ${resultLabel}
                        </span>
                    </td>

                    <td>
                        ${formatHistoryProbability(record.ckd_probability)}
                    </td>

                    <td>
                        ${formatHistoryProbability(record.not_ckd_probability)}
                    </td>

                    <td>
                        ${escapeHtml(record.model_name)}
                    </td>

                    <td>
                        <div class="history-actions">
                            <button
                                type="button"
                                class="history-action-btn"
                                data-history-action="view"
                                data-history-id="${escapeHtml(record.id)}">
                                Chi tiết
                            </button>

                            <button
                                type="button"
                                class="history-action-btn delete"
                                data-history-action="delete"
                                data-history-id="${escapeHtml(record.id)}">
                                Xóa
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }).join("");
    }


    // ==========================================
    // XEM CHI TIET LICH SU
    // ==========================================

    async function openHistoryDetail(recordId) {
        const detailPanel = getElement("historyDetail");

        if (!detailPanel) {
            return;
        }

        try {
            const data = await requestJSON(
                `/history/${encodeURIComponent(recordId)}`,
                {
                    cache: "no-store"
                }
            );

            const record = data.item;

            if (!record) {
                throw new Error(
                    "Không tìm thấy dữ liệu chi tiết."
                );
            }

            setText(
                "detailRecordId",
                `#${record.id}`
            );

            setText(
                "detailCreatedAt",
                formatHistoryDate(record.created_at)
            );

            setText(
                "detailPrediction",
                Number(record.prediction) === 1
                    ? "Có bệnh"
                    : "Không bệnh"
            );

            setText(
                "detailCkdProbability",
                formatHistoryProbability(record.ckd_probability)
            );

            setText(
                "detailNotCkdProbability",
                formatHistoryProbability(record.not_ckd_probability)
            );

            let inputData = record.input_data || {};

            if (typeof inputData === "string") {
                try {
                    inputData = JSON.parse(inputData);
                } catch {
                    inputData = {};
                }
            }

            const detailGrid = getElement("detailInputGrid");

            if (detailGrid) {
                detailGrid.innerHTML = Object.entries(
                    FIELD_LABELS
                ).map(([key, label]) => {
                    const value = inputData[key];

                    const displayValue =
                        value === null ||
                            value === undefined ||
                            value === ""
                            ? "Không có dữ liệu"
                            : escapeHtml(value);

                    return `
                        <div class="detail-input-item">
                            <span>${escapeHtml(label)} (${key})</span>
                            <strong>${displayValue}</strong>
                        </div>
                    `;
                }).join("");
            }

            detailPanel.classList.remove("hidden");

            detailPanel.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        } catch (error) {
            showToast(
                error.message,
                "error",
                "Không thể tải chi tiết"
            );
        }
    }


    // ==========================================
    // MODAL XAC NHAN XOA
    // ==========================================

    function openDeleteModal(recordId) {
        if (!deleteModal) {
            return;
        }

        pendingDeleteId = recordId;

        setText(
            "deleteModalRecord",
            `Mã lượt phân tích: #${recordId}`
        );

        deleteModal.classList.remove("hidden");

        deleteModal.setAttribute(
            "aria-hidden",
            "false"
        );

        document.body.style.overflow = "hidden";

        if (confirmDeleteButton) {
            confirmDeleteButton.disabled = false;

            confirmDeleteButton.textContent =
                "Xóa lượt phân tích";

            confirmDeleteButton.focus();
        }
    }


    function closeDeleteModal() {
        if (!deleteModal) {
            return;
        }

        deleteModal.classList.add("hidden");

        deleteModal.setAttribute(
            "aria-hidden",
            "true"
        );

        document.body.style.overflow = "";

        pendingDeleteId = null;

        if (confirmDeleteButton) {
            confirmDeleteButton.disabled = false;

            confirmDeleteButton.textContent =
                "Xóa lượt phân tích";
        }
    }


    const cancelDeleteButton = getElement("cancelDeleteBtn");

    if (cancelDeleteButton) {
        cancelDeleteButton.addEventListener(
            "click",
            closeDeleteModal
        );
    }

    if (deleteModal) {
        deleteModal.addEventListener("click", event => {
            if (event.target === deleteModal) {
                closeDeleteModal();
            }
        });
    }

    document.addEventListener("keydown", event => {
        if (
            event.key === "Escape" &&
            deleteModal &&
            !deleteModal.classList.contains("hidden")
        ) {
            closeDeleteModal();
        }
    });


    if (confirmDeleteButton) {
        confirmDeleteButton.addEventListener(
            "click",
            async () => {
                if (pendingDeleteId === null) {
                    return;
                }

                const recordId = pendingDeleteId;

                confirmDeleteButton.disabled = true;

                confirmDeleteButton.textContent =
                    "Đang xóa...";

                try {
                    await requestJSON(
                        `/history/${encodeURIComponent(recordId)}`,
                        {
                            method: "DELETE"
                        }
                    );

                    const detailId = getElement(
                        "detailRecordId"
                    );

                    if (
                        detailId &&
                        detailId.textContent === `#${recordId}`
                    ) {
                        const detailPanel = getElement(
                            "historyDetail"
                        );

                        if (detailPanel) {
                            detailPanel.classList.add("hidden");
                        }
                    }

                    closeDeleteModal();

                    await loadHistory();

                    showToast(
                        `Lượt phân tích #${recordId} đã được xóa.`,
                        "success",
                        "Đã xóa dữ liệu"
                    );

                } catch (error) {
                    closeDeleteModal();

                    showToast(
                        error.message,
                        "error",
                        "Không thể xóa dữ liệu"
                    );
                }
            }
        );
    }


    // ==========================================
    // SU KIEN BANG LICH SU
    // ==========================================

    if (historyTableBody) {
        historyTableBody.addEventListener("click", event => {
            const button = event.target.closest(
                "[data-history-action]"
            );

            if (!button) {
                return;
            }

            const recordId = button.dataset.historyId;
            const action = button.dataset.historyAction;

            if (action === "view") {
                openHistoryDetail(recordId);
            }

            if (action === "delete") {
                openDeleteModal(recordId);
            }
        });
    }

    // Nut dong chi tiet.
    const closeHistoryDetailButton = getElement(
        "closeHistoryDetailBtn"
    );

    if (closeHistoryDetailButton) {
        closeHistoryDetailButton.addEventListener(
            "click",
            () => {
                const detailPanel = getElement("historyDetail");

                if (detailPanel) {
                    detailPanel.classList.add("hidden");
                }
            }
        );
    }


    // ==========================================
    // DIEN GIAI: NGUONG TAU, CHI SO THIEU, THAM CHIEU Y KHOA
    // ==========================================

    function displayClinicalInsight(insight) {
        const panel = getElement("insightPanel");
        if (!panel) {
            return;
        }
        if (!insight) {
            panel.classList.add("hidden");
            return;
        }
        panel.classList.remove("hidden");

        // ---- 1. Che do nguong tau (tinh phia client tu xac suat) ----
        const thr = insight.threshold || {};
        const ckdProb = Number(thr.ckd_probability);

        const renderVerdict = tau => {
            const positive = ckdProb >= tau;
            const verdict = getElement("thresholdVerdict");
            if (verdict) {
                verdict.className =
                    "threshold-verdict " + (positive ? "verdict-ckd" : "verdict-not-ckd");
                verdict.innerHTML = `
                    <strong>${positive ? "CẢNH BÁO: nghi ngờ CKD" : "Chưa phát hiện dấu hiệu CKD"}</strong>
                    <span>P(CKD) = ${formatPercent(ckdProb * 100, 2)}
                    ${positive ? "≥" : "<"} ngưỡng τ = ${String(tau).replace(".", ",")}
                    ${positive && tau < 0.5 && ckdProb < 0.5
                        ? " → khuyến nghị xét nghiệm lại / khám chuyên khoa"
                        : ""}</span>`;
            }
            document.querySelectorAll(".threshold-btn").forEach(btn => {
                btn.classList.toggle("active", Number(btn.dataset.tau) === tau);
            });
        };

        document.querySelectorAll(".threshold-btn").forEach(btn => {
            btn.onclick = () => renderVerdict(Number(btn.dataset.tau));
        });
        renderVerdict(0.5);
        setText("thresholdNote", thr.note || "");

        // ---- 2. Chi so thieu ----
        const missing = insight.missing || { count: 0, total: 24, items: [], warnings: [] };
        let missingHtml = `
            <div class="completeness-bar">
                <div class="completeness-fill" style="width:${((missing.total - missing.count) / missing.total) * 100}%"></div>
            </div>
            <p class="completeness-label">
                Đã nhập <strong>${missing.total - missing.count}/${missing.total}</strong> chỉ số
                ${missing.count === 0
                    ? "– dữ liệu đầy đủ, không cần điền khuyết."
                    : `– <strong>${missing.count}</strong> chỉ số bị thiếu được Pipeline tự điền bằng thống kê học từ tập Train (không dùng nhãn):`}
            </p>`;

        if (missing.count > 0) {
            missingHtml += `<div class="missing-chip-list">` + missing.items.map(item => `
                <span class="missing-chip ${item.is_core ? "core" : ""}">
                    ${escapeHtml(item.label)}
                    <em>→ ${escapeHtml(item.imputed_value)} (${escapeHtml(item.method)})</em>
                </span>`).join("") + `</div>`;
        }
        (missing.warnings || []).forEach(text => {
            missingHtml += `<div class="insight-warning">⚠ ${escapeHtml(text)}</div>`;
        });
        setHTML("missingSummary", missingHtml);

        // ---- 3. Tham chieu y khoa ----
        const flags = insight.reference_flags || [];
        if (flags.length === 0) {
            setHTML(
                "referenceFlags",
                `<p class="completeness-label">Không có chỉ số nào nằm ngoài ngưỡng tham chiếu trong các giá trị đã nhập.</p>`
            );
        } else {
            setHTML(
                "referenceFlags",
                `<div class="reference-list">` + flags.map(flag => `
                    <div class="reference-item ${escapeHtml(flag.status)}">
                        <div>
                            <strong>${escapeHtml(flag.label)}</strong>
                            <small>${escapeHtml(flag.note)}</small>
                        </div>
                        <div class="reference-values">
                            <span class="reference-actual">${escapeHtml(flag.value)} ${escapeHtml(flag.unit)}</span>
                            <small>Tham chiếu: ${escapeHtml(flag.reference)}</small>
                        </div>
                    </div>`).join("") + `</div>
                    <p class="insight-note">Ngưỡng tham chiếu mang tính khái quát cho người trưởng thành; đối chiếu với biểu đồ SHAP bên dưới để thấy chỉ số bất thường nào thực sự kéo điểm dự đoán của mô hình.</p>`
            );
        }
    }


    function renderShapWaterfall(baseValue, rawScore, probability, contributions) {
        const container = getElement("shapWaterfallChart");
        if (!container || !Array.isArray(contributions) || contributions.length === 0) {
            if (container) container.innerHTML = "";
            return;
        }

        const topN = 9;
        const sorted = [...contributions].sort(
            (a, b) => Math.abs(Number(b.contribution)) - Math.abs(Number(a.contribution))
        );
        const topItems = sorted.slice(0, topN);
        const restItems = sorted.slice(topN);

        const steps = topItems.map(item => {
            const valStr =
                item.input_value === null || item.input_value === undefined
                    ? "thiếu"
                    : String(item.input_value);
            return {
                label: `${item.label || item.feature} = ${valStr}`,
                impact: Number(item.contribution),
                isRest: false
            };
        });

        if (restItems.length > 0) {
            const restSum = restItems.reduce((acc, it) => acc + Number(it.contribution), 0);
            steps.push({
                label: `${restItems.length} đặc trưng còn lại`,
                impact: restSum,
                isRest: true
            });
        }

        // Compute cumulative waterfall coordinates from bottom (baseValue) to top (rawScore)
        // Standard SHAP waterfall renders largest contributors at the top, accumulating from baseValue at bottom.
        const orderedBottomUp = [...steps].reverse();
        let running = baseValue;
        const bars = orderedBottomUp.map(step => {
            const start = running;
            const end = running + step.impact;
            running = end;
            return { ...step, start, end };
        });
        const displayRows = [...bars].reverse();

        const allVals = [baseValue, rawScore];
        displayRows.forEach(b => {
            allVals.push(b.start, b.end);
        });
        const minVal = Math.min(...allVals);
        const maxVal = Math.max(...allVals);
        const span = Math.max(maxVal - minVal, 0.25);
        const padVal = span * 0.16;
        const domainMin = minVal - padVal;
        const domainMax = maxVal + padVal;

        const svgWidth = 760;
        const marginLeft = 240;
        const marginRight = 78;
        const marginTop = 44;
        const marginBottom = 42;
        const rowHeight = 31;
        const barHeight = 19;
        const plotWidth = svgWidth - marginLeft - marginRight;
        const svgHeight = marginTop + displayRows.length * rowHeight + marginBottom;

        const xOf = v => marginLeft + ((v - domainMin) / (domainMax - domainMin)) * plotWidth;
        const xBase = xOf(baseValue);
        const xFinal = xOf(rawScore);

        let rowsSvg = "";
        displayRows.forEach((row, idx) => {
            const yCenter = marginTop + idx * rowHeight + rowHeight / 2;
            const yTop = yCenter - barHeight / 2;
            const x1 = xOf(row.start);
            const x2 = xOf(row.end);
            const isPos = row.impact >= 0;
            const fill = row.isRest
                ? (isPos ? "#e09f9c" : "#8fbdb6")
                : (isPos ? "#c65d58" : "#267d73");
            const rectX = Math.min(x1, x2);
            const rectW = Math.max(Math.abs(x2 - x1), 2.5);

            // Subtle row guide line
            rowsSvg += `<line x1="${marginLeft}" y1="${yCenter}" x2="${svgWidth - marginRight}" y2="${yCenter}" stroke="#edf2f0" stroke-width="1" />`;

            // Connector dotted line to next step below
            if (idx < displayRows.length - 1) {
                const nextYCenter = marginTop + (idx + 1) * rowHeight + rowHeight / 2;
                rowsSvg += `<line x1="${x1}" y1="${yTop + barHeight}" x2="${x1}" y2="${nextYCenter - barHeight / 2}" stroke="#9fb0ad" stroke-width="1" stroke-dasharray="2.5,2.5" />`;
            }

            // Waterfall bar
            rowsSvg += `<rect x="${rectX.toFixed(1)}" y="${yTop.toFixed(1)}" width="${rectW.toFixed(1)}" height="${barHeight}" rx="3.5" fill="${fill}" opacity="0.93" />`;

            // Feature label on left
            const safeLabel = escapeHtml(row.label.length > 32 ? row.label.slice(0, 30) + "…" : row.label);
            rowsSvg += `<text x="${marginLeft - 10}" y="${yCenter + 4}" text-anchor="end" font-size="11.5" font-weight="${row.isRest ? "500" : "600"}" fill="${row.isRest ? "#6b7d7a" : "#203a40"}">${safeLabel}</text>`;

            // Impact value label next to bar
            const impactText = `${isPos ? "+" : ""}${formatNumber(row.impact, 3)}`;
            const textOnRight = rectX + rectW + 46 < svgWidth - 6;
            const labelX = textOnRight ? rectX + rectW + 6 : rectX - 6;
            const anchor = textOnRight ? "start" : "end";
            rowsSvg += `<text x="${labelX.toFixed(1)}" y="${yCenter + 4}" text-anchor="${anchor}" font-size="11" font-weight="700" fill="${isPos ? "#a94440" : "#1b665f"}">${impactText}</text>`;
        });

        const yPlotTop = marginTop - 6;
        const yPlotBottom = svgHeight - marginBottom + 6;
        const probText = formatPercent(probability * 100, 2);

        const svg = `
            <svg class="shap-waterfall-svg" viewBox="0 0 ${svgWidth} ${svgHeight}" role="img" aria-label="Biểu đồ SHAP Waterfall giải thích dự đoán">
                <line x1="${xBase.toFixed(1)}" y1="${yPlotTop}" x2="${xBase.toFixed(1)}" y2="${yPlotBottom}" stroke="#839592" stroke-width="1.3" stroke-dasharray="4,3" />
                <line x1="${xFinal.toFixed(1)}" y1="${yPlotTop}" x2="${xFinal.toFixed(1)}" y2="${yPlotBottom}" stroke="#203a40" stroke-width="1.6" stroke-dasharray="4,3" />
                ${rowsSvg}
                <text x="${xFinal.toFixed(1)}" y="18" text-anchor="middle" font-size="11.5" font-weight="700" fill="#203a40">
                    f(x) = ${formatNumber(rawScore, 3)} (${probText} CKD)
                </text>
                <text x="${xBase.toFixed(1)}" y="${svgHeight - 14}" text-anchor="middle" font-size="11" font-weight="600" fill="#647774">
                    E[f(x)] = ${formatNumber(baseValue, 3)} (Mức nền)
                </text>
            </svg>
        `;

        container.innerHTML = `
            <div class="shap-waterfall-header">
                <div>
                    <h4>Biểu đồ SHAP Waterfall (Thác nước cộng dồn thời gian thực)</h4>
                    <p>Cộng dồn từ điểm nền E[f(x)] đến điểm dự đoán f(x) của bệnh nhân hiện tại</p>
                </div>
            </div>
            ${svg}
        `;
    }

    // ==========================================
    // GIAI THICH KET QUA TREE SHAP
    // ==========================================

    function displayCalculation(calculation) {
        const panel = getElement("calculationPanel");

        if (!panel) {
            return;
        }

        if (
            !calculation ||
            !Array.isArray(calculation.contributions)
        ) {
            panel.classList.add("hidden");
            console.warn(
                "Backend chưa trả về dữ liệu giải thích TreeSHAP cho lượt dự đoán này."
            );
            return;
        }

        panel.classList.remove("hidden");

        const baseValue = Number(calculation.base_value);

        const contributionSum = Number(
            calculation.contribution_sum
        );

        const rawScore = Number(calculation.raw_score);

        const probability = Number(
            calculation.reconstructed_ckd_probability
        );

        setText(
            "calculationBaseValue",
            formatNumber(baseValue, 5)
        );

        setText(
            "calculationContributionSum",
            formatNumber(contributionSum, 5)
        );

        setText(
            "calculationRawScore",
            formatNumber(rawScore, 5)
        );

        setText(
            "calculationProbability",
            formatPercent(probability * 100, 2)
        );


        // Cong diem SHAP
        setHTML(
            "calculationEquation",
            `
                <p>
                    <strong>
                        Bước 1 – Cộng điểm đóng góp của các thuộc tính:
                    </strong>
                </p>

                <code>
                    F(x) = F₀ + Σφᵢ
                    = ${formatNumber(baseValue, 5)}
                    + (${formatNumber(contributionSum, 5)})
                    = ${formatNumber(rawScore, 5)}
                </code>

                <p>
                    <strong>
                        Bước 2 – Chuyển điểm đầu ra thành xác suất:
                    </strong>
                </p>

                <code>
                    P(Có bệnh) = 1 / (1 + e<sup>−F(x)</sup>)
                    = ${formatPercent(probability * 100, 2)}
                </code>
            `
        );

        renderShapWaterfall(baseValue, rawScore, probability, calculation.contributions);

        // Dong gop tung thuoc tinh
        const contributions = calculation.contributions;

        const maxImpact = Math.max(
            ...contributions.map(item =>
                Math.abs(Number(item.contribution))
            ),
            0.000001
        );

        setHTML(
            "calculationContributions",
            contributions.map(item => {
                const impact = Number(item.contribution);

                const positive = impact > 0;

                const width = Math.max(
                    0.5,
                    Math.abs(impact) / maxImpact * 100
                );

                const inputValue =
                    item.input_value === null ||
                        item.input_value === undefined
                        ? "Đang thiếu – được Pipeline điền khuyết"
                        : item.input_value;

                return `
                    <div class="calculation-impact-row ${positive ? "toward-ckd" : "toward-not-ckd"}">

                        <div class="calculation-impact-heading">

                            <div>
                                <strong>
                                    ${escapeHtml(item.label)}
                                </strong>

                                <small>
                                    Giá trị nhập: ${escapeHtml(inputValue)}
                                </small>
                            </div>

                            <div class="calculation-impact-number">
                                <strong>
                                    ${impact > 0 ? "+" : ""}
                                    ${formatNumber(impact, 5)}
                                </strong>

                                <small>
                                    ${escapeHtml(item.direction_label)}
                                </small>
                            </div>

                        </div>

                        <div class="calculation-impact-track">
                            <span style="width:${width}%"></span>
                        </div>

                    </div>
                `;
            }).join("")
        );

        setText(
            "calculationNote",
            `${calculation.explanation || ""} ${calculation.limitation || ""}`
        );
    }


    // ==========================================
    // LAM MOI LICH SU
    // ==========================================

    const refreshButton = getElement("refreshHistoryBtn");

    if (refreshButton) {
        refreshButton.addEventListener(
            "click",
            async () => {
                await loadHistory();

                showToast(
                    "Danh sách lịch sử đã được cập nhật.",
                    "success",
                    "Đã làm mới"
                );
            }
        );
    }


    // ==========================================
    // KHOI TAO UNG DUNG
    // ==========================================

    showWizardStep(0);

    checkBackend();

    // Tai 10 mau khi mo website
    loadPredictionSamples();

})();
