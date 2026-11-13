
// ==========================================
// DU LIEU MAU
// ==========================================

const NUMERIC_FIELDS = new Set([
    "age", "bp", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc"
]);

const CKD_SAMPLE = {
    age: 65,
    bp: 90,
    sg: "1.01",
    al: "3",
    su: "1",
    rbc: "abnormal",
    pc: "abnormal",
    pcc: "present",
    ba: "present",
    bgr: 190,
    bu: 75,
    sc: 4.5,
    sod: 130,
    pot: 5.2,
    hemo: 8.5,
    pcv: 28,
    wc: 9500,
    rc: 3.2,
    htn: "yes",
    dm: "yes",
    cad: "yes",
    appet: "poor",
    pe: "yes",
    ane: "yes"
};

const NOT_CKD_SAMPLE = {
    age: 30,
    bp: 70,
    sg: "1.025",
    al: "0",
    su: "0",
    rbc: "normal",
    pc: "normal",
    pcc: "notpresent",
    ba: "notpresent",
    bgr: 90,
    bu: 22,
    sc: 0.8,
    sod: 142,
    pot: 4.1,
    hemo: 15.5,
    pcv: 45,
    wc: 6500,
    rc: 5.2,
    htn: "no",
    dm: "no",
    cad: "no",
    appet: "good",
    pe: "no",
    ane: "no"
};


// ==========================================
// CAU HINH
// ==========================================

const pageNames = {
    dashboard: "Tổng quan",
    analysis: "Phân tích bệnh thận",
    history: "Lịch sử phân tích"
};

const HISTORY_FIELD_LABELS = {
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


// ==========================================
// DOM ELEMENTS
// ==========================================

const form = document.getElementById("predictionForm");

const predictButton = document.getElementById("predictBtn");

const predictButtonText = document.getElementById(
    "predictBtnText"
);

const resultSection = document.getElementById("resultSection");

const resultCard = document.getElementById("resultCard");

const historyTableBody = document.getElementById(
    "historyTableBody"
);

const deleteModal = document.getElementById("deleteModal");

const confirmDeleteButton = document.getElementById(
    "confirmDeleteBtn"
);


// ==========================================
// WIZARD ELEMENTS
// ==========================================

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

const prevStepButton = document.getElementById(
    "prevStepBtn"
);

const nextStepButton = document.getElementById(
    "nextStepBtn"
);

const wizardHints = [
    "Nhập thông tin cơ bản và tiền sử bệnh lý.",
    "Nhập các chỉ số xét nghiệm nước tiểu.",
    "Nhập các chỉ số xét nghiệm máu trước khi phân tích."
];

let currentWizardStep = 0;
let pendingDeleteId = null;


// ==========================================
// HAM HO TRO
// ==========================================

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


function toPercentage(value) {
    if (value === undefined || value === null || value === "") {
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

    if (percentage === null) {
        return "--";
    }

    return `${percentage.toFixed(2)}%`;
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


// ==========================================
// THONG BAO TRONG WEBSITE
// ==========================================

function showToast(message, type = "success", title = "") {
    const container = document.getElementById("toastContainer");

    const toast = document.createElement("div");

    toast.className = `toast ${type}`;

    const defaultTitles = {
        success: "Thao tác thành công",
        error: "Đã xảy ra lỗi",
        warning: "Thông báo"
    };

    const icons = {
        success: "✓",
        error: "!",
        warning: "!"
    };

    const toastTitle =
        title || defaultTitles[type] || "Thông báo";

    const iconText = icons[type] || "i";

    toast.innerHTML = `
        <div class="toast-icon">${iconText}</div>

        <div class="toast-content">
            <strong>${escapeHtml(toastTitle)}</strong>
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

    let removeTimer;

    function removeToast() {
        clearTimeout(removeTimer);

        toast.classList.add("removing");

        setTimeout(() => {
            toast.remove();
        }, 200);
    }

    toast.querySelector(".toast-close").addEventListener(
        "click",
        removeToast
    );

    removeTimer = setTimeout(removeToast, 4000);
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

    const pageName = pageNames[viewName] || "Tổng quan";

    document.getElementById("pageHeading").textContent =
        viewName === "dashboard"
            ? "Tổng quan hệ thống"
            : pageName;

    document.getElementById("breadcrumbCurrent").textContent =
        pageName;

    if (viewName === "history") {
        loadHistory();
    }

    if (viewName === "analysis") {
        showWizardStep(currentWizardStep);
    }

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


// Su kien menu.
document.querySelectorAll("[data-view-target]").forEach(button => {
    button.addEventListener("click", () => {
        showView(button.dataset.viewTarget);
    });
});


// Nut bat dau phan tich.
document.getElementById("startAnalysisBtn").addEventListener(
    "click",
    () => {
        showView("analysis");
    }
);


// Nut mo phan tich.
document.getElementById("modelAnalysisBtn").addEventListener(
    "click",
    () => {
        showView("analysis");
    }
);


// ==========================================
// MULTI-STEP FORM
// ==========================================

// Hien thi tung buoc cua bieu mau.
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

    const progress = (
        currentWizardStep / (wizardSections.length - 1)
    ) * 100;

    wizardProgressFill.style.width = `${progress}%`;

    wizardStepCounter.textContent =
        `Bước ${currentWizardStep + 1} / ${wizardSections.length}`;

    wizardStepHint.textContent =
        wizardHints[currentWizardStep];

    prevStepButton.classList.toggle(
        "hidden",
        currentWizardStep === 0
    );

    nextStepButton.classList.toggle(
        "hidden",
        currentWizardStep === wizardSections.length - 1
    );

    predictButton.classList.toggle(
        "hidden",
        currentWizardStep !== wizardSections.length - 1
    );
}


// Kiem tra cac truong tai buoc hien tai.
function validateCurrentWizardStep() {
    const controls = wizardSections[
        currentWizardStep
    ].querySelectorAll("input, select");

    for (const input of controls) {
        if (!input.checkValidity()) {
            input.reportValidity();
            return false;
        }
    }

    return true;
}


// Kiem tra toan bo bieu mau truoc khi gui API.
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


// Nut tiep tuc.
nextStepButton.addEventListener("click", () => {
    if (!validateCurrentWizardStep()) {
        return;
    }

    showWizardStep(currentWizardStep + 1);

    document.querySelector(".wizard-progress").scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
});


// Nut quay lai.
prevStepButton.addEventListener("click", () => {
    showWizardStep(currentWizardStep - 1);

    document.querySelector(".wizard-progress").scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
});


// ==========================================
// KIEM TRA BACKEND
// ==========================================

async function checkBackend() {
    const dashboardStatus = document.getElementById(
        "dashboardModelStatus"
    );

    const dashboardDescription = document.getElementById(
        "dashboardModelDescription"
    );

    const sidebarDot = document.getElementById(
        "sidebarStatusDot"
    );

    const sidebarText = document.getElementById(
        "sidebarStatusText"
    );

    try {
        const response = await fetch("/health");

        const data = await response.json();

        const modelLoaded = data.loaded ?? data.model_loaded;

        if (
            !response.ok ||
            modelLoaded === false
        ) {
            throw new Error("Mô hình chưa sẵn sàng.");
        }

        dashboardStatus.textContent = "Sẵn sàng";

        dashboardDescription.textContent =
            "Mô hình đã được tải";

        sidebarDot.classList.add("online");

        sidebarDot.classList.remove("offline");

        sidebarText.textContent = "Backend đang hoạt động";

    } catch (error) {
        dashboardStatus.textContent = "Chưa kết nối";

        dashboardDescription.textContent =
            "Kiểm tra backend FastAPI";

        sidebarDot.classList.add("offline");

        sidebarDot.classList.remove("online");

        sidebarText.textContent = "Backend chưa sẵn sàng";
    }
}


// ==========================================
// DU LIEU MINH HOA
// ==========================================

function fillSampleData(sample) {
    Object.entries(sample).forEach(([fieldName, value]) => {
        const input = document.getElementById(fieldName);

        if (input) {
            input.value = String(value);
        }
    });

    resultSection.classList.add("hidden");

    // Du lieu mau duoc dien vao ca 24 truong.
    // Dua nguoi dung ve buoc dau tien.
    showWizardStep(0);

    document.querySelector(".wizard-progress").scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

    showToast(
        "Đã điền dữ liệu mẫu vào cả 24 thuộc tính.",
        "success",
        "Đã tải dữ liệu mẫu"
    );
}


// Mau CKD.
document.getElementById("sampleCkdBtn").addEventListener(
    "click",
    () => {
        fillSampleData(CKD_SAMPLE);
    }
);


// Mau Not CKD.
document.getElementById("sampleHealthyBtn").addEventListener(
    "click",
    () => {
        fillSampleData(NOT_CKD_SAMPLE);
    }
);


// ==========================================
// THU THAP DU LIEU
// ==========================================

function collectPatientData() {
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
// XU LY KET QUA DU DOAN
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


// Xac dinh nhan du doan.
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
        label.includes("notchronic")
    ) {
        return false;
    }

    if (label.includes("ckd")) {
        return true;
    }

    throw new Error(
        "Không xác định được nhãn dự đoán từ phản hồi API."
    );
}


// Hien thi phan tram.
function displayPercentage(value, textId, barId) {
    const textElement = document.getElementById(textId);

    const barElement = document.getElementById(barId);

    if (value === null) {
        textElement.textContent = "Không có dữ liệu";

        barElement.style.width = "0%";

        return;
    }

    textElement.textContent = `${value.toFixed(2)}%`;

    barElement.style.width = `${value}%`;
}


// Hien thi ket qua.
function displayResult(data) {
    const isCkd = determinePrediction(data);

    const probabilities = extractProbabilities(data);

    resultCard.classList.remove(
        "result-ckd",
        "result-not-ckd"
    );

    resultCard.classList.add(
        isCkd ? "result-ckd" : "result-not-ckd"
    );

    document.getElementById("resultIcon").textContent =
        isCkd ? "CKD" : "OK";

    document.getElementById("resultTitle").textContent =
        isCkd
            ? "Mô hình dự đoán CKD"
            : "Mô hình dự đoán Not CKD";

    document.getElementById("resultDescription").textContent =
        data.message || (
            isCkd
                ? "Mô hình phân loại dữ liệu đầu vào vào nhóm CKD."
                : "Mô hình phân loại dữ liệu đầu vào vào nhóm Not CKD."
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

    const probabilityNote = document.getElementById(
        "probabilityNote"
    );

    if (
        probabilities.ckd === null ||
        probabilities.notCkd === null
    ) {
        probabilityNote.textContent =
            "API không cung cấp đầy đủ xác suất cho hai lớp.";
    } else {
        probabilityNote.textContent =
            "Đây là xác suất dự đoán của mô hình, không phải xác suất chẩn đoán lâm sàng.";
    }

    document.getElementById("resultTime").textContent =
        "Thời gian phân tích: " +
        formatHistoryDate(
            data.created_at || new Date().toISOString()
        );

    resultSection.classList.remove("hidden");

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// ==========================================
// XU LY LOI API
// ==========================================

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
        }).join("\n");
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
// GUI DU LIEU DEN BACKEND
// ==========================================

form.addEventListener("submit", async event => {
    event.preventDefault();

    // Kiem tra tat ca cac buoc truoc khi gui du lieu.
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

    predictButton.disabled = true;

    predictButtonText.textContent = "Đang phân tích...";

    try {
        const response = await fetch("/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(patientData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(getErrorMessage(data));
        }

        displayResult(data);

        if (data.history_id !== undefined) {
            showToast(
                `Kết quả đã được lưu với mã lượt #${data.history_id}.`,
                "success",
                "Phân tích hoàn tất"
            );
        }

    } catch (error) {
        showToast(
            error.message,
            "error",
            "Không thể phân tích"
        );

    } finally {
        predictButton.disabled = false;

        predictButtonText.textContent = "Phân tích dữ liệu";
    }
});


// ==========================================
// RESET FORM
// ==========================================

form.addEventListener("reset", () => {
    setTimeout(() => {
        resultSection.classList.add("hidden");

        showWizardStep(0);

    }, 0);
});


// ==========================================
// LICH SU PHAN TICH
// ==========================================


// Tai danh sach lich su.
async function loadHistory() {
    const loading = document.getElementById(
        "historyLoading"
    );

    const errorBox = document.getElementById(
        "historyError"
    );

    loading.classList.remove("hidden");

    errorBox.classList.add("hidden");

    historyTableBody.innerHTML = `
        <tr>
            <td colspan="7" class="history-empty">
                Đang tải dữ liệu...
            </td>
        </tr>
    `;

    try {
        const response = await fetch("/history");

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail ||
                "Không thể tải lịch sử phân tích."
            );
        }

        document.getElementById("historyTotal").textContent =
            data.total ?? (data.items || []).length;

        renderHistory(data.items || []);

    } catch (error) {
        historyTableBody.innerHTML = `
            <tr>
                <td colspan="7" class="history-empty">
                    Không thể tải dữ liệu lịch sử.
                </td>
            </tr>
        `;

        errorBox.textContent = error.message;

        errorBox.classList.remove("hidden");

    } finally {
        loading.classList.add("hidden");
    }
}


// Hien thi danh sach lich su.
function renderHistory(records) {
    if (!records.length) {
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
            ? "CKD"
            : "Not CKD";

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


// Xem chi tiet mot lan phan tich.
async function openHistoryDetail(recordId) {
    const detailPanel = document.getElementById(
        "historyDetail"
    );

    try {
        const response = await fetch(
            `/history/${encodeURIComponent(recordId)}`
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail ||
                "Không thể tải chi tiết lịch sử."
            );
        }

        const record = data.item;

        document.getElementById("detailRecordId").textContent =
            `#${record.id}`;

        document.getElementById("detailCreatedAt").textContent =
            formatHistoryDate(record.created_at);

        document.getElementById("detailPrediction").textContent =
            record.prediction_label;

        document.getElementById("detailCkdProbability").textContent =
            formatHistoryProbability(record.ckd_probability);

        document.getElementById(
            "detailNotCkdProbability"
        ).textContent =
            formatHistoryProbability(record.not_ckd_probability);

        const inputData = record.input_data || {};

        const detailGrid = document.getElementById(
            "detailInputGrid"
        );

        detailGrid.innerHTML = Object.entries(
            HISTORY_FIELD_LABELS
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
    pendingDeleteId = recordId;

    document.getElementById("deleteModalRecord").textContent =
        `Mã lượt phân tích: #${recordId}`;

    deleteModal.classList.remove("hidden");

    deleteModal.setAttribute("aria-hidden", "false");

    document.body.style.overflow = "hidden";

    confirmDeleteButton.disabled = false;

    confirmDeleteButton.textContent = "Xóa lượt phân tích";

    confirmDeleteButton.focus();
}


function closeDeleteModal() {
    deleteModal.classList.add("hidden");

    deleteModal.setAttribute("aria-hidden", "true");

    document.body.style.overflow = "";

    pendingDeleteId = null;

    confirmDeleteButton.disabled = false;

    confirmDeleteButton.textContent = "Xóa lượt phân tích";
}


// Nut huy xoa.
document.getElementById("cancelDeleteBtn").addEventListener(
    "click",
    closeDeleteModal
);


// Click ben ngoai hop thoai.
deleteModal.addEventListener("click", event => {
    if (event.target === deleteModal) {
        closeDeleteModal();
    }
});


// Phim Escape.
document.addEventListener("keydown", event => {
    if (
        event.key === "Escape" &&
        !deleteModal.classList.contains("hidden")
    ) {
        closeDeleteModal();
    }
});


// Xac nhan xoa.
confirmDeleteButton.addEventListener("click", async () => {
    if (pendingDeleteId === null) {
        return;
    }

    const recordId = pendingDeleteId;

    confirmDeleteButton.disabled = true;

    confirmDeleteButton.textContent = "Đang xóa...";

    try {
        const response = await fetch(
            `/history/${encodeURIComponent(recordId)}`,
            {
                method: "DELETE"
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail ||
                "Không thể xóa lượt phân tích."
            );
        }

        const detailId = document.getElementById(
            "detailRecordId"
        ).textContent;

        if (detailId === `#${recordId}`) {
            document.getElementById("historyDetail")
                .classList.add("hidden");
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
});


// ==========================================
// CAC SU KIEN LICH SU
// ==========================================

// Chi tiet va xoa.
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


// Lam moi danh sach.
document.getElementById("refreshHistoryBtn").addEventListener(
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


// Dong chi tiet.
document.getElementById("closeHistoryDetailBtn").addEventListener(
    "click",
    () => {
        document.getElementById("historyDetail")
            .classList.add("hidden");
    }
);


// ==========================================
// KHOI TAO UNG DUNG
// ==========================================

showWizardStep(0);

checkBackend();
