const appConfig = window.APP_CONFIG || {};

function showToast(message, type = "success") {
    const container = document.getElementById("toast-container");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    window.setTimeout(() => toast.remove(), 3200);
}

function setButtonLoading(button, loading) {
    if (!button) return;
    const label = button.querySelector(".button-label");
    const spinner = button.querySelector(".button-spinner");
    button.disabled = loading;
    if (label) label.style.opacity = loading ? "0.65" : "1";
    if (spinner) spinner.classList.toggle("hidden", !loading);
}

function initUploadPage() {
    const form = document.getElementById("upload-form");
    const dropzone = document.getElementById("dropzone");
    const input = document.getElementById("image-input");
    const previewCard = document.getElementById("preview-card");
    const previewImage = document.getElementById("preview-image");
    const previewName = document.getElementById("preview-name");
    const previewMeta = document.getElementById("preview-meta");
    const uploadButton = document.getElementById("upload-button");

    if (!form || !dropzone || !input) return;

    const renderPreview = (file) => {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewName.textContent = file.name;
            previewMeta.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
            previewCard.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    };

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.add("is-dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.remove("is-dragover");
        });
    });

    dropzone.addEventListener("drop", (event) => {
        const [file] = event.dataTransfer.files;
        if (!file) return;
        input.files = event.dataTransfer.files;
        renderPreview(file);
    });

    input.addEventListener("change", () => {
        const [file] = input.files;
        renderPreview(file);
    });

    form.addEventListener("submit", () => {
        setButtonLoading(uploadButton, true);
    });
}

function getVisibleFilterCards() {
    return Array.from(document.querySelectorAll("[data-filter-card]")).filter((card) => card.style.display !== "none");
}

function getSelectedFilterIds() {
    return Array.from(document.querySelectorAll(".batch-checkbox:checked")).map((input) => input.value);
}

function updateSelectedCount() {
    const target = document.getElementById("selected-count");
    if (!target) return;
    const count = getSelectedFilterIds().length;
    target.textContent = `${count} selected`;
}

function prependRecentResult(result) {
    const grid = document.getElementById("recent-results-grid");
    if (!grid) return;
    const card = document.createElement("article");
    card.className = "result-card dynamic-image-card";
    card.innerHTML = `
        <div class="image-frame image-loading" data-image-frame>
            <div class="image-progress" data-image-progress>0%</div>
            <img class="deferred-image" data-src="${result.image_url}" alt="${result.filter_name}">
        </div>
        <div class="result-meta compact-meta">
            <strong>${result.filter_name}</strong>
            <span>${result.file_size_label}</span>
            <span>${result.generation_time_label}</span>
            <div class="result-actions">
                <a class="ghost-button small" href="/download/${result.filter_id}">Download</a>
                <button class="ghost-button small copy-link-button" type="button" data-copy-link="${result.image_url}">Copy Link</button>
            </div>
        </div>
    `;
    grid.prepend(card);
    initDeferredImages(card);
}

async function postJson(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || data.success === false) {
        throw new Error(data.error || "Something went wrong.");
    }
    return data;
}

function initFilterSearch() {
    const input = document.getElementById("filter-search");
    if (!input) return;
    const cards = Array.from(document.querySelectorAll("[data-filter-card]"));
    const sections = Array.from(document.querySelectorAll(".category-block"));

    input.addEventListener("input", () => {
        const query = input.value.trim().toLowerCase();
        cards.forEach((card) => {
            const matches = !query || card.dataset.search.includes(query);
            card.style.display = matches ? "" : "none";
            card.classList.toggle("is-matched", Boolean(query && matches));
        });

        sections.forEach((section) => {
            const hasVisibleCard = Array.from(section.querySelectorAll("[data-filter-card]")).some(
                (card) => card.style.display !== "none",
            );
            section.style.display = hasVisibleCard ? "" : "none";
        });
    });
}

function initFilterActions() {
    const batchButton = document.getElementById("generate-selected-button");
    const resetButton = document.querySelector("[data-reset-session]");
    const batchCheckboxes = Array.from(document.querySelectorAll(".batch-checkbox"));
    const selectVisibleButton = document.getElementById("select-visible-button");
    const clearSelectionButton = document.getElementById("clear-selection-button");

    batchCheckboxes.forEach((checkbox) => {
        checkbox.addEventListener("change", updateSelectedCount);
    });
    updateSelectedCount();

    if (selectVisibleButton) {
        selectVisibleButton.addEventListener("click", () => {
            getVisibleFilterCards().forEach((card) => {
                const checkbox = card.querySelector(".batch-checkbox");
                if (checkbox) checkbox.checked = true;
            });
            updateSelectedCount();
        });
    }

    if (clearSelectionButton) {
        clearSelectionButton.addEventListener("click", () => {
            document.querySelectorAll(".batch-checkbox").forEach((checkbox) => {
                checkbox.checked = false;
            });
            updateSelectedCount();
        });
    }

    if (batchButton) {
        batchButton.addEventListener("click", async () => {
            const filterIds = getSelectedFilterIds();
            if (!filterIds.length) {
                showToast("Select at least one filter first.", "error");
                return;
            }

            try {
                setButtonLoading(batchButton, true);
                updateBatchProgress(batchButton, 0, filterIds.length);
                const data = await postJson(appConfig.applyMultipleUrl, { filter_ids: filterIds });
                let successCount = 0;
                data.results.forEach((item, index) => {
                    updateBatchProgress(batchButton, index + 1, filterIds.length);
                    if (item.success) {
                        successCount += 1;
                        prependRecentResult(item);
                    } else {
                        showToast(`Failed: ${item.filter_id}`, "error");
                    }
                });
                showToast(`Generated ${successCount} hosted filter${successCount === 1 ? "" : "s"}.`);
            } catch (error) {
                showToast(error.message, "error");
            } finally {
                updateBatchProgress(batchButton, 0, 0);
                setButtonLoading(batchButton, false);
            }
        });
    }

    if (resetButton) {
        resetButton.addEventListener("click", async () => {
            try {
                await postJson(appConfig.resetSessionUrl, {});
                window.location.href = "/";
            } catch (error) {
                showToast(error.message, "error");
            }
        });
    }
}

async function copyToClipboard(text) {
    await navigator.clipboard.writeText(text);
}

function renderProcessResult(result) {
    const outputState = document.getElementById("process-output-state");
    outputState.innerHTML = `
        <article class="result-card process-result-card">
            <div class="image-frame image-loading process-image-frame" data-image-frame>
                <div class="image-progress" data-image-progress>0%</div>
                <img id="process-output-image" class="deferred-image" data-src="${result.image_url}" alt="${result.filter_name}">
            </div>
            <div class="result-meta compact-meta">
                <strong>${result.filter_name}</strong>
                <span>${result.file_size_label}</span>
                <span>${result.generation_time_label}</span>
                <div class="result-actions">
                    <a class="ghost-button small" href="/download/${result.filter_id}">Download</a>
                    <button class="ghost-button small copy-link-button" type="button" data-copy-link="${result.image_url}">Copy Link</button>
                </div>
            </div>
        </article>
    `;
    initDeferredImages(outputState);
}

function initProcessPage() {
    const container = document.querySelector("[data-process-page]");
    const button = document.getElementById("process-button");
    const status = document.getElementById("process-status");
    if (!container || !button || !status) return;

    const filterId = container.dataset.filterId;

    const runProcess = async () => {
        try {
            status.textContent = "Processing and uploading to ImgBB...";
            setButtonLoading(button, true);
            const result = await postJson(`${appConfig.processApiBase}${filterId}`, {});
            renderProcessResult(result);
            status.textContent = "Finished. Output uploaded to ImgBB.";
            showToast(`${result.filter_name} is ready.`);
        } catch (error) {
            status.textContent = "Processing failed.";
            showToast(error.message, "error");
        } finally {
            setButtonLoading(button, false);
        }
    };

    button.addEventListener("click", runProcess);

    if (container.dataset.autoProcess !== "false" && !document.getElementById("process-output-image")) {
        runProcess();
    }
}

function initCopyLinkButtons() {
    document.addEventListener("click", async (event) => {
        const button = event.target.closest(".copy-link-button");
        if (!button) return;
        try {
            await copyToClipboard(button.dataset.copyLink);
            showToast("Link copied.");
        } catch (error) {
            showToast("Could not copy the link.", "error");
        }
    });
}

function updateBatchProgress(button, completed, total) {
    const label = button?.querySelector(".button-label");
    if (!label) return;
    label.textContent = total ? `Generating ${completed}/${total}` : "Generate Selected";
}

function applyOrientationClass(image, frame) {
    frame.classList.remove("is-landscape", "is-portrait", "is-square");
    const width = image.naturalWidth || 1;
    const height = image.naturalHeight || 1;
    if (width > height * 1.15) {
        frame.classList.add("is-landscape");
    } else if (height > width * 1.15) {
        frame.classList.add("is-portrait");
    } else {
        frame.classList.add("is-square");
    }
}

function loadDeferredImage(image, frame) {
    const src = image.dataset.src;
    const progress = frame.querySelector("[data-image-progress]");
    if (!src || image.dataset.loaded === "true") return;

    const request = new XMLHttpRequest();
    request.open("GET", src, true);
    request.responseType = "blob";
    request.onprogress = (event) => {
        if (!progress) return;
        if (event.lengthComputable && event.total > 0) {
            const percent = Math.max(1, Math.min(100, Math.round((event.loaded / event.total) * 100)));
            progress.textContent = `${percent}%`;
        } else {
            progress.textContent = "Loading";
        }
    };
    request.onload = () => {
        if (request.status < 200 || request.status >= 300) {
            if (progress) progress.textContent = "Error";
            return;
        }
        const objectUrl = URL.createObjectURL(request.response);
        image.onload = () => {
            image.dataset.loaded = "true";
            applyOrientationClass(image, frame);
            frame.classList.remove("image-loading");
            if (progress) progress.remove();
        };
        image.src = objectUrl;
    };
    request.onerror = () => {
        if (progress) progress.textContent = "Error";
    };
    request.send();
}

function initDeferredImages(scope = document) {
    const images = Array.from(scope.querySelectorAll(".deferred-image"));
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            const image = entry.target;
            const frame = image.closest("[data-image-frame]");
            if (frame) loadDeferredImage(image, frame);
            observer.unobserve(image);
        });
    }, { rootMargin: "180px" });

    images.forEach((image) => observer.observe(image));
}

document.addEventListener("DOMContentLoaded", () => {
    initUploadPage();
    initFilterSearch();
    initFilterActions();
    initProcessPage();
    initCopyLinkButtons();
    initDeferredImages();
});
