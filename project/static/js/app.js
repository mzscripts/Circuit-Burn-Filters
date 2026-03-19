const appConfig = window.APP_CONFIG || {};
const PENDING_JOBS_KEY = "circuit-burn-pending-jobs";
const RESULT_CACHE_KEY = "circuit-burn-results-cache";
const UPLOAD_CACHE_KEY = "circuit-burn-upload-cache";
let pendingJobPollHandle = null;

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

function readJsonStorage(key) {
    try {
        const raw = window.localStorage.getItem(key);
        return raw ? JSON.parse(raw) : {};
    } catch (_error) {
        return {};
    }
}

function writeJsonStorage(key, value) {
    window.localStorage.setItem(key, JSON.stringify(value));
}

function readPendingJobStore() {
    return readJsonStorage(PENDING_JOBS_KEY);
}

function writePendingJobStore(store) {
    writeJsonStorage(PENDING_JOBS_KEY, store);
}

function readResultCacheStore() {
    return readJsonStorage(RESULT_CACHE_KEY);
}

function writeResultCacheStore(store) {
    writeJsonStorage(RESULT_CACHE_KEY, store);
}

function readUploadCacheStore() {
    return readJsonStorage(UPLOAD_CACHE_KEY);
}

function writeUploadCacheStore(store) {
    writeJsonStorage(UPLOAD_CACHE_KEY, store);
}

function listPendingJobs() {
    const store = readPendingJobStore();
    return Object.values(store).filter((job) => job.client_id === appConfig.clientId);
}

function listCachedResults() {
    const store = readResultCacheStore();
    return Object.values(store).filter((item) => item.client_id === appConfig.clientId);
}

function upsertPendingJob(job) {
    if (!job?.job_id || !appConfig.clientId) return;
    const store = readPendingJobStore();
    store[job.job_id] = {
        ...store[job.job_id],
        ...job,
        client_id: appConfig.clientId,
    };
    writePendingJobStore(store);
}

function removePendingJob(jobId) {
    const store = readPendingJobStore();
    if (!store[jobId]) return;
    delete store[jobId];
    writePendingJobStore(store);
}

function upsertCachedResult(result) {
    if (!result?.filter_id || !appConfig.clientId) return;
    const store = readResultCacheStore();
    store[`${appConfig.clientId}:${result.filter_id}`] = {
        ...result,
        client_id: appConfig.clientId,
    };
    writeResultCacheStore(store);
}

function cacheCurrentUpload() {
    const sourceImage = document.querySelector(".source-card img");
    const sourceName = document.querySelector(".source-meta strong");
    if (!sourceImage || !sourceName || !appConfig.clientId) return;
    const store = readUploadCacheStore();
    store[appConfig.clientId] = {
        display_url: sourceImage.currentSrc || sourceImage.src,
        original_name: sourceName.textContent || "",
    };
    writeUploadCacheStore(store);
}

function clearLocalClientCache() {
    const pendingStore = readPendingJobStore();
    Object.keys(pendingStore).forEach((jobId) => {
        if (pendingStore[jobId]?.client_id === appConfig.clientId) {
            delete pendingStore[jobId];
        }
    });
    writePendingJobStore(pendingStore);

    const resultStore = readResultCacheStore();
    Object.keys(resultStore).forEach((key) => {
        if (resultStore[key]?.client_id === appConfig.clientId) {
            delete resultStore[key];
        }
    });
    writeResultCacheStore(resultStore);

    const uploadStore = readUploadCacheStore();
    delete uploadStore[appConfig.clientId];
    writeUploadCacheStore(uploadStore);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function htmlToElement(markup) {
    const template = document.createElement("template");
    template.innerHTML = markup.trim();
    return template.content.firstElementChild;
}

function upsertMarkup(container, selector, markup, { prepend = false } = {}) {
    if (!container) return null;
    const element = htmlToElement(markup);
    const existing = selector ? container.querySelector(selector) : null;
    if (existing) {
        existing.replaceWith(element);
    } else if (prepend) {
        container.prepend(element);
    } else {
        container.append(element);
    }
    return element;
}

function createResultCardMarkup(result, variant = "gallery") {
    const articleClass = variant === "gallery" ? "gallery-card dynamic-image-card" : "result-card dynamic-image-card";
    const metaClass = variant === "gallery" ? "gallery-copy" : "result-meta";
    return `
        <article class="${articleClass}" data-result-card data-filter-id="${escapeHtml(result.filter_id)}">
            <div class="image-frame image-loading" data-image-frame>
                <div class="image-progress" data-image-progress>0%</div>
                <img class="deferred-image" data-src="${escapeHtml(result.image_url)}" alt="${escapeHtml(result.filter_name)}">
            </div>
            <div class="${metaClass} compact-meta">
                <strong>${escapeHtml(result.filter_name)}</strong>
                <span>${escapeHtml(result.file_size_label)}</span>
                <span>${escapeHtml(result.generation_time_label)}</span>
                <div class="result-actions">
                    <a class="ghost-button small" href="${escapeHtml(result.image_url)}" download target="_blank" rel="noopener">Download</a>
                </div>
            </div>
        </article>
    `;
}

function createPendingCardMarkup(job, variant = "gallery") {
    const articleClass = variant === "gallery"
        ? "gallery-card dynamic-image-card job-card"
        : "result-card dynamic-image-card job-card";
    const metaClass = variant === "gallery" ? "gallery-copy" : "result-meta";
    const label = job.status === "processing" ? "Generating..." : "Queued...";
    return `
        <article class="${articleClass}" data-job-card data-job-id="${escapeHtml(job.job_id)}" data-filter-id="${escapeHtml(job.filter_id)}">
            <div class="image-frame job-image-frame">
                <div class="job-badge">${label}</div>
            </div>
            <div class="${metaClass} compact-meta">
                <strong>${escapeHtml(job.filter_name)}</strong>
                <span class="job-status-line">${label}</span>
                <span>Open Results any time. This job keeps running.</span>
            </div>
        </article>
    `;
}

function createFailedCardMarkup(job, variant = "gallery") {
    const articleClass = variant === "gallery"
        ? "gallery-card dynamic-image-card job-card job-card-failed"
        : "result-card dynamic-image-card job-card job-card-failed";
    const metaClass = variant === "gallery" ? "gallery-copy" : "result-meta";
    return `
        <article class="${articleClass}" data-job-card data-job-id="${escapeHtml(job.job_id)}" data-filter-id="${escapeHtml(job.filter_id)}">
            <div class="image-frame job-image-frame">
                <div class="job-badge job-badge-error">Failed</div>
            </div>
            <div class="${metaClass} compact-meta">
                <strong>${escapeHtml(job.filter_name)}</strong>
                <span class="job-status-line">${escapeHtml(job.error || "Generation failed.")}</span>
                <span>Tap the card again to retry.</span>
            </div>
        </article>
    `;
}

function markFilterCardSelected(filterId, selected = true) {
    const selector = `.filter-card-button[data-filter-id="${CSS.escape(filterId)}"]`;
    document.querySelectorAll(selector).forEach((card) => {
        card.classList.toggle("is-selected", selected);
    });
}

function updateResultsGalleryMeta() {
    const grid = document.getElementById("results-gallery-grid");
    const count = document.getElementById("results-count");
    const emptyState = document.getElementById("results-empty-state");
    if (!grid) return;

    const totalCards = grid.querySelectorAll("[data-result-card], [data-job-card]").length;
    const resultCount = grid.querySelectorAll("[data-result-card]").length;
    if (count) {
        count.textContent = `${resultCount} output${resultCount === 1 ? "" : "s"}`;
    }
    if (emptyState) {
        emptyState.classList.toggle("hidden", totalCards > 0);
    }
}

function renderResultIntoGallery(result) {
    const grid = document.getElementById("results-gallery-grid");
    if (!grid) return;
    const selector = `[data-filter-id="${CSS.escape(result.filter_id)}"]`;
    const element = upsertMarkup(grid, selector, createResultCardMarkup(result, "gallery"), { prepend: true });
    if (element) initDeferredImages(element);
    upsertCachedResult(result);
    markFilterCardSelected(result.filter_id, true);
    updateResultsGalleryMeta();
}

function renderResultIntoRecent(result) {
    const grid = document.getElementById("recent-results-grid");
    if (!grid) return;
    const selector = `[data-filter-id="${CSS.escape(result.filter_id)}"]`;
    const element = upsertMarkup(grid, selector, createResultCardMarkup(result, "recent"), { prepend: true });
    if (element) initDeferredImages(element);
    upsertCachedResult(result);
    markFilterCardSelected(result.filter_id, true);
}

function renderPendingInGallery(job) {
    const grid = document.getElementById("results-gallery-grid");
    if (!grid) return;
    const selector = `[data-filter-id="${CSS.escape(job.filter_id)}"]`;
    upsertMarkup(grid, selector, createPendingCardMarkup(job, "gallery"), { prepend: true });
    markFilterCardSelected(job.filter_id, true);
    updateResultsGalleryMeta();
}

function renderPendingInRecent(job) {
    const grid = document.getElementById("recent-results-grid");
    if (!grid) return;
    const selector = `[data-filter-id="${CSS.escape(job.filter_id)}"]`;
    upsertMarkup(grid, selector, createPendingCardMarkup(job, "recent"), { prepend: true });
    markFilterCardSelected(job.filter_id, true);
}

function renderFailedJob(job) {
    const recentGrid = document.getElementById("recent-results-grid");
    if (recentGrid) {
        upsertMarkup(
            recentGrid,
            `[data-job-id="${CSS.escape(job.job_id)}"], [data-filter-id="${CSS.escape(job.filter_id)}"]`,
            createFailedCardMarkup(job, "recent"),
            { prepend: true },
        );
    }

    const galleryGrid = document.getElementById("results-gallery-grid");
    if (galleryGrid) {
        upsertMarkup(
            galleryGrid,
            `[data-job-id="${CSS.escape(job.job_id)}"], [data-filter-id="${CSS.escape(job.filter_id)}"]`,
            createFailedCardMarkup(job, "gallery"),
            { prepend: true },
        );
        updateResultsGalleryMeta();
    }

    markFilterCardSelected(job.filter_id, false);
}

function renderPendingJob(job) {
    renderPendingInRecent(job);
    renderPendingInGallery(job);
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

async function fetchJob(jobId) {
    const response = await fetch(`${appConfig.jobStatusBase}${jobId}`);
    const data = await response.json();
    if (!response.ok || data.success === false) {
        throw new Error(data.error || "Could not load job status.");
    }
    return data.job;
}

async function queueSingleFilter(card) {
    if (!card) return;
    const filterId = card.dataset.filterId;
    const filterName = card.dataset.filterName;
    if (!filterId || card.classList.contains("is-loading")) return;

    try {
        card.classList.add("is-selected", "is-loading");
        const data = await postJson(`${appConfig.processApiBase}${filterId}`, {});
        upsertPendingJob(data.job);
        renderPendingJob(data.job);
        showToast(`${filterName} queued.`);
    } catch (error) {
        card.classList.remove("is-selected");
        showToast(error.message, "error");
    } finally {
        card.classList.remove("is-loading");
    }
}

function hydrateCachedResults() {
    listCachedResults()
        .sort((left, right) => (left.sequence_number || 0) - (right.sequence_number || 0))
        .forEach((result) => {
            renderResultIntoRecent(result);
            renderResultIntoGallery(result);
        });
}

function hydratePendingJobs() {
    listPendingJobs().forEach((job) => renderPendingJob(job));
}

async function pollPendingJobs() {
    const jobs = listPendingJobs();
    if (!jobs.length) return;

    const updates = await Promise.allSettled(jobs.map((job) => fetchJob(job.job_id)));
    updates.forEach((update, index) => {
        const previousJob = jobs[index];
        if (update.status !== "fulfilled") {
            return;
        }

        const job = update.value;
        if (job.status === "queued" || job.status === "processing") {
            upsertPendingJob(job);
            renderPendingJob(job);
            return;
        }

        removePendingJob(job.job_id);
        if (job.status === "completed" && job.result) {
            renderResultIntoRecent(job.result);
            renderResultIntoGallery(job.result);
            showToast(`${job.filter_name} is ready.`);
            return;
        }

        renderFailedJob(job);
        showToast(job.error || `${previousJob.filter_name} failed.`, "error");
    });
}

function ensurePendingJobPolling() {
    if (pendingJobPollHandle) return;
    pendingJobPollHandle = window.setInterval(() => {
        pollPendingJobs().catch((error) => {
            console.error(error);
        });
    }, 2500);
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
        clearLocalClientCache();
        setButtonLoading(uploadButton, true);
    });
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
    const resetButton = document.querySelector("[data-reset-session]");
    const filterCards = Array.from(document.querySelectorAll(".filter-card-button"));

    filterCards.forEach((card) => {
        card.addEventListener("click", () => {
            queueSingleFilter(card);
        });
        card.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            queueSingleFilter(card);
        });
    });

    if (resetButton) {
        resetButton.addEventListener("click", async () => {
            try {
                await postJson(appConfig.resetSessionUrl, {});
                clearLocalClientCache();
                window.location.href = "/";
            } catch (error) {
                showToast(error.message, "error");
            }
        });
    }
}

function initResultsPage() {
    updateResultsGalleryMeta();
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
            URL.revokeObjectURL(objectUrl);
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
    if (!images.length) return;
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
    initResultsPage();
    initDeferredImages();
    hydrateCachedResults();
    hydratePendingJobs();
    cacheCurrentUpload();
    ensurePendingJobPolling();
    pollPendingJobs().catch((error) => {
        console.error(error);
    });
});
