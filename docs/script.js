const EXAMPLES = {
    "225": {
        file: "225.txt",
    },
    "ai-threat": {
        file: "ai-threat.txt",
    },
};

function setActiveTab(selectedKey) {
    for (const button of document.querySelectorAll(".tab-button")) {
        const isActive = button.dataset.example === selectedKey;
        button.classList.toggle("active", isActive);
        button.setAttribute("aria-selected", isActive ? "true" : "false");
    }
}

async function loadExample(key) {
    const spec = EXAMPLES[key];
    const outputEl = document.getElementById("example-output");
    const sourceEl = document.getElementById("example-source");
    const rawLinkEl = document.getElementById("raw-link");

    if (!spec || !outputEl || !sourceEl || !rawLinkEl) return;

    setActiveTab(key);
    sourceEl.innerHTML = `Source: <code>${spec.file}</code>`;
    rawLinkEl.href = spec.file;
    rawLinkEl.textContent = spec.file;

    outputEl.textContent = "Loading…";

    try {
        const res = await fetch(spec.file, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        outputEl.textContent = text.trimEnd();
    } catch (err) {
        outputEl.textContent = `Failed to load ${spec.file}.\n\n${String(err)}`;
    }
}

function initExamples() {
    const buttons = document.querySelectorAll(".tab-button");
    for (const button of buttons) {
        button.addEventListener("click", () => {
            const key = button.dataset.example;
            loadExample(key);
        });
    }

    loadExample("225");
}

document.addEventListener("DOMContentLoaded", initExamples);
