// Hallucination Detection Studio Frontend Logic

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const promptInput = document.getElementById("promptInput");
  const completionInput = document.getElementById("completionInput");
  const promptCharCount = document.getElementById("promptCharCount");
  const completionCharCount = document.getElementById("completionCharCount");
  const thresholdRange = document.getElementById("thresholdRange");
  const thresholdValueBadge = document.getElementById("thresholdValueBadge");
  const analyzeButton = document.getElementById("analyzeButton");
  const analyzeSpinner = document.getElementById("analyzeSpinner");
  const analyzeBtnText = document.getElementById("analyzeBtnText");
  const presetChipsContainer = document.getElementById("presetChipsContainer");
  const verdictBadge = document.getElementById("verdictBadge");

  // Metrics
  const valMaxRisk = document.getElementById("valMaxRisk");
  const valMeanRisk = document.getElementById("valMeanRisk");
  const valSpansCount = document.getElementById("valSpansCount");
  const valSentenceRatio = document.getElementById("valSentenceRatio");
  const cardMaxRisk = document.getElementById("cardMaxRisk");

  // Views
  const tokenHeatmapContainer = document.getElementById("tokenHeatmapContainer");
  const sentenceListContainer = document.getElementById("sentenceListContainer");
  const spansTableBody = document.getElementById("spansTableBody");
  const jsonExportContent = document.getElementById("jsonExportContent");
  const copyJsonBtn = document.getElementById("copyJsonBtn");

  // State
  let cachedData = null;
  let activePresetId = null;

  // Character counters
  function updateCounters() {
    promptCharCount.textContent = `${promptInput.value.length} chars`;
    completionCharCount.textContent = `${completionInput.value.length} chars`;
  }
  promptInput.addEventListener("input", updateCounters);
  completionInput.addEventListener("input", updateCounters);

  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const targetTabId = btn.getAttribute("data-tab");
      document.querySelectorAll(".tab-content").forEach((view) => {
        view.style.display = view.id === targetTabId ? "block" : "none";
      });
    });
  });

  // Copy JSON
  copyJsonBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(jsonExportContent.textContent).then(() => {
      const orig = copyJsonBtn.textContent;
      copyJsonBtn.textContent = "Copied!";
      setTimeout(() => {
        copyJsonBtn.textContent = orig;
      }, 1800);
    });
  });

  // Interpolate Heatmap Color
  function getHeatmapColor(prob, threshold) {
    if (prob < threshold * 0.6) {
      return {
        bg: "rgba(16, 185, 129, 0.12)",
        color: "#6ee7b7",
        border: "1px solid rgba(16, 185, 129, 0.2)",
      };
    } else if (prob < threshold) {
      return {
        bg: "rgba(245, 158, 11, 0.18)",
        color: "#fde68a",
        border: "1px solid rgba(245, 158, 11, 0.35)",
      };
    } else {
      const intensity = Math.min(1.0, (prob - threshold) / (1.0 - threshold + 0.05));
      const alpha = 0.25 + 0.45 * intensity;
      return {
        bg: `rgba(244, 63, 94, ${alpha})`,
        color: "#ffe4e6",
        border: `1px solid rgba(244, 63, 94, ${Math.min(1.0, alpha + 0.3)})`,
        boxShadow: `0 0 ${Math.round(8 * intensity)}px rgba(244, 63, 94, 0.4)`,
      };
    }
  }

  // Render Analytics Views
  function renderAnalytics(data, currentThreshold) {
    if (!data || !data.tokens) return;

    // 1. Recalculate token states based on current threshold
    const tokens = data.tokens.map((t) => ({
      ...t,
      is_hallucinated: t.prob >= currentThreshold,
    }));

    // 2. Continuous Spans
    const spans = [];
    let i = 0;
    while (i < tokens.length) {
      if (tokens[i].is_hallucinated) {
        let j = i;
        while (j < tokens.length && tokens[j].is_hallucinated) {
          j++;
        }
        const spanTokens = tokens.slice(i, j);
        const spanText = spanTokens.map((t) => t.text).join("");
        const spanProb = spanTokens.reduce((acc, t) => acc + t.prob, 0) / spanTokens.length;
        spans.push({
          text: spanText,
          start_token: i,
          end_token: j - 1,
          score: spanProb,
        });
        i = j;
      } else {
        i++;
      }
    }

    // 3. Sentences
    const sentences = (data.sentences || []).map((s) => ({
      ...s,
      is_hallucinated: s.score >= currentThreshold,
    }));

    // 4. Summary Metrics
    const allProbs = tokens.map((t) => t.prob);
    const maxRisk = allProbs.length ? Math.max(...allProbs) : 0;
    const meanRisk = allProbs.length ? allProbs.reduce((a, b) => a + b, 0) / allProbs.length : 0;
    const isHallucinated = maxRisk >= currentThreshold;

    valMaxRisk.textContent = `${(maxRisk * 100).toFixed(1)}%`;
    valMeanRisk.textContent = `${(meanRisk * 100).toFixed(1)}%`;
    valSpansCount.textContent = spans.length;
    const hallucinatedSents = sentences.filter((s) => s.is_hallucinated).length;
    valSentenceRatio.textContent = `${hallucinatedSents} / ${sentences.length}`;

    if (isHallucinated) {
      cardMaxRisk.classList.add("danger");
      cardMaxRisk.classList.remove("success");
      verdictBadge.style.display = "inline-flex";
      verdictBadge.className = "status-pill";
      verdictBadge.style.background = "rgba(244, 63, 94, 0.15)";
      verdictBadge.style.borderColor = "rgba(244, 63, 94, 0.4)";
      verdictBadge.style.color = "#fda4af";
      verdictBadge.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f43f5e;box-shadow:0 0 8px #f43f5e;"></span> Risk: Hallucination Flagged`;
    } else {
      cardMaxRisk.classList.remove("danger");
      cardMaxRisk.classList.add("success");
      verdictBadge.style.display = "inline-flex";
      verdictBadge.className = "status-pill";
      verdictBadge.style.background = "rgba(16, 185, 129, 0.15)";
      verdictBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
      verdictBadge.style.color = "#6ee7b7";
      verdictBadge.innerHTML = `<span class="pulse-dot"></span> Verdict: Verified Factual`;
    }

    // 5. Render Heatmap
    tokenHeatmapContainer.innerHTML = "";
    tokens.forEach((t) => {
      const span = document.createElement("span");
      span.className = "token-span";
      const style = getHeatmapColor(t.prob, currentThreshold);
      span.style.backgroundColor = style.bg;
      span.style.color = style.color;
      span.style.border = style.border;
      if (style.boxShadow) span.style.boxShadow = style.boxShadow;

      // Handle spaces cleanly
      if (t.text === " ") {
        span.innerHTML = "&nbsp;";
      } else if (t.text === "\n") {
        span.innerHTML = "<br/>";
      } else {
        span.textContent = t.text;
      }

      // Tooltip
      const tooltip = document.createElement("div");
      tooltip.className = "token-tooltip";
      const statusLabel = t.is_hallucinated ? "🚨 Flagged" : "✅ Factual";
      tooltip.textContent = `Token #${t.index}: ${(t.prob * 100).toFixed(1)}% | ${statusLabel}`;
      span.appendChild(tooltip);

      tokenHeatmapContainer.appendChild(span);
    });

    // 6. Render Sentences
    sentenceListContainer.innerHTML = "";
    sentences.forEach((s, idx) => {
      const card = document.createElement("div");
      card.className = `sentence-card ${s.is_hallucinated ? "hallucinated" : "factual"}`;

      const tagClass = s.is_hallucinated ? "tag-hallucinated" : "tag-factual";
      const tagText = s.is_hallucinated ? "🚨 Hallucinated" : "✅ Factual";

      card.innerHTML = `
        <div class="sentence-card-top">
          <span class="sentence-tag ${tagClass}">${tagText} (Sentence ${idx + 1})</span>
          <div class="sentence-scores">
            Max: ${(s.score * 100).toFixed(1)}% &nbsp;|&nbsp; Mean: ${(s.mean_score * 100).toFixed(1)}%
          </div>
        </div>
        <div class="sentence-text">${escapeHtml(s.text)}</div>
      `;
      sentenceListContainer.appendChild(card);
    });

    // 7. Render Spans Table
    spansTableBody.innerHTML = "";
    if (spans.length === 0) {
      spansTableBody.innerHTML = `
        <tr>
          <td colspan="4" style="text-align:center; color:var(--text-faint); padding:2rem;">
            No hallucinated spans detected at threshold ${currentThreshold.toFixed(2)}.
          </td>
        </tr>
      `;
    } else {
      spans.forEach((sp) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><span class="span-code">${escapeHtml(sp.text)}</span></td>
          <td style="font-family:var(--font-mono); font-size:0.82rem; color:var(--text-muted);">
            [${sp.start_token} : ${sp.end_token}]
          </td>
          <td style="font-family:var(--font-mono); font-weight:600; color:#fda4af;">
            ${(sp.score * 100).toFixed(1)}%
          </td>
          <td>
            <span class="sentence-tag tag-hallucinated">Hallucinated</span>
          </td>
        `;
        spansTableBody.appendChild(row);
      });
    }

    // 8. JSON Export View
    const exportPayload = {
      overall_verdict: isHallucinated ? "HALLUCINATED" : "FACTUAL",
      decision_threshold: currentThreshold,
      max_token_risk: maxRisk,
      mean_token_risk: meanRisk,
      flagged_spans_count: spans.length,
      spans: spans,
      sentences: sentences,
      tokens: tokens,
    };
    jsonExportContent.textContent = JSON.stringify(exportPayload, null, 2);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Live Slider Response
  thresholdRange.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    thresholdValueBadge.textContent = val.toFixed(2);
    if (cachedData) {
      renderAnalytics(cachedData, val);
    }
  });

  // Execute Analysis API
  async function performAnalysis() {
    const prompt = promptInput.value.trim();
    const completion = completionInput.value.trim();
    const threshold = parseFloat(thresholdRange.value);

    if (!completion) {
      alert("Please enter a completion text to audit.");
      return;
    }

    // UI Loading state
    analyzeSpinner.style.display = "inline-block";
    analyzeBtnText.textContent = "Analyzing Representations...";
    analyzeButton.disabled = true;

    try {
      const resp = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          completion: completion,
          threshold: threshold,
        }),
      });

      if (!resp.ok) {
        throw new Error(`API error: ${resp.statusText}`);
      }

      const data = await resp.json();
      cachedData = data;
      renderAnalytics(data, threshold);
    } catch (err) {
      console.error(err);
      alert(`Detection failed: ${err.message}`);
    } finally {
      analyzeSpinner.style.display = "none";
      analyzeBtnText.textContent = "Audit for Hallucinations";
      analyzeButton.disabled = false;
    }
  }

  analyzeButton.addEventListener("click", performAnalysis);

  // Load Presets
  async function loadPresets() {
    try {
      const res = await fetch("/api/presets");
      if (!res.ok) return;
      const presets = await res.json();

      presetChipsContainer.innerHTML = "";
      presets.forEach((preset, index) => {
        const chip = document.createElement("button");
        chip.className = `preset-chip ${index === 0 ? "active" : ""}`;
        chip.id = `preset-${preset.id}`;
        chip.innerHTML = `
          <div class="preset-title">
            <span>${preset.title}</span>
            <span style="font-size:0.7rem; color:var(--text-faint); font-weight:normal;">${preset.category}</span>
          </div>
          <div class="preset-desc">${preset.description}</div>
        `;

        chip.addEventListener("click", () => {
          document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
          chip.classList.add("active");
          promptInput.value = preset.prompt;
          completionInput.value = preset.completion;
          updateCounters();
          performAnalysis();
        });

        presetChipsContainer.appendChild(chip);

        if (index === 0) {
          promptInput.value = preset.prompt;
          completionInput.value = preset.completion;
          updateCounters();
        }
      });

      // Run initial analysis on first preset
      performAnalysis();
    } catch (e) {
      console.warn("Could not load presets", e);
    }
  }

  loadPresets();
});
