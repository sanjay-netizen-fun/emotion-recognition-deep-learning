/* Emotion Recognition web demo — vanilla JS, no build step. */
(() => {
  "use strict";

  const EMOTIONS = [
    { name: "Angry", emoji: "😠", color: "#ef4444" },
    { name: "Disgust", emoji: "🤢", color: "#84cc16" },
    { name: "Fear", emoji: "😨", color: "#a855f7" },
    { name: "Happy", emoji: "😄", color: "#fbbf24" },
    { name: "Neutral", emoji: "😐", color: "#94a3b8" },
    { name: "Sad", emoji: "😢", color: "#3b82f6" },
    { name: "Surprise", emoji: "😲", color: "#f97316" },
  ];
  const EMOTION_BY_NAME = new Map(EMOTIONS.map((item) => [item.name, item]));

  const els = {
    stage: document.getElementById("stage"),
    video: document.getElementById("video"),
    still: document.getElementById("still"),
    overlay: document.getElementById("overlay"),
    stageEmpty: document.getElementById("stage-empty"),
    stageBadge: document.getElementById("stage-badge"),
    stageBadgeText: document.getElementById("stage-badge-text"),
    dropHint: document.getElementById("drop-hint"),
    cameraBtn: document.getElementById("camera-btn"),
    cameraBtnText: document.getElementById("camera-btn-text"),
    flipBtn: document.getElementById("flip-btn"),
    uploadBtn: document.getElementById("upload-btn"),
    emptyStart: document.getElementById("empty-start"),
    emptyUpload: document.getElementById("empty-upload"),
    fileInput: document.getElementById("file-input"),
    detectToggle: document.getElementById("detect-toggle"),
    statusPill: document.getElementById("status-pill"),
    statusText: document.getElementById("status-text"),
    modelTag: document.getElementById("model-tag"),
    hero: document.getElementById("emotion-hero"),
    emoji: document.getElementById("emotion-emoji"),
    name: document.getElementById("emotion-name"),
    confidence: document.getElementById("emotion-confidence"),
    meter: document.getElementById("confidence-meter"),
    scores: document.getElementById("scores"),
    statFaces: document.getElementById("stat-faces"),
    statLatency: document.getElementById("stat-latency"),
    statFps: document.getElementById("stat-fps"),
    toast: document.getElementById("toast"),
  };

  const state = {
    stream: null,
    running: false,
    busy: false,
    facingMode: "user",
    detect: els.detectToggle.checked,
    mode: "idle", // idle | camera | image
    frameTimes: [],
    lastImageFile: null,
    lastResult: null,
    toastTimer: null,
    warm: false,
    pending: false,
    lastSentAt: 0,
  };

  const scoreRows = new Map();
  let captureCanvas = document.createElement("canvas");

  /* ---------------------------------------------------------------- utils */

  function setStatus(tone, text) {
    els.statusPill.dataset.state = tone;
    els.statusText.textContent = text;
  }

  function showToast(message, tone = "info") {
    els.toast.textContent = message;
    els.toast.dataset.tone = tone;
    els.toast.hidden = false;
    requestAnimationFrame(() => els.toast.classList.add("visible"));
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => {
      els.toast.classList.remove("visible");
      setTimeout(() => {
        els.toast.hidden = true;
      }, 220);
    }, 4200);
  }

  function setBadge(text) {
    if (!text) {
      els.stageBadge.hidden = true;
      return;
    }
    els.stageBadgeText.textContent = text;
    els.stageBadge.hidden = false;
  }

  function buildScores() {
    els.scores.textContent = "";
    scoreRows.clear();
    for (const emotion of EMOTIONS) {
      const row = document.createElement("div");
      row.className = "score-row";
      row.dataset.emotion = emotion.name;

      const label = document.createElement("span");
      label.className = "score-label";
      label.textContent = emotion.name;

      const track = document.createElement("div");
      track.className = "score-track";
      const fill = document.createElement("div");
      fill.className = "score-fill";
      fill.style.setProperty("--bar-color", emotion.color);
      track.appendChild(fill);

      const value = document.createElement("span");
      value.className = "score-value";
      value.textContent = "0%";

      row.append(label, track, value);
      els.scores.appendChild(row);
      scoreRows.set(emotion.name, { row, fill, value });
    }
  }

  function resetScores() {
    for (const entry of scoreRows.values()) {
      entry.fill.style.width = "0%";
      entry.value.textContent = "0%";
      entry.row.classList.remove("is-top");
    }
  }

  function setHero({ emotion = null, title = "Waiting", subtitle = "No frame analysed yet", emoji = "·" }) {
    const match = EMOTION_BY_NAME.get(emotion);
    els.hero.dataset.emotion = match ? match.name : "none";
    els.emoji.textContent = emoji || (match ? match.emoji : "·");
    els.name.textContent = title;
    els.confidence.textContent = subtitle;
    els.meter.style.width = "0%";
  }

  function stopStream() {
    if (state.stream) {
      for (const track of state.stream.getTracks()) track.stop();
      state.stream = null;
    }
    els.video.srcObject = null;
    els.video.classList.remove("visible");
  }

  function clearOverlay() {
    const ctx = els.overlay.getContext("2d");
    ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  }

  function sizeOverlay() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const rect = els.stage.getBoundingClientRect();
    els.overlay.width = Math.max(1, Math.round(rect.width * dpr));
    els.overlay.height = Math.max(1, Math.round(rect.height * dpr));
  }

  /** Letterboxed content rectangle of an object-fit:contain element, in stage px. */
  function contentRect(intrinsicWidth, intrinsicHeight) {
    const stageRect = els.stage.getBoundingClientRect();
    const scale = Math.min(stageRect.width / intrinsicWidth, stageRect.height / intrinsicHeight);
    const width = intrinsicWidth * scale;
    const height = intrinsicHeight * scale;
    return {
      x: (stageRect.width - width) / 2,
      y: (stageRect.height - height) / 2,
      width,
      height,
    };
  }

  function roundRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    if (typeof ctx.roundRect === "function") {
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, r);
      return;
    }
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function drawOverlay(result, intrinsicWidth, intrinsicHeight, mirrored) {
    sizeOverlay();
    const ctx = els.overlay.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const stageRect = els.stage.getBoundingClientRect();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, stageRect.width, stageRect.height);

    if (!result || !result.faces || !result.faces.length) return;

    const rect = contentRect(intrinsicWidth, intrinsicHeight);
    const scaleX = rect.width / result.width;
    const scaleY = rect.height / result.height;

    for (const face of result.faces) {
      const [bx, by, bw, bh] = face.box;
      let x = rect.x + bx * scaleX;
      let y = rect.y + by * scaleY;
      const width = bw * scaleX;
      const height = bh * scaleY;
      if (mirrored) x = rect.x + rect.width - (x - rect.x) - width;

      const emotion = EMOTION_BY_NAME.get(face.emotion);
      const color = emotion ? emotion.color : "#ffffff";

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.shadowColor = "rgba(0,0,0,0.45)";
      ctx.shadowBlur = 6;
      roundRect(ctx, x, y, width, height, 10);
      ctx.stroke();

      const label = `${face.emotion} ${face.percent.toFixed(0)}%`;
      const fontSize = Math.max(11, Math.min(15, width / 9));
      ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif`;
      const paddingX = 7;
      const paddingY = 4;
      const textWidth = ctx.measureText(label).width;
      const chipWidth = textWidth + paddingX * 2;
      const chipHeight = fontSize + paddingY * 2;

      let chipX = x;
      let chipY = y - chipHeight - 6;
      if (chipY < 2) chipY = y + 6;
      chipX = Math.max(2, Math.min(chipX, stageRect.width - chipWidth - 2));

      ctx.shadowBlur = 0;
      ctx.fillStyle = color;
      roundRect(ctx, chipX, chipY, chipWidth, chipHeight, 7);
      ctx.fill();

      ctx.fillStyle = "#101014";
      ctx.textBaseline = "middle";
      ctx.fillText(label, chipX + paddingX, chipY + chipHeight / 2 + 0.5);
      ctx.restore();
    }
  }

  /* --------------------------------------------------------------- results */

  function largestFace(faces) {
    let best = null;
    let bestArea = -1;
    for (const face of faces) {
      const area = face.box[2] * face.box[3];
      if (area > bestArea) {
        bestArea = area;
        best = face;
      }
    }
    return best;
  }

  function updateResults(result, { mirrored = false, intrinsic = null } = {}) {
    state.lastResult = result;

    if (!result || !result.faces || !result.faces.length) {
      setHero({
        title: "No face",
        subtitle: state.detect
          ? "No face detected — try turning off “Detect faces”"
          : "No face in frame",
        emoji: "🫥",
      });
      resetScores();
      els.statFaces.textContent = "0";
      els.statLatency.textContent = result ? `${Math.round(result.elapsed_ms)} ms` : "—";
      clearOverlay();
      return;
    }

    const primary = largestFace(result.faces);
    const emotion = EMOTION_BY_NAME.get(primary.emotion);
    setHero({
      emotion: primary.emotion,
      title: primary.emotion,
      subtitle: `${primary.percent.toFixed(1)}% confidence`,
      emoji: emotion ? emotion.emoji : "·",
    });
    els.meter.style.width = `${Math.max(2, primary.percent)}%`;
    els.meter.style.background = `linear-gradient(90deg, ${emotion ? emotion.color : "#7c9cff"}, ${
      emotion ? emotion.color : "#7c9cff"
    }cc)`;

    for (const [name, entry] of scoreRows) {
      const value = primary.scores && typeof primary.scores[name] === "number" ? primary.scores[name] : 0;
      const percent = Math.max(0, Math.min(100, value * 100));
      entry.fill.style.width = `${percent}%`;
      entry.value.textContent = `${percent.toFixed(percent >= 10 ? 0 : 1)}%`;
      entry.row.classList.toggle("is-top", name === primary.emotion);
    }

    els.statFaces.textContent = String(result.count);

    if (intrinsic) drawOverlay(result, intrinsic.width, intrinsic.height, mirrored);
  }

  function markLatency(result, startedAt) {
    const elapsed = result && typeof result.elapsed_ms === "number" ? result.elapsed_ms : performance.now() - startedAt;
    els.statLatency.textContent = `${Math.round(elapsed)} ms`;
  }

  function noteFrameTime() {
    const now = performance.now();
    state.frameTimes.push(now);
    if (state.frameTimes.length > 15) state.frameTimes.shift();
    if (state.frameTimes.length >= 3) {
      const span = state.frameTimes[state.frameTimes.length - 1] - state.frameTimes[0];
      const fps = ((state.frameTimes.length - 1) * 1000) / Math.max(span, 1);
      els.statFps.textContent = `${fps.toFixed(1)} fps`;
    }
  }

  /* ------------------------------------------------------------------- API */

  async function analyzeBlob(blob, { mirrored = false, intrinsic = null } = {}) {
    const form = new FormData();
    form.append("image", blob, "frame.jpg");
    form.append("detect", state.detect ? "1" : "0");
    form.append("max_faces", "5");

    const startedAt = performance.now();
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      const error = new Error(payload.error || `Request failed (${response.status})`);
      error.status = response.status;
      throw error;
    }

    markLatency(payload, startedAt);
    updateResults(payload, { mirrored, intrinsic });
    noteFrameTime();
    return payload;
  }

  async function warmUp() {
    try {
      const response = await fetch("/api/warmup", { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || "Model warm-up failed");
      state.warm = true;
      if (state.mode !== "camera") setStatus("ready", "Model ready");
      els.modelTag.textContent = payload.model_loaded ? "Keras CNN · 48×48" : "Keras CNN";
    } catch (error) {
      setStatus("error", "Model unavailable");
      showToast(`Model could not be loaded: ${error.message}`, "error");
    }
  }

  /* ---------------------------------------------------------------- camera */

  async function startCamera() {
    if (state.running) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast("Camera access needs HTTPS or localhost in this browser.", "error");
      setStatus("error", "Camera unsupported");
      return;
    }

    setStatus("busy", "Starting camera");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: state.facingMode,
          width: { ideal: 960 },
          height: { ideal: 720 },
        },
      });

      state.stream = stream;
      els.video.srcObject = stream;
      els.video.classList.add("visible");
      els.video.classList.toggle("mirrored", state.facingMode === "user");
      els.still.hidden = true;
      els.stageEmpty.hidden = true;
      await els.video.play();

      state.running = true;
      state.mode = "camera";
      state.lastImageFile = null;
      state.frameTimes = [];
      els.cameraBtn.classList.add("danger");
      els.cameraBtn.classList.remove("primary");
      els.cameraBtnText.textContent = "Stop camera";
      els.flipBtn.disabled = false;
      setStatus("live", "Camera live");
      setBadge(null);
      clearOverlay();
      loop();
    } catch (error) {
      stopStream();
      state.running = false;
      els.cameraBtn.classList.add("primary");
      els.cameraBtn.classList.remove("danger");
      els.cameraBtnText.textContent = "Start camera";
      els.stageEmpty.hidden = false;
      setStatus("error", "Camera blocked");
      const reason =
        error && error.name === "NotAllowedError"
          ? "Camera permission was denied. Allow access in your browser settings."
          : error && error.name === "NotFoundError"
            ? "No camera found on this device."
            : `Could not start the camera: ${error.message || error}`;
      showToast(reason, "error");
    }
  }

  function stopCamera() {
    state.running = false;
    state.mode = "idle";
    stopStream();
    clearOverlay();
    els.cameraBtn.classList.add("primary");
    els.cameraBtn.classList.remove("danger");
    els.cameraBtnText.textContent = "Start camera";
    els.flipBtn.disabled = true;
    els.stageEmpty.hidden = false;
    els.stageBadge.hidden = true;
    setStatus(state.warm ? "ready" : "idle", state.warm ? "Model ready" : "Idle");
    els.statFps.textContent = "—";
  }

  function grabFrame() {
    const width = els.video.videoWidth;
    const height = els.video.videoHeight;
    if (!width || !height) return Promise.resolve(null);

    const maxSide = 640;
    const scale = Math.min(1, maxSide / Math.max(width, height));
    const targetWidth = Math.max(1, Math.round(width * scale));
    const targetHeight = Math.max(1, Math.round(height * scale));

    captureCanvas.width = targetWidth;
    captureCanvas.height = targetHeight;
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(els.video, 0, 0, targetWidth, targetHeight);

    return new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.82));
  }

  async function loop() {
    if (!state.running) return;
    if (state.pending || document.hidden) {
      setTimeout(loop, 120);
      return;
    }

    state.pending = true;
    state.lastSentAt = performance.now();
    const mirrored = state.facingMode === "user";
    try {
      const blob = await grabFrame();
      if (!blob) {
        setTimeout(loop, 200);
        return;
      }
      if (!state.warm) setBadge("Loading model…");
      await analyzeBlob(blob, {
        mirrored,
        intrinsic: { width: els.video.videoWidth, height: els.video.videoHeight },
      });
      setBadge(null);
      setStatus("live", "Camera live");
    } catch (error) {
      setBadge(null);
      setStatus("error", "Analysis failed");
      if (error.status === 503 && state.detect) {
        state.detect = false;
        els.detectToggle.checked = false;
        showToast("Face detection is unavailable here — switched to full-frame analysis.", "error");
      } else {
        showToast(error.message || "Analysis failed", "error");
      }
    } finally {
      state.pending = false;
      const wait = Math.max(0, 90 - (performance.now() - state.lastSentAt));
      if (state.running) setTimeout(loop, wait);
    }
  }

  /* ----------------------------------------------------------------- image */

  let currentImageUrl = null;

  async function analyzeFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showToast("Please choose an image file.", "error");
      return;
    }

    stopCamera();
    state.lastImageFile = file;
    state.mode = "image";
    setStatus("busy", "Analysing image");
    setBadge("Analysing…");

    if (currentImageUrl) URL.revokeObjectURL(currentImageUrl);
    currentImageUrl = URL.createObjectURL(file);
    els.still.src = currentImageUrl;
    els.still.hidden = false;
    els.stageEmpty.hidden = true;
    els.video.classList.remove("visible");
    els.overlay.getContext("2d").clearRect(0, 0, els.overlay.width, els.overlay.height);

    try {
      try {
        await els.still.decode();
      } catch (_) {
        /* decode() is not available everywhere; naturalWidth may still be set */
      }
      const intrinsic = {
        width: els.still.naturalWidth,
        height: els.still.naturalHeight,
      };
      await analyzeBlob(file, { mirrored: false, intrinsic });
      setStatus(state.warm ? "ready" : "idle", "Image analysed");
    } catch (error) {
      showToast(error.message || "Could not analyse that image", "error");
      setStatus("error", "Analysis failed");
    } finally {
      setBadge(null);
    }
  }

  /* --------------------------------------------------------------- events */

  els.cameraBtn.addEventListener("click", () => {
    if (state.running) stopCamera();
    else startCamera();
  });
  els.emptyStart.addEventListener("click", startCamera);
  els.emptyUpload.addEventListener("click", () => els.fileInput.click());
  els.uploadBtn.addEventListener("click", () => els.fileInput.click());

  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files && els.fileInput.files[0];
    if (file) analyzeFile(file);
    els.fileInput.value = "";
  });

  els.flipBtn.addEventListener("click", async () => {
    if (!state.running) return;
    state.facingMode = state.facingMode === "user" ? "environment" : "user";
    stopCamera();
    await startCamera();
  });

  els.detectToggle.addEventListener("change", () => {
    state.detect = els.detectToggle.checked;
    if (state.mode === "image" && state.lastImageFile) {
      analyzeFile(state.lastImageFile);
    }
  });

  // Drag & drop
  ["dragenter", "dragover"].forEach((type) => {
    els.stage.addEventListener(type, (event) => {
      event.preventDefault();
      els.stage.classList.add("dragover");
      els.dropHint.hidden = false;
    });
  });
  ["dragleave", "dragend"].forEach((type) => {
    els.stage.addEventListener(type, (event) => {
      if (type === "dragleave" && event.relatedTarget && els.stage.contains(event.relatedTarget)) return;
      els.stage.classList.remove("dragover");
      els.dropHint.hidden = true;
    });
  });
  els.stage.addEventListener("drop", (event) => {
    event.preventDefault();
    els.stage.classList.remove("dragover");
    els.dropHint.hidden = true;
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) analyzeFile(file);
  });

  window.addEventListener("paste", (event) => {
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          analyzeFile(file);
          break;
        }
      }
    }
  });

  window.addEventListener("resize", () => {
    sizeOverlay();
    if (state.lastResult) {
      const intrinsic =
        state.mode === "camera"
          ? { width: els.video.videoWidth, height: els.video.videoHeight }
          : { width: els.still.naturalWidth, height: els.still.naturalHeight };
      if (intrinsic.width && intrinsic.height) {
        drawOverlay(
          state.lastResult,
          intrinsic.width,
          intrinsic.height,
          state.mode === "camera" && state.facingMode === "user",
        );
      }
    }
  });

  window.addEventListener("pagehide", stopStream);
  window.addEventListener("beforeunload", stopStream);

  /* ----------------------------------------------------------------- boot */

  async function boot() {
    buildScores();
    setHero({ subtitle: "Loading model…" });
    setStatus("busy", "Loading model");
    sizeOverlay();

    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      if (!response.ok) throw new Error(`status ${response.status}`);
      const health = await response.json();
      els.modelTag.textContent = health.detection_available ? "Keras CNN · Haar" : "Keras CNN · 48×48";
      if (!health.detection_available) {
        els.detectToggle.checked = false;
        els.detectToggle.disabled = true;
        state.detect = false;
      }
    } catch (error) {
      setStatus("error", "Server offline");
      showToast("Could not reach the inference server.", "error");
      return;
    }

    await warmUp();
    if (!state.warm) return;
    setHero({});
    setStatus("ready", "Model ready");
  }

  boot();
})();
