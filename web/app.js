/**
 * Guitar AI Studio - Frontend Application
 * Handles audio playback, dual-stem Demucs mixer (play-along mode),
 * interactive waveform rendering, real-time synchronous chord display,
 * and intelligent guitar tab extraction with volume restrictions.
 */

// Application State
let currentTrack = "sample_song.wav";
let audioDuration = 0;
let isPlaying = false;
let loopRegion = false;
let regionStart = 0;
let regionEnd = 3.0;

let chordTimeline = [];
let audioBuffer = null;
let animationFrameId = null;

// Stem & Mixer State
let isStemMode = false;
let stemUrls = null; // { guitar_url, no_guitar_url }
let masterVolume = 0.9;
let guitarVolume = 1.0;
let guitarMuted = false;
let noGuitarVolume = 1.0;
let noGuitarMuted = false;

// DOM Elements
const audio = document.getElementById("audioElement");
const audioGuitar = document.getElementById("audioStemGuitar");
const audioNoGuitar = document.getElementById("audioStemNoGuitar");

const trackSelect = document.getElementById("trackSelect");
const audioUpload = document.getElementById("audioUpload");
const currentTrackTitle = document.getElementById("currentTrackTitle");
const timeDisplay = document.getElementById("timeDisplay");

// Harmony Elements
const currentChordEl = document.getElementById("currentChord");
const chordConfidenceEl = document.getElementById("chordConfidence");
const nextChordEl = document.getElementById("nextChord");
const nextChordTimerEl = document.getElementById("nextChordTimer");
const progressionStrip = document.getElementById("progressionStrip");
const harmonyStatus = document.getElementById("harmonyStatus");

// Stems Elements
const stemsStatus = document.getElementById("stemsStatus");
const stemsPrompt = document.getElementById("stemsPrompt");
const stemsMixer = document.getElementById("stemsMixer");
const btnSeparateStems = document.getElementById("btnSeparateStems");
const stemsBtnIcon = document.getElementById("stemsBtnIcon");
const stemsBtnText = document.getElementById("stemsBtnText");

const guitarChannel = document.getElementById("guitarChannel");
const btnMuteGuitar = document.getElementById("btnMuteGuitar");
const volGuitar = document.getElementById("volGuitar");
const volGuitarVal = document.getElementById("volGuitarVal");
const guitarStateText = document.getElementById("guitarStateText");
const guitarHint = document.getElementById("guitarHint");

const noGuitarChannel = document.getElementById("noGuitarChannel");
const btnMuteNoGuitar = document.getElementById("btnMuteNoGuitar");
const volNoGuitar = document.getElementById("volNoGuitar");
const volNoGuitarVal = document.getElementById("volNoGuitarVal");
const noGuitarStateText = document.getElementById("noGuitarStateText");

// Waveform & Region Elements
const waveformWrapper = document.getElementById("waveformWrapper");
const waveformCanvas = document.getElementById("waveformCanvas");
const selectionOverlay = document.getElementById("selectionOverlay");
const handleLeft = document.getElementById("handleLeft");
const handleRight = document.getElementById("handleRight");
const playhead = document.getElementById("playhead");
const regionTimeEl = document.getElementById("regionTime");
const regionDurationEl = document.getElementById("regionDuration");

// Controls
const btnPlay = document.getElementById("btnPlay");
const playIcon = document.getElementById("playIcon");
const playText = document.getElementById("playText");
const btnLoopRegion = document.getElementById("btnLoopRegion");
const btnResetRegion = document.getElementById("btnResetRegion");
const volumeSlider = document.getElementById("volumeSlider");

// Tabs
const chkDemucsTabs = document.getElementById("chkDemucsTabs");
const guitarMuteWarning = document.getElementById("guitarMuteWarning");
const btnExtractTabs = document.getElementById("btnExtractTabs");
const tabBtnIcon = document.getElementById("tabBtnIcon");
const tabBtnText = document.getElementById("tabBtnText");
const asciiTabContent = document.getElementById("asciiTabContent");
const tabStatus = document.getElementById("tabStatus");
const tabActions = document.getElementById("tabActions");
const btnCopyTab = document.getElementById("btnCopyTab");
const btnDownloadTab = document.getElementById("btnDownloadTab");
const notesSummary = document.getElementById("notesSummary");
const notesChips = document.getElementById("notesChips");

// --- Initialization ---
async function init() {
  if (window.location.protocol === "file:") {
    const banner = document.createElement("div");
    banner.style.cssText =
      "background:#7c3aed; color:#fff; padding:14px 20px; font-weight:600; text-align:center; font-size:0.95rem; z-index:9999; box-shadow:0 4px 15px rgba(0,0,0,0.6); position:sticky; top:0;";
    banner.innerHTML = `
      ⚠️ Внимание: страница открыта как локальный файл (<code>file://</code>), поэтому браузер блокирует аудиопоток и API.<br>
      Пожалуйста, запустите <code>start_studio.bat</code> и откройте приложение по адресу: 
      <a href="http://127.0.0.1:8080" style="color:#ffffff; text-decoration:underline; font-weight:bold; margin-left:6px;">http://127.0.0.1:8080</a>
    `;
    document.body.prepend(banner);
  }

  setupEventListeners();
  await loadTrackList();
  if (trackSelect.value) {
    loadTrack(trackSelect.value);
  }
}

// --- Event Listeners ---
function setupEventListeners() {
  trackSelect.addEventListener("change", (e) => loadTrack(e.target.value));
  audioUpload.addEventListener("change", handleFileUpload);

  btnPlay.addEventListener("click", togglePlay);
  btnLoopRegion.addEventListener("click", toggleLoopRegion);
  btnResetRegion.addEventListener("click", resetRegionToFull);
  volumeSlider.addEventListener("input", onMasterVolumeChange);

  // Stems & Mixer Listeners
  btnSeparateStems.addEventListener("click", requestStemSeparation);
  btnMuteGuitar.addEventListener("click", toggleGuitarMute);
  volGuitar.addEventListener("input", onGuitarVolumeChange);
  btnMuteNoGuitar.addEventListener("click", toggleNoGuitarMute);
  volNoGuitar.addEventListener("input", onNoGuitarVolumeChange);

  // Tabs Listeners
  btnExtractTabs.addEventListener("click", requestTabExtraction);
  btnCopyTab.addEventListener("click", copyTabToClipboard);
  btnDownloadTab.addEventListener("click", downloadTabFile);

  // Audio Events
  audio.addEventListener("loadedmetadata", onAudioLoadedMetadata);
  audio.addEventListener("ended", onAudioEnded);
  audio.addEventListener("error", onAudioError);

  audioGuitar.addEventListener("error", onAudioError);
  audioNoGuitar.addEventListener("error", onAudioError);

  // Spacebar Hotkey
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") {
      e.preventDefault();
      togglePlay();
    }
  });

  setupWaveformInteractions();
}

function onAudioError(e) {
  const err = e.target.error;
  console.warn("Audio element error event:", err);
}

// --- Track Loading & APIs ---
async function loadTrackList() {
  try {
    const res = await fetch("/api/tracks");
    const data = await res.json();
    trackSelect.innerHTML = "";

    data.tracks.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t.filename;
      opt.textContent = t.label;
      trackSelect.appendChild(opt);
    });

    if (data.tracks.length > 0) {
      trackSelect.value = data.tracks[0].filename;
    }
  } catch (err) {
    console.error("Failed to load tracks:", err);
    harmonyStatus.textContent = "Сервер не запущен (запустите start_studio.bat)";
  }
}

async function loadTrack(filename) {
  currentTrack = filename;
  currentTrackTitle.textContent = filename;
  pauseAudio();

  const streamUrl = `/audio/${encodeURIComponent(filename)}`;
  audio.src = streamUrl;
  audio.load();

  // Reset stems mode for new track
  isStemMode = false;
  stemUrls = null;
  audioGuitar.src = "";
  audioNoGuitar.src = "";

  // 1. Load waveform visually
  loadAudioWaveform(streamUrl);

  // 2. Step 1: Detect harmony chords FIRST
  await loadChords(filename);

  // 3. Step 2: Check if stems already exist for Demucs play-along
  await checkStemsStatus(filename);

  // Update tabs button restriction
  updateGuitarRestrictionUI();
}

async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  harmonyStatus.textContent = "Загрузка файла...";
  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    await loadTrackList();
    trackSelect.value = data.filename;
    loadTrack(data.filename);
  } catch (err) {
    alert("Ошибка загрузки файла: " + err.message);
  }
}

// --- Chord Harmony ---
async function loadChords(filename) {
  harmonyStatus.textContent = "Анализ гармонии...";
  currentChordEl.textContent = "—";
  chordConfidenceEl.textContent = "Вычисление...";
  progressionStrip.innerHTML = '<span style="color:#8e95a5; font-size:0.8rem;">Анализ аккордов...</span>';

  try {
    const res = await fetch(`/api/chords?file=${encodeURIComponent(filename)}`);
    const data = await res.json();
    chordTimeline = data.segments || [];
    renderProgressionStrip();
    harmonyStatus.textContent = `Готово (${chordTimeline.length} аккордов)`;
  } catch (err) {
    harmonyStatus.textContent = "Ошибка анализа";
    console.error("Chord error:", err);
  }
}

function renderProgressionStrip() {
  progressionStrip.innerHTML = "";
  if (!chordTimeline.length) return;

  chordTimeline.forEach((seg, index) => {
    const chip = document.createElement("div");
    chip.className = "chord-chip";
    chip.dataset.index = index;
    chip.dataset.start = seg.start;
    chip.dataset.end = seg.end;

    chip.innerHTML = `
      <div class="chip-chord">${seg.chord}</div>
      <div class="chip-time">${seg.start.toFixed(1)}s</div>
    `;

    chip.addEventListener("click", () => {
      seekAudio(seg.start);
      updatePlayheadPosition();
      updateHarmonyUI();
    });

    progressionStrip.appendChild(chip);
  });
}

// --- Demucs Stem Separation & Play-Along Mixer ---
async function checkStemsStatus(filename) {
  stemsStatus.textContent = "Проверка...";
  try {
    const res = await fetch(`/api/stems-status?file=${encodeURIComponent(filename)}`);
    const data = await res.json();
    if (data.separated) {
      applyStemMode(data.guitar_url, data.no_guitar_url);
      stemsStatus.textContent = "Стемы готовы";
    } else {
      disableStemMode();
      stemsStatus.textContent = "Стемы не разделены";
    }
  } catch (err) {
    console.warn("Stem check error:", err);
    disableStemMode();
  }
}

function applyStemMode(guitarUrl, noGuitarUrl) {
  isStemMode = true;
  stemUrls = { guitar_url: guitarUrl, no_guitar_url: noGuitarUrl };

  audioGuitar.src = guitarUrl;
  audioNoGuitar.src = noGuitarUrl;
  audioGuitar.load();
  audioNoGuitar.load();

  stemsPrompt.style.display = "none";
  stemsMixer.style.display = "grid";

  applyMixerVolumes();
  updateGuitarRestrictionUI();
}

function disableStemMode() {
  isStemMode = false;
  stemUrls = null;
  stemsPrompt.style.display = "flex";
  stemsMixer.style.display = "none";
  updateGuitarRestrictionUI();
}

async function requestStemSeparation() {
  btnSeparateStems.disabled = true;
  stemsBtnIcon.textContent = "⏳";
  stemsBtnText.textContent = "Demucs разделяет трек... (подождите)";
  stemsStatus.textContent = "Разделение стемов (htdemucs_6s)...";

  try {
    const res = await fetch("/api/separate-stems", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file: currentTrack }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    applyStemMode(data.guitar_url, data.no_guitar_url);
    stemsStatus.textContent = `Готово за ${data.elapsed_time}с`;
    alert("🎉 " + data.message + "\nТеперь вы можете выключать или приглушать гитару и подыгрывать под минус!");
  } catch (err) {
    alert("Ошибка разделения стемов: " + err.message);
    stemsStatus.textContent = "Ошибка Demucs";
  } finally {
    btnSeparateStems.disabled = false;
    stemsBtnIcon.textContent = "🎛️";
    stemsBtnText.textContent = "Разделить на стемы (Demucs)";
  }
}

// Mixer Controls
function toggleGuitarMute() {
  guitarMuted = !guitarMuted;
  btnMuteGuitar.textContent = guitarMuted ? "Включить" : "Выключить";
  btnMuteGuitar.classList.toggle("is-muted", guitarMuted);
  guitarChannel.classList.toggle("channel-muted", guitarMuted);
  guitarStateText.textContent = guitarMuted ? "Заглушено" : "Звучит";
  applyMixerVolumes();
  updateGuitarRestrictionUI();
}

function onGuitarVolumeChange(e) {
  guitarVolume = parseFloat(e.target.value);
  volGuitarVal.textContent = `${Math.round(guitarVolume * 100)}%`;
  if (guitarVolume > 0 && guitarMuted) {
    toggleGuitarMute();
  }
  applyMixerVolumes();
  updateGuitarRestrictionUI();
}

function toggleNoGuitarMute() {
  noGuitarMuted = !noGuitarMuted;
  btnMuteNoGuitar.textContent = noGuitarMuted ? "Включить" : "Выключить";
  btnMuteNoGuitar.classList.toggle("is-muted", noGuitarMuted);
  noGuitarChannel.classList.toggle("channel-muted", noGuitarMuted);
  noGuitarStateText.textContent = noGuitarMuted ? "Заглушено" : "Звучит";
  applyMixerVolumes();
}

function onNoGuitarVolumeChange(e) {
  noGuitarVolume = parseFloat(e.target.value);
  volNoGuitarVal.textContent = `${Math.round(noGuitarVolume * 100)}%`;
  if (noGuitarVolume > 0 && noGuitarMuted) {
    toggleNoGuitarMute();
  }
  applyMixerVolumes();
}

function onMasterVolumeChange(e) {
  masterVolume = parseFloat(e.target.value);
  applyMixerVolumes();
}

function applyMixerVolumes() {
  if (isStemMode) {
    audio.volume = 0;
    audioGuitar.volume = guitarMuted ? 0 : guitarVolume * masterVolume;
    audioNoGuitar.volume = noGuitarMuted ? 0 : noGuitarVolume * masterVolume;
  } else {
    audio.volume = masterVolume;
  }
}

// Restriction: Disable tab extraction ONLY if stems are separated AND guitar is muted or volume = 0
function updateGuitarRestrictionUI() {
  if (!isStemMode) {
    // Standard mode (stems NOT separated): NO restriction or warning should ever appear
    if (guitarMuteWarning) guitarMuteWarning.style.display = "none";
    if (btnExtractTabs) btnExtractTabs.disabled = false;
    return;
  }

  const isEffectiveGuitarZero = (guitarMuted || guitarVolume <= 0.01);

  if (isEffectiveGuitarZero) {
    btnExtractTabs.disabled = true;
    guitarMuteWarning.style.display = "block";
    guitarHint.textContent = "⚠️ Гитара заглушена: табы недоступны";
    guitarHint.className = "guitar-status-hint is-warning";
    guitarChannel.classList.add("channel-muted");
  } else {
    btnExtractTabs.disabled = false;
    guitarMuteWarning.style.display = "none";
    guitarHint.textContent = "Гитара включена: можно снимать табы";
    guitarHint.className = "guitar-status-hint";
    if (!guitarMuted) guitarChannel.classList.remove("channel-muted");
  }
}

// --- Waveform Rendering (Offline Canvas) ---
async function loadAudioWaveform(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const arrayBuffer = await response.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    audioCtx.decodeAudioData(
      arrayBuffer,
      (decoded) => {
        audioBuffer = decoded;
        drawWaveform();
      },
      (err) => console.warn("decodeAudioData error:", err)
    );
  } catch (err) {
    console.error("Waveform decode error:", err);
  }
}

function drawWaveform() {
  if (!audioBuffer) return;
  const canvas = waveformCanvas;
  const ctx = canvas.getContext("2d");

  const dpr = window.devicePixelRatio || 1;
  const rect = waveformWrapper.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const width = rect.width;
  const height = rect.height;

  const rawData = audioBuffer.getChannelData(0);
  const step = Math.ceil(rawData.length / width);
  const amp = height / 2;

  ctx.clearRect(0, 0, width, height);

  // Black-Violet-White Gradient
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#c084fc");
  gradient.addColorStop(0.5, "#ffffff");
  gradient.addColorStop(1, "#7c3aed");
  ctx.fillStyle = gradient;

  for (let i = 0; i < width; i++) {
    let min = 1.0;
    let max = -1.0;
    for (let j = 0; j < step; j++) {
      const datum = rawData[i * step + j];
      if (datum < min) min = datum;
      if (datum > max) max = datum;
    }
    const yTop = (1 + min) * amp;
    const yBottom = Math.max(1, (max - min) * amp);
    ctx.fillRect(i, yTop, 1, yBottom);
  }

  updateSelectionOverlayPosition();
}

window.addEventListener("resize", () => {
  if (audioBuffer) drawWaveform();
});

// --- Waveform Interaction & Region Selection ---
function setupWaveformInteractions() {
  let isDragging = false;
  let dragType = null;
  let startX = 0;
  let initialRegionStart = 0;
  let initialRegionEnd = 0;

  function getTimeFromX(clientX) {
    const rect = waveformWrapper.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    return (x / rect.width) * (audioDuration || 1);
  }

  handleLeft.addEventListener("mousedown", (e) => {
    e.stopPropagation();
    isDragging = true;
    dragType = "left";
  });

  handleRight.addEventListener("mousedown", (e) => {
    e.stopPropagation();
    isDragging = true;
    dragType = "right";
  });

  selectionOverlay.addEventListener("mousedown", (e) => {
    if (e.target === handleLeft || e.target === handleRight) return;
    isDragging = true;
    dragType = "move";
    startX = e.clientX;
    initialRegionStart = regionStart;
    initialRegionEnd = regionEnd;
  });

  waveformWrapper.addEventListener("mousedown", (e) => {
    if (e.target.closest("#selectionOverlay")) return;
    const clickTime = getTimeFromX(e.clientX);
    seekAudio(clickTime);
    updatePlayheadPosition();

    isDragging = true;
    dragType = "create";
    regionStart = clickTime;
    regionEnd = clickTime + 0.1;
    updateSelectionOverlayPosition();
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const curTime = getTimeFromX(e.clientX);

    if (dragType === "create") {
      regionEnd = Math.max(regionStart + 0.1, curTime);
    } else if (dragType === "left") {
      regionStart = Math.min(curTime, regionEnd - 0.2);
    } else if (dragType === "right") {
      regionEnd = Math.max(curTime, regionStart + 0.2);
    } else if (dragType === "move") {
      const rect = waveformWrapper.getBoundingClientRect();
      const dt = ((e.clientX - startX) / rect.width) * audioDuration;
      const dur = initialRegionEnd - initialRegionStart;
      let newStart = initialRegionStart + dt;
      let newEnd = initialRegionEnd + dt;
      if (newStart < 0) {
        newStart = 0;
        newEnd = dur;
      }
      if (newEnd > audioDuration) {
        newEnd = audioDuration;
        newStart = audioDuration - dur;
      }
      regionStart = Math.max(0, newStart);
      regionEnd = Math.min(audioDuration, newEnd);
    }

    updateSelectionOverlayPosition();
  });

  window.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      dragType = null;
    }
  });
}

function updateSelectionOverlayPosition() {
  if (!audioDuration) return;
  const rect = waveformWrapper.getBoundingClientRect();
  const leftPx = (regionStart / audioDuration) * rect.width;
  const widthPx = ((regionEnd - regionStart) / audioDuration) * rect.width;

  selectionOverlay.style.left = `${leftPx}px`;
  selectionOverlay.style.width = `${Math.max(8, widthPx)}px`;

  regionTimeEl.textContent = `${regionStart.toFixed(2)}s – ${regionEnd.toFixed(2)}s`;
  regionDurationEl.textContent = `${(regionEnd - regionStart).toFixed(2)}s`;
}

function resetRegionToFull() {
  regionStart = 0;
  regionEnd = audioDuration || 3.0;
  updateSelectionOverlayPosition();
}

function toggleLoopRegion() {
  loopRegion = !loopRegion;
  btnLoopRegion.style.background = loopRegion ? "rgba(147, 51, 234, 0.4)" : "";
  btnLoopRegion.style.borderColor = loopRegion ? "#9333ea" : "";
}

// --- Playback Controls (Supporting Dual-Stem Sync) ---
function togglePlay() {
  if (isPlaying) {
    pauseAudio();
  } else {
    playAudio();
  }
}

function getCurrentAudioTime() {
  return isStemMode ? (audioGuitar.currentTime || audioNoGuitar.currentTime || 0) : audio.currentTime;
}

function seekAudio(timeInSec) {
  if (isStemMode) {
    audioGuitar.currentTime = timeInSec;
    audioNoGuitar.currentTime = timeInSec;
  }
  audio.currentTime = timeInSec;
}

function playAudio() {
  const curTime = getCurrentAudioTime();
  if (loopRegion && curTime >= regionEnd) {
    seekAudio(regionStart);
  }

  applyMixerVolumes();

  if (isStemMode) {
    // Sync both stems
    const t = audioGuitar.currentTime || 0;
    audioGuitar.currentTime = t;
    audioNoGuitar.currentTime = t;

    Promise.all([audioGuitar.play(), audioNoGuitar.play()])
      .then(() => {
        isPlaying = true;
        playIcon.textContent = "⏸";
        playText.textContent = "Пауза";
        startPlaybackLoop();
      })
      .catch((err) => console.log("Stem play error:", err));
  } else {
    audio
      .play()
      .then(() => {
        isPlaying = true;
        playIcon.textContent = "⏸";
        playText.textContent = "Пауза";
        startPlaybackLoop();
      })
      .catch((err) => console.log("Audio play error:", err));
  }
}

function pauseAudio() {
  audio.pause();
  audioGuitar.pause();
  audioNoGuitar.pause();
  isPlaying = false;
  playIcon.textContent = "▶";
  playText.textContent = "Играть";
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
}

function onAudioLoadedMetadata() {
  audioDuration = audio.duration;
  regionStart = 0;
  regionEnd = Math.min(3.0, audioDuration);
  updateSelectionOverlayPosition();
  updateTimeDisplay();
  if (audioBuffer) drawWaveform();
}

function onAudioEnded() {
  pauseAudio();
  seekAudio(0);
  updatePlayheadPosition();
  updateHarmonyUI();
}

// --- Playback Loop & Real-Time Sync ---
function startPlaybackLoop() {
  function loop() {
    if (!isPlaying) return;

    const curTime = getCurrentAudioTime();

    // A-B Loop Check
    if (loopRegion && curTime >= regionEnd) {
      seekAudio(regionStart);
    }

    // Stem resync check (if drifted > 40ms)
    if (isStemMode) {
      const diff = Math.abs(audioGuitar.currentTime - audioNoGuitar.currentTime);
      if (diff > 0.04) {
        audioNoGuitar.currentTime = audioGuitar.currentTime;
      }
    }

    updatePlayheadPosition();
    updateTimeDisplay();
    updateHarmonyUI();

    animationFrameId = requestAnimationFrame(loop);
  }
  animationFrameId = requestAnimationFrame(loop);
}

function updatePlayheadPosition() {
  if (!audioDuration) return;
  const curTime = getCurrentAudioTime();
  const rect = waveformWrapper.getBoundingClientRect();
  const x = (curTime / audioDuration) * rect.width;
  playhead.style.left = `${x}px`;
}

function updateTimeDisplay() {
  const curTime = getCurrentAudioTime();
  timeDisplay.textContent = `${formatTime(curTime)} / ${formatTime(audioDuration)}`;
}

function formatTime(sec) {
  if (isNaN(sec)) return "00:00.0";
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return `${m < 10 ? "0" + m : m}:${s < 10 ? "0" + s : s}`;
}

// --- Synchronous Harmony UI ---
let lastActiveChord = "";

function updateHarmonyUI() {
  if (!chordTimeline.length) return;
  const curTime = getCurrentAudioTime();

  let currentSeg = null;
  let nextSeg = null;

  for (let i = 0; i < chordTimeline.length; i++) {
    const seg = chordTimeline[i];
    if (seg.start <= curTime && curTime < seg.end) {
      currentSeg = seg;
      if (i + 1 < chordTimeline.length) {
        nextSeg = chordTimeline[i + 1];
      }
      break;
    }
  }

  if (!currentSeg && curTime >= chordTimeline[chordTimeline.length - 1].start) {
    currentSeg = chordTimeline[chordTimeline.length - 1];
  }

  if (currentSeg) {
    const chordName = currentSeg.chord === "N" ? "—" : currentSeg.chord;
    if (chordName !== lastActiveChord) {
      lastActiveChord = chordName;
      currentChordEl.textContent = chordName;
      currentChordEl.classList.add("active-pulse");
      setTimeout(() => currentChordEl.classList.remove("active-pulse"), 180);
    }

    chordConfidenceEl.textContent =
      currentSeg.chord === "N"
        ? "Пауза / тишина"
        : `Уверенность: ${Math.round(currentSeg.confidence * 100)}%`;

    if (nextSeg) {
      const timeRemaining = Math.max(0, nextSeg.start - curTime);
      nextChordEl.textContent = nextSeg.chord;
      nextChordTimerEl.textContent = `через ${timeRemaining.toFixed(1)} сек`;
    } else {
      nextChordEl.textContent = "—";
      nextChordTimerEl.textContent = "конец композиции";
    }

    const chips = progressionStrip.querySelectorAll(".chord-chip");
    chips.forEach((chip) => {
      const cStart = parseFloat(chip.dataset.start);
      const cEnd = parseFloat(chip.dataset.end);
      if (cStart <= curTime && curTime < cEnd) {
        if (!chip.classList.contains("current-chip")) {
          chip.classList.add("current-chip");
          chip.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
        }
      } else {
        chip.classList.remove("current-chip");
      }
    });
  }
}

// --- Tab Transcription Request with Guitar Restriction ---
async function requestTabExtraction() {
  // Restriction check
  if (isStemMode && (guitarMuted || guitarVolume <= 0.01)) {
    alert("⚠️ Гитара выключена или её громкость на нуле!\nВключите дорожку гитары в микшере, чтобы снять табулатуру.");
    return;
  }

  if (regionEnd <= regionStart) {
    alert("Выберите корректный отрезок времени на аудиоволне!");
    return;
  }

  const useDemucs = isStemMode || (chkDemucsTabs && chkDemucsTabs.checked);

  btnExtractTabs.disabled = true;
  tabBtnIcon.textContent = "⏳";
  tabBtnText.textContent = useDemucs && !isStemMode ? "Demucs AI + Ноты..." : "Анализ нот...";
  tabStatus.textContent = useDemucs && !isStemMode
    ? `Изоляция гитары через Demucs AI и генерация табов [${regionStart.toFixed(2)}s – ${regionEnd.toFixed(2)}s]...`
    : `Извлечение табулатур [${regionStart.toFixed(2)}s – ${regionEnd.toFixed(2)}s]...`;

  const payload = {
    file: currentTrack,
    start: regionStart,
    end: regionEnd,
    use_demucs: useDemucs,
    guitar_volume: isStemMode ? (guitarMuted ? 0 : guitarVolume) : 1.0,
  };

  try {
    const res = await fetch("/api/tabs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (data.error) {
      throw new Error(data.error);
    }

    asciiTabContent.textContent = data.tab;
    tabStatus.textContent = `Готово за ${data.elapsed_time}s • Ноты: ${data.num_notes}`;
    tabActions.style.display = "flex";
    renderNotePills(data.notes || []);

  } catch (err) {
    tabStatus.textContent = "Ошибка генерации табов";
    asciiTabContent.textContent = `Ошибка: ${err.message}`;
  } finally {
    updateGuitarRestrictionUI();
    tabBtnIcon.textContent = "⚡";
    tabBtnText.textContent = "Снять табы кусочка";
  }
}

function renderNotePills(notes) {
  notesChips.innerHTML = "";
  if (!notes.length) {
    notesSummary.style.display = "none";
    return;
  }

  notesSummary.style.display = "block";
  const noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

  notes.forEach((n) => {
    const chip = document.createElement("span");
    chip.className = "note-chip";
    const noteName = noteNames[n.pitch % 12];
    const octave = Math.floor(n.pitch / 12) - 1;
    chip.innerHTML = `<strong>${noteName}${octave}</strong> (${n.start}s – ${n.end}s)`;
    notesChips.appendChild(chip);
  });
}

function copyTabToClipboard() {
  const text = asciiTabContent.textContent;
  navigator.clipboard.writeText(text).then(() => {
    btnCopyTab.textContent = "✓ Скопировано!";
    setTimeout(() => (btnCopyTab.textContent = "📋 Скопировать"), 2000);
  });
}

function downloadTabFile() {
  const text = asciiTabContent.textContent;
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `guitar_tab_${regionStart.toFixed(1)}_${regionEnd.toFixed(1)}.txt`;
  a.click();
}

// Start
document.addEventListener("DOMContentLoaded", init);
