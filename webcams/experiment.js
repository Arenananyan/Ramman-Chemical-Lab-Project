"use strict";

const experimentMode = el("experiment-mode");
const chatInput = el("chat-input");
const chatLog = el("chat-log");
let experimentRun = null;
let phase = "idle";
let sending = false;

function addMessage(role, message) {
  const line = document.createElement("div");
  const speaker = document.createElement("strong");
  const body = document.createElement("p");
  line.className = "chat-line";
  line.dataset.role = role;
  speaker.textContent = role;
  body.textContent = message;
  line.append(speaker, body);
  chatLog.append(line);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function showExperimentState() {
  const awaiting = phase === "awaiting";
  const labels = {idle: "Ready", awaiting: "Check head position", cancelled: "Cancelled",
    completed: "Complete", expired: "Position check expired", failed: "Stopped"};
  el("run-phase").textContent = sending ? "Working…" : labels[phase] || phase;
  el("position-check").hidden = !awaiting;
  el("approve-position").disabled = sending;
  el("reject-position").disabled = sending;
  el("start-demo").disabled = sending || awaiting;
  el("chat-form").querySelector("button").disabled = sending;
  experimentMode.disabled = sending || awaiting;
  el("live-check").hidden = experimentMode.value !== "live";
}

function frameBase64(canvas) {
  return png(canvas).then(blob => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",", 2)[1]);
    reader.onerror = () => reject(new Error("Could not read the captured frame."));
    reader.readAsDataURL(blob);
  }));
}

async function demoFrames() {
  if (!sides.every(side => live(side) && videos[side].readyState >= 2 &&
      videos[side].videoWidth === 1024 && videos[side].videoHeight === 576)) {
    throw new Error("Connect both cameras in Calibrated pair mode at 1024 × 576.");
  }
  // Capture calibrated, unadjusted frames before any printer movement.
  const frames = sides.map(side => {
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 576;
    drawExposure(canvas, videos[side], 1);
    return canvas;
  });
  return Promise.all(frames.map(frameBase64));
}

async function sendExperiment(text) {
  if (sending || !text) return;
  sending = true;
  showExperimentState();
  addMessage("You", text);
  const mode = experimentMode.value;
  const intent = text.toLowerCase().replace(/[^a-z0-9 ]/g, "").trim();
  try {
    const request = {text, mode, run_id: experimentRun};
    if (mode === "live" && ["start", "start demo", "run demo", "start experiment"].includes(intent)) {
      if (!el("live-confirmed").checked) throw new Error("Check the live setup before starting.");
      request.verified_setup = true;
      request.frames = await demoFrames();
      el("live-confirmed").checked = false;
      addMessage("Cameras", "Captured a calibrated image pair. Locating the cylinder.");
    }
    const response = await fetch("/experiment/chat", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(request)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Experiment request failed.");
    phase = result.phase;
    experimentRun = result.run_id;
    for (const message of result.messages) addMessage(message.role, message.text);
  } catch (error) {
    addMessage("Coordinator", error.message);
  } finally {
    sending = false;
    showExperimentState();
  }
}

el("start-demo").addEventListener("click", () => sendExperiment("start demo"));
el("approve-position").addEventListener("click", () => sendExperiment("yes"));
el("reject-position").addEventListener("click", () => sendExperiment("no"));
el("chat-form").addEventListener("submit", event => {
  event.preventDefault();
  const text = chatInput.value.trim();
  chatInput.value = "";
  sendExperiment(text);
});
experimentMode.addEventListener("change", () => {
  showExperimentState();
  if (experimentMode.value === "live" && el("camera-resolution").value !== "calibrated") {
    el("camera-resolution").value = "calibrated";
    if (sides.some(live)) connect();
  }
});
fetch("/experiment/status").then(response => response.json()).then(config => {
  experimentMode.querySelector('[value="live"]').disabled = !config.live_available;
  el("experiment-engine").textContent = config.groq_model
    ? "Groq · " + config.groq_model : "Local command mode";
  el("experiment-availability").textContent = config.live_available
    ? "Use a single upright cylinder and water for the live demo."
    : config.reason;
}).catch(() => {
  el("experiment-availability").textContent = "Experiment service unavailable.";
});
