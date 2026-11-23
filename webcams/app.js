"use strict";

const sides = ["left", "right"];
const el = id => document.getElementById(id);
const selects = Object.fromEntries(sides.map(side => [side, el(side + "-device")]));
const videos = Object.fromEntries(sides.map(side => [side, el(side + "-video")]));
const streams = {left: null, right: null};
let generation = 0;
let busy = false;
let urls = [];

function message(text, error = false) {
  el("status").textContent = text;
  el("status").classList.toggle("error", error);
}
function live(side) {
  return streams[side]?.getVideoTracks().some(track => track.readyState === "live");
}
function buttons() {
  el("connect").disabled = busy;
  el("refresh").disabled = busy;
  el("stop").disabled = !busy && !sides.some(live);
  el("capture").disabled = busy || !sides.every(side => live(side) && videos[side].readyState >= 2);
  sides.forEach(side => {selects[side].disabled = busy || !selects[side].options.length;});
}
function stopSide(side) {
  streams[side]?.getTracks().forEach(track => track.stop());
  streams[side] = null;
  videos[side].srcObject = null;
  el(side + "-empty").hidden = false;
  el(side + "-info").textContent = "Offline";
}
function stop() {
  generation++;
  busy = false;
  sides.forEach(stopSide);
  buttons();
  message("Cameras stopped.");
}
function explain(error) {
  const reasons = {
    NotAllowedError: "Camera access was denied. Allow cameras in the browser's site settings, then connect again.",
    NotReadableError: "Could not open a camera. Close other camera apps and check USB connections or bandwidth.",
    NotFoundError: "No camera was found. Check the USB connections and refresh devices.",
    OverconstrainedError: "The selected camera is unavailable. Refresh devices and try again."
  };
  return reasons[error.name] || error.message || "Camera connection failed.";
}
async function listDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) throw new Error("Open http://localhost:8768 in Safari or Chrome.");
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter(item => item.kind === "videoinput");
  const old = sides.map(side => selects[side].value);
  sides.forEach(side => selects[side].replaceChildren());
  cameras.forEach((camera, index) => sides.forEach(side =>
    selects[side].add(new Option((camera.label || "Camera") + " · " + (index + 1), camera.deviceId))));
  const preferred = [...cameras].sort((a, b) => Number(/a4tech/i.test(b.label)) - Number(/a4tech/i.test(a.label)));
  const used = new Set();
  sides.forEach((side, index) => {
    const next = cameras.find(camera => camera.deviceId === old[index] && !used.has(camera.deviceId)) ||
      preferred.find(camera => !used.has(camera.deviceId));
    selects[side].value = next?.deviceId || "";
    if (next) used.add(next.deviceId);
  });
  return cameras;
}
async function open(side, id, run) {
  if (!id) return;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {deviceId: {exact: id}, width: {ideal: 1920}, height: {ideal: 1080}, frameRate: {ideal: 30}}
  });
  if (run !== generation) {
    stream.getTracks().forEach(track => track.stop());
    return;
  }
  streams[side] = stream;
  const track = stream.getVideoTracks()[0];
  track.addEventListener("ended", () => {
    if (streams[side] !== stream) return;
    stopSide(side);
    buttons();
    message("The " + side + " camera disconnected. Reconnect it and connect cameras again.", true);
  });
  videos[side].srcObject = stream;
  await videos[side].play();
  if (run !== generation) return;
  const settings = track.getSettings();
  el(side + "-info").textContent = settings.width + " × " + settings.height;
  el(side + "-empty").hidden = true;
}
async function connect() {
  if (busy) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    message("Open http://localhost:8768 in Safari or Chrome to access cameras.", true);
    return;
  }
  const run = ++generation;
  busy = true;
  sides.forEach(stopSide);
  buttons();
  message("Allow camera access when your browser asks.");
  try {
    const permission = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
    permission.getTracks().forEach(track => track.stop());
    if (run !== generation) return;
    const cameras = await listDevices();
    if (run !== generation) return;
    if (!cameras.length) throw new Error("No cameras detected.");
    await open("left", selects.left.value, run);
    if (run !== generation) return;
    await open("right", selects.right.value, run);
    if (run !== generation) return;
    message(cameras.length < 2 ? "Only one camera is available. Connect a second webcam for paired capture." :
      "Both cameras are live. Choose devices above to change viewpoints.");
  } catch (error) {
    if (run === generation) message(explain(error), true);
  } finally {
    if (run === generation) {busy = false; buttons();}
  }
}
for (const side of sides) {
  videos[side].addEventListener("loadeddata", buttons);
  selects[side].addEventListener("change", () => {
    if (selects.left.value && selects.left.value === selects.right.value) {
      const other = side === "left" ? "right" : "left";
      selects[other].value = [...selects[other].options].find(option => option.value !== selects[side].value)?.value || "";
    }
    connect();
  });
}
function png(canvas) {
  return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Image encoding failed.")), "image/png"));
}
async function capture() {
  if (el("capture").disabled) return;
  const run = generation;
  busy = true;
  buttons();
  try {
    // Draw both frames in one event-loop turn before encoding either image.
    const frames = sides.map(side => {
      const canvas = document.createElement("canvas");
      canvas.width = videos[side].videoWidth;
      canvas.height = videos[side].videoHeight;
      canvas.getContext("2d").drawImage(videos[side], 0, 0);
      return canvas;
    });
    const pair = document.createElement("canvas");
    pair.width = frames[0].width + frames[1].width;
    pair.height = Math.max(...frames.map(frame => frame.height));
    const context = pair.getContext("2d");
    context.fillStyle = "#101716";
    context.fillRect(0, 0, pair.width, pair.height);
    context.drawImage(frames[0], 0, 0);
    context.drawImage(frames[1], frames[0].width, 0);
    const blobs = await Promise.all([pair, ...frames].map(png));
    if (run !== generation) return;
    urls.forEach(url => URL.revokeObjectURL(url));
    urls = blobs.map(blob => URL.createObjectURL(blob));
    const stamp = new Date().toISOString().replaceAll(":", "-");
    ["pair", "left", "right"].forEach((name, index) => {
      el(name + "-download").href = urls[index];
      el(name + "-download").download = stamp + "-" + name + ".png";
    });
    el("preview").src = urls[0];
    el("details").textContent = new Date().toLocaleTimeString() + " · " +
      frames.map((frame, index) => sides[index] + " " + frame.width + " × " + frame.height).join(" / ");
    el("result").hidden = false;
    message("Pair captured. Download the frames for calibration or reconstruction.");
  } catch (error) {
    message(error.message, true);
  } finally {
    if (run === generation) {busy = false; buttons();}
  }
}
el("connect").addEventListener("click", connect);
el("stop").addEventListener("click", stop);
el("capture").addEventListener("click", capture);
el("refresh").addEventListener("click", async () => {
  try {
    await listDevices();
    buttons();
    message("Device list refreshed. Connect cameras to apply the selections.");
  } catch (error) {message(explain(error), true);}
});
navigator.mediaDevices?.addEventListener("devicechange", () => message("Devices changed. Refresh devices and connect cameras again."));
window.addEventListener("pagehide", () => {
  generation++;
  sides.forEach(stopSide);
  urls.forEach(url => URL.revokeObjectURL(url));
});
buttons();
