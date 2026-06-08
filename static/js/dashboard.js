const el = (id) => document.getElementById(id);

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`Failed: ${url}`);
  return res.json();
}

function setBar(id, value) {
  const v = Math.max(0, Math.min(100, value || 0));
  const bar = el(id);
  bar.style.width = `${v}%`;
  bar.textContent = `${v.toFixed(0)}%`;
}

async function refreshStats() {
  const s = await fetchJSON("/stats");
  el("liveTime").textContent = s.timestamp;
  el("ovFps").textContent = s.fps;
  el("ovActive").textContent = s.active_ids_total;
  el("ovDetections").textContent = s.total_detections;
  el("c1Count").textContent = s.c1.total_count;
  el("c1Active").textContent = s.c1.active_ids.length;
  el("c1Status").textContent = s.c1.status;
  el("c1Last").textContent = s.c1.last_detection_time;
  el("c2Count").textContent = s.c2.total_count;
  el("c2Active").textContent = s.c2.active_ids.length;
  el("c2Status").textContent = s.c2.status;
  el("c2Last").textContent = s.c2.last_detection_time;
  el("sysState").textContent = s.recording ? "ONLINE" : "PAUSED";
  el("recordDot").style.opacity = s.recording ? "1" : "0.35";
  el("btnToggle").textContent = s.recording ? "Pause" : "Start";
  el("modelName").textContent = s.model_name;
  el("gpuName").textContent = s.gpu_device;
  el("verInfo").textContent = s.version;
}

async function refreshHealth() {
  const h = await fetchJSON("/health");
  setBar("gpuBar", h.gpu_usage);
  setBar("cpuBar", h.cpu_usage);
  setBar("ramBar", h.ram_usage);
  el("cameraStatus").textContent = h.camera_status;
  el("inferSpeed").textContent = h.inference_speed_ms;
}

async function refreshEvents() {
  const payload = await fetchJSON("/events");
  el("eventLog").innerHTML = payload.events.map((line) => `<div>${line}</div>`).join("");
}

async function toggleInference() {
  await fetchJSON("/toggle_inference", { method: "POST" });
}

async function snapshot() {
  const res = await fetchJSON("/snapshot", { method: "POST" });
  if (res.ok) {
    console.log("snapshot saved", res.path);
  }
}

function enterFullscreen() {
  const target = el("videoContainer");
  if (target.requestFullscreen) target.requestFullscreen();
}

el("btnToggle").addEventListener("click", toggleInference);
el("btnSnapshot").addEventListener("click", snapshot);
el("btnFullscreen").addEventListener("click", enterFullscreen);

setInterval(() => refreshStats().catch(console.error), 500);
setInterval(() => refreshHealth().catch(console.error), 1000);
setInterval(() => refreshEvents().catch(console.error), 1000);
refreshStats().catch(console.error);
refreshHealth().catch(console.error);
refreshEvents().catch(console.error);
