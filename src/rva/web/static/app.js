// Dashboard client: WebSocket for live stats, REST polling for history and events.
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n ?? 0).toLocaleString();
const secs = (s) => (s == null ? "–" : `${Number(s).toFixed(1)} s`);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

let sessionId = null;

// ---------------------------------------------------------------- live stats (WebSocket)
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/stats?interval=0.5`);
  ws.onmessage = (msg) => render(JSON.parse(msg.data));
  ws.onclose = () => setTimeout(connect, 1500);
}

function render(s) {
  const state = s.status ?? "idle";
  $("status").dataset.state = state;
  $("status-text").textContent = s.error ? `${state}: ${s.error}` : state;
  $("source-label").textContent = s.source ?? "—";
  if (s.session_id && s.session_id !== sessionId) {
    sessionId = s.session_id;
    refreshEvents();
  }
  const t = s.totals ?? {};
  $("kpi-in").textContent = fmt(t.in);
  $("kpi-out").textContent = fmt(t.out);
  $("kpi-occ").textContent = fmt(t.occupancy);
  $("kpi-vis").textContent = fmt(t.visible);
  $("kpi-fps").textContent = Number(s.fps ?? 0).toFixed(1);
  $("kpi-frames").textContent = fmt(s.frames_processed);

  if (s.zones?.length) {
    $("zones-body").innerHTML = s.zones.map((z) => `<tr>
      <td>${esc(z.name)}</td><td class="num">${z.occupancy}</td><td class="num">${z.total_visits}</td>
      <td class="num">${secs(z.avg_dwell_s)}</td><td class="num">${secs(z.max_dwell_s)}</td></tr>`).join("");
  }
  if (s.lines?.length) {
    $("lines-body").innerHTML = s.lines.map((l) => {
      const by = Object.entries(l.by_class ?? {}).map(([c, v]) => `${esc(c)} ${v.in}/${v.out}`).join(", ") || "–";
      return `<tr><td>${esc(l.name)}</td><td class="num">${l.in}</td><td class="num">${l.out}</td><td class="muted">${by}</td></tr>`;
    }).join("");
  }
}

// ---------------------------------------------------------------- events table
const EVENT_LABEL = { line_cross: "Line cross", zone_enter: "Zone enter", zone_exit: "Zone exit" };
const EVENT_COLOR = { line_cross: "var(--series-2)", zone_enter: "var(--series-1)", zone_exit: "var(--text-muted)" };

async function refreshEvents() {
  const q = sessionId ? `&session_id=${sessionId}` : "";
  const rows = await fetch(`/api/events?limit=60${q}`).then((r) => r.json()).catch(() => null);
  if (!rows) return;
  $("events-count").textContent = `${rows.length} most recent · persisted to SQLite`;
  if (!rows.length) return;
  $("events-body").innerHTML = rows.map((e) => {
    const detail = e.type === "line_cross" ? e.direction : e.dwell_s != null ? `dwell ${secs(e.dwell_s)}` : "";
    return `<tr><td class="muted">${esc(e.ts.slice(11, 19))}</td>
      <td><span class="tag" style="--c:${EVENT_COLOR[e.type]}">${EVENT_LABEL[e.type] ?? esc(e.type)}</span></td>
      <td>${esc(e.name)}</td><td>#${e.track_id}</td><td>${esc(e.class_name)}</td><td class="num">${esc(detail)}</td></tr>`;
  }).join("");
}

// ---------------------------------------------------------------- occupancy chart (SVG)
const NS = "http://www.w3.org/2000/svg";
const PAD = { l: 34, r: 10, t: 10, b: 22 };
let points = [];

function el(name, attrs, parent) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  parent?.appendChild(n);
  return n;
}

function niceMax(v) {
  if (v <= 4) return 4;
  const step = 10 ** Math.floor(Math.log10(v));
  return Math.ceil(v / step) * step;
}

function drawChart() {
  const svg = $("chart-svg");
  const { width: W, height: H } = svg.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
  const ymax = niceMax(Math.max(1, ...points.map((p) => p.occupancy)));
  const n = Math.max(points.length - 1, 1);
  const x = (i) => PAD.l + (i / n) * iw;
  const y = (v) => PAD.t + ih - (v / ymax) * ih;

  for (let k = 0; k <= 4; k++) {
    const v = (ymax / 4) * k;
    el("line", { x1: PAD.l, x2: W - PAD.r, y1: y(v), y2: y(v), class: k ? "gridline" : "baseline" }, svg);
    el("text", { x: PAD.l - 6, y: y(v) + 4, "text-anchor": "end", class: "tick" }, svg).textContent = Number.isInteger(v) ? v : v.toFixed(1);
  }
  if (points.length) {
    const ends = [points[0], points[points.length - 1]];
    ends.forEach((p, j) => {
      el("text", { x: j ? W - PAD.r : PAD.l, y: H - 4, "text-anchor": j ? "end" : "start", class: "tick" }, svg)
        .textContent = p.ts.slice(11, 19);
    });
  }
  if (points.length > 1) {
    const d = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.occupancy).toFixed(1)}`).join("");
    el("path", { d, class: "series" }, svg);
  }
  const cross = el("line", { y1: PAD.t, y2: PAD.t + ih, class: "crosshair", visibility: "hidden" }, svg);
  const marker = el("circle", { r: 4.5, class: "marker", visibility: "hidden" }, svg);
  const tip = $("tooltip");
  const hit = el("rect", { x: PAD.l, y: PAD.t, width: iw, height: ih, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (ev) => {
    if (!points.length) return;
    const box = svg.getBoundingClientRect();
    const i = Math.round(((ev.clientX - box.left - PAD.l) / iw) * n);
    const p = points[Math.min(Math.max(i, 0), points.length - 1)];
    const idx = points.indexOf(p);
    for (const [node, vis] of [[cross, "visible"], [marker, "visible"]]) node.setAttribute("visibility", vis);
    cross.setAttribute("x1", x(idx)); cross.setAttribute("x2", x(idx));
    marker.setAttribute("cx", x(idx)); marker.setAttribute("cy", y(p.occupancy));
    tip.hidden = false;
    tip.style.left = `${x(idx)}px`;
    tip.style.top = `${y(p.occupancy) - 10}px`;
    tip.innerHTML = `<span class="t">${esc(p.ts.slice(11, 19))} UTC</span><br><b>${p.occupancy}</b> in zones · in ${p.in} / out ${p.out}`;
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    marker.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
}

async function refreshHistory() {
  const rows = await fetch("/api/history?limit=300").then((r) => r.json()).catch(() => null);
  if (rows) { points = rows; drawChart(); }
}

// ---------------------------------------------------------------- controls
async function post(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  if (!res.ok) alert(`Request failed: ${(await res.json()).detail ?? res.status}`);
}

$("source-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const raw = $("source-input").value.trim();
  await post("/api/source/start", raw ? { uri: /^\d+$/.test(raw) ? Number(raw) : raw } : null);
  reloadStream();
});
$("stop-btn").addEventListener("click", () => post("/api/source/stop"));

function reloadStream() { $("video").src = `/stream.mjpg?t=${Date.now()}`; }
$("video").addEventListener("error", () => setTimeout(reloadStream, 1500));
window.addEventListener("resize", drawChart);

connect();
refreshHistory();
refreshEvents();
setInterval(refreshHistory, 2000);
setInterval(refreshEvents, 2000);
