// Job buttons, SSE progress, live training charts, review keys and the side drawer.
(function () {
  "use strict";
  const page = () => document.getElementById("page");
  const charts = {};

  function setState(state) { page().dataset.jobState = state; }

  function panelFor(kind) { return document.getElementById("job-" + kind); }

  function showProgress(kind, j) {
    const el = panelFor(kind);
    if (!el) return;
    el.classList.add("show");
    el.classList.toggle("error", j.state === "error");
    el.classList.toggle("done", j.state === "done");
    const txt = el.querySelector(".txt");
    const spin = el.querySelector(".spin");
    if (txt) txt.textContent = j.state === "error" ? (j.error || "Failed") : (j.message || (j.state === "queued" ? "Starting" : ""));
    if (spin) spin.style.display = (j.state === "running" || j.state === "queued") ? "inline-block" : "none";
    const pct = j.total ? Math.round((100 * j.progress) / j.total) : (j.state === "done" ? 100 : 4);
    el.querySelector(".bar span").style.width = Math.max(4, Math.min(100, pct)) + "%";
    el.querySelector(".cnt").textContent = j.total ? j.progress + " / " + j.total : "";
  }

  async function refreshContent() {
    const res = await fetch(location.href, { headers: { "X-Partial": "1" } });
    const html = await res.text();
    const doc = new DOMParser().parseFromString(html, "text/html");
    const fresh = doc.getElementById("content");
    if (fresh) {
      document.getElementById("content").replaceWith(fresh);
      if (window.htmx) htmx.process(fresh);
      initPage();
    }
  }

  function follow(jobId, kind) {
    setState("running");
    document.querySelectorAll("[data-job]").forEach((b) => (b.disabled = true));
    const es = new EventSource("/api/jobs/" + jobId + "/events");
    es.addEventListener("job", (ev) => {
      const j = JSON.parse(ev.data);
      setState(j.state === "queued" ? "running" : j.state);
      showProgress(kind, j);
      if (j.metrics) addMetrics(j.metrics);
    });
    es.addEventListener("end", async (ev) => {
      es.close();
      const j = JSON.parse(ev.data);
      await refreshContent();
      if (j.state === "error") {
        showProgress(kind, { state: "error", error: j.error });
      }
      setState(j.state);
    });
    es.onerror = () => { /* the browser retries; the stream ends with an "end" event */ };
  }

  async function startJob(kind) {
    setState("queued");
    showProgress(kind, { state: "queued", message: "Starting", progress: 0, total: 0 });
    const res = await fetch("/api/jobs/" + kind, { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      setState("error");
      showProgress(kind, { state: "error", error: body.detail || "Could not start the job" });
      return;
    }
    follow(body.id, kind);
  }

  // ---------- charts ----------
  function makeChart(id, color, yMin, yMax) {
    const el = document.getElementById(id);
    if (!el || !window.Chart) return null;
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(el, {
      type: "line",
      data: { labels: [], datasets: [{ data: [], borderColor: color, backgroundColor: color + "22", fill: true,
        tension: 0.3, pointRadius: 0, borderWidth: 2 }] },
      options: { animation: false, responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: "step" }, ticks: { maxTicksLimit: 7 }, grid: { display: false } },
          y: { min: yMin, max: yMax, ticks: { maxTicksLimit: 5 } } } },
    });
    return charts[id];
  }

  function addMetrics(rows) {
    const map = [["c-loss", "loss"], ["c-margin", "margin"], ["c-acc", "accuracy"]];
    for (const [id, key] of map) {
      const ch = charts[id];
      if (!ch) continue;
      for (const r of rows) {
        ch.data.labels.push(r.step);
        ch.data.datasets[0].data.push(r[key]);
      }
      ch.update();
    }
  }

  async function initCharts() {
    const box = document.getElementById("charts");
    if (!box) return;
    makeChart("c-loss", "#3554d1");
    makeChart("c-margin", "#13795b");
    makeChart("c-acc", "#6d3fc0", 0, 1);
    const run = box.dataset.run;
    if (run) {
      const res = await fetch("/api/train/" + run + "/metrics");
      if (res.ok) addMetrics(await res.json());
    }
  }

  // ---------- drawer ----------
  function openDrawer() {
    const body = document.getElementById("drawer-body");
    const t = body.querySelector("[data-drawer-title]");
    document.getElementById("drawer-title").textContent = t ? t.dataset.drawerTitle : "Details";
    document.getElementById("drawer").classList.add("open");
    document.getElementById("drawer-bg").classList.add("open");
  }
  function closeDrawer() {
    document.getElementById("drawer").classList.remove("open");
    document.getElementById("drawer-bg").classList.remove("open");
  }

  // ---------- review keys ----------
  let focusIdx = 0;
  function items() { return Array.from(document.querySelectorAll("#queue .ritem")); }
  function focusItem(i) {
    const list = items();
    if (!list.length) return;
    focusIdx = Math.max(0, Math.min(list.length - 1, i));
    list.forEach((el, k) => el.classList.toggle("focus", k === focusIdx));
    list[focusIdx].scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  function firstPending() {
    const list = items();
    const k = list.findIndex((el) => !el.dataset.decision);
    return k < 0 ? 0 : k;
  }

  function initPage() {
    initCharts();
    const tog = document.getElementById("note-toggle");
    if (tog) {
      tog.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
        tog.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
        document.getElementById("note-mode").value = b.dataset.view;
        const pid = document.getElementById("patient-pick").value;
        htmx.ajax("GET", "/partials/note?patient=" + encodeURIComponent(pid) + "&view=" + b.dataset.view, "#note-view");
      }));
    }
    document.querySelectorAll("#pair-filter .chip").forEach((c) => c.addEventListener("click", () => {
      document.querySelectorAll("#pair-filter .chip").forEach((x) => x.classList.toggle("on", x === c));
    }));
  }

  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-job]");
    if (b && !b.disabled) { e.preventDefault(); startJob(b.dataset.job); return; }
    if (e.target.closest("[data-close-drawer]") || e.target.id === "drawer-bg") closeDrawer();
    const row = e.target.closest("tr[data-pair]");
    if (row) {
      document.querySelectorAll("tr[data-pair]").forEach((r) => r.classList.toggle("sel", r === row));
    }
  });

  document.addEventListener("htmx:afterSwap", (e) => {
    if (e.detail.target && e.detail.target.id === "drawer-body") openDrawer();
    if (e.detail.target && e.detail.target.closest && document.getElementById("queue")) {
      if (e.detail.elt && e.detail.elt.classList && e.detail.elt.classList.contains("ritem")) focusItem(focusIdx);
    }
  });
  document.addEventListener("htmx:load", (e) => {
    if (e.detail.elt && e.detail.elt.classList && e.detail.elt.classList.contains("ritem") && items().length) {
      focusItem(firstPending());
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeDrawer(); return; }
    if (e.target.matches("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    const list = items();
    if (!list.length) return;
    const k = e.key.toLowerCase();
    if (k === "j") { focusItem(focusIdx + 1); return; }
    if (k === "k") { focusItem(focusIdx - 1); return; }
    const act = { a: "approve", f: "flip", d: "drop" }[k];
    if (!act) return;
    const el = list[focusIdx];
    const btn = el && el.querySelector('[data-act="' + act + '"]');
    if (btn) {
      e.preventDefault();
      htmx.trigger(btn, "click");
      setTimeout(() => focusItem(firstPending()), 350);
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    initPage();
    const p = page();
    if (p.dataset.activeJob) follow(p.dataset.activeJob, p.dataset.activeKind);
  });
})();
