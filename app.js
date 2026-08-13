const $ = (sel) => document.querySelector(sel);

const state = {
  data: null,
  lessons: {},
  category: "all",
  status: "all",
  q: "",
};

const statusLabel = {
  ok: "成功",
  fail: "失败",
  running: "运行中",
  pending: "等待",
};

if (window.mermaid) {
  window.mermaid.initialize({
    startOnLoad: false,
    theme: "dark",
    securityLevel: "loose",
    fontFamily: "Outfit, IBM Plex Mono, sans-serif",
    fontSize: 18,
    flowchart: {
      useMaxWidth: false,
      htmlLabels: true,
      curve: "basis",
      padding: 16,
      nodeSpacing: 48,
      rankSpacing: 56,
    },
    themeVariables: {
      darkMode: true,
      background: "#080c11",
      primaryColor: "#15443d",
      primaryTextColor: "#e8eef4",
      primaryBorderColor: "#3ee0b3",
      lineColor: "#7cf0d0",
      secondaryColor: "#2a1d18",
      tertiaryColor: "#121820",
      nodeTextColor: "#e8eef4",
      fontSize: "18px",
    },
  });
}

function imgUrl(project, name) {
  if (!name) return "";
  return `gallery/${project.id}/${encodeURIComponent(name)}`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderChips(categories) {
  const el = $("#chips");
  const all = [{ id: "all", label: "全部方向" }, ...categories];
  el.innerHTML = all
    .map(
      (c) =>
        `<button class="chip ${state.category === c.id ? "active" : ""}" data-cat="${c.id}">${c.label}</button>`
    )
    .join("");
  el.querySelectorAll(".chip").forEach((btn) => {
    btn.onclick = () => {
      state.category = btn.dataset.cat;
      render();
    };
  });
}

function visibleProjects() {
  const q = state.q.trim().toLowerCase();
  return state.data.projects.filter((p) => {
    if (state.category !== "all" && p.category !== state.category) return false;
    if (state.status !== "all" && p.status !== state.status) return false;
    if (!q) return true;
    const lesson = state.lessons[p.id];
    const extra = lesson
      ? `${lesson.subtitle} ${(lesson.sections || []).map((s) => s.h + s.p).join(" ")}`
      : "";
    return `${p.id} ${p.title} ${p.stack} ${p.entry} ${extra}`.toLowerCase().includes(q);
  });
}

function renderGrid() {
  const items = visibleProjects();
  const el = $("#grid");
  if (!items.length) {
    el.innerHTML = `<p class="empty">没有匹配的实验。</p>`;
    return;
  }
  el.innerHTML = items
    .map((p) => {
      const cover = p.images && p.images[0] ? `style="background-image:url('${imgUrl(p, p.images[0])}')"` : "";
      const lv = "Lv." + (p.level || "?");
      const sub = (state.lessons[p.id] && state.lessons[p.id].subtitle) || p.stack;
      return `<article class="card" data-id="${p.id}">
        <div class="thumb" ${cover}>
          <span class="badge">${p.id}</span>
          <span class="status ${p.status}">${statusLabel[p.status] || p.status}</span>
        </div>
        <div class="card-body">
          <h3>${p.title}</h3>
          <p>${p.category_label} · ${sub}</p>
          <div class="lv">${lv}${p.elapsed_sec != null ? " · " + p.elapsed_sec.toFixed(2) + "s" : ""}</div>
        </div>
      </article>`;
    })
    .join("");
  el.querySelectorAll(".card").forEach((card) => {
    card.onclick = () => openDrawer(card.dataset.id);
  });
}

function renderTracks() {
  const el = $("#track-list");
  el.innerHTML = (state.data.categories || [])
    .map((c) => {
      const n = state.data.projects.filter((p) => p.category === c.id && p.status === "ok").length;
      const t = state.data.projects.filter((p) => p.category === c.id).length;
      return `<div class="track"><b>${c.label}</b><span>${c.range} · ${n}/${t} 已出图</span></div>`;
    })
    .join("");
}

function renderStats() {
  $("#stat-done").textContent = state.data.completed ?? 0;
  $("#stat-fail").textContent = state.data.failed ?? 0;
  $("#stat-pending").textContent = state.data.pending ?? 0;
  $("#stat-gpu").textContent = "T4";
  const ts = state.data.updated_at;
  $("#updated").textContent = ts
    ? `最近一次 Action 回写 · ${ts} · 目标 GPU ${state.data.gpu_target || "T4"} · 点卡片看讲解`
    : "等待第一次 T4 回写…";
}

function render() {
  if (!state.data) return;
  renderChips(state.data.categories || []);
  renderStats();
  renderGrid();
  renderTracks();
}

function renderLesson(id) {
  const lesson = state.lessons[id];
  const el = $("#d-lesson");
  if (!lesson) {
    el.innerHTML = "";
    return;
  }
  const sections = (lesson.sections || [])
    .map(
      (s) =>
        `<article><h4>${escapeHtml(s.h)}</h4><p>${escapeHtml(s.p)}</p></article>`
    )
    .join("");
  const terms = (lesson.terms || [])
    .map((t) => `<div><dt>${escapeHtml(t.k)}</dt><dd>${escapeHtml(t.v)}</dd></div>`)
    .join("");
  const diagrams = (lesson.diagrams || [])
    .map(
      (d, i) =>
        `<div class="diagram" data-title="${escapeHtml(d.title)}"><h4>${escapeHtml(d.title)}</h4><p class="hint">点击图放大 · 滚轮缩放 · 拖拽移动</p><pre class="mermaid" id="m-${id}-${i}">${String(d.code).replace(/&/g, "&amp;").replace(/</g, "&lt;")}</pre></div>`
    )
    .join("");
  el.innerHTML = `
    <h3>给小白的讲解</h3>
    ${sections}
    <h3>名词对照</h3>
    <dl class="terms">${terms}</dl>
    <h3>流程图解</h3>
    ${diagrams}
  `;
  const run = window.mermaid
    ? window.mermaid.run({ querySelector: "#d-lesson .mermaid" })
    : Promise.resolve();
  Promise.resolve(run)
    .catch(() => {})
    .then(() => bindDiagramZoom(el));
}

function openDrawer(id) {
  const p = state.data.projects.find((x) => x.id === id);
  if (!p) return;
  const lesson = state.lessons[id];
  const drawer = $("#drawer");
  drawer.hidden = false;
  document.body.style.overflow = "hidden";
  $("#d-kicker").textContent = `${p.id} · ${p.category_label} · ${p.stack}`;
  $("#d-title").textContent = p.title;
  $("#d-sub").textContent = lesson
    ? lesson.subtitle
    : p.error
      ? `运行失败：${p.error}`
      : "T4 已跑完，产物由 GitHub Action 写入 Pages。";

  const images = p.images || [];
  const show = (name) => {
    $("#d-hero").innerHTML = name
      ? `<img alt="${p.title}" src="${imgUrl(p, name)}" />`
      : `<div class="thumb"></div>`;
    $("#d-thumbs").innerHTML = images
      .map(
        (n) =>
          `<img class="${n === name ? "active" : ""}" alt="${n}" src="${imgUrl(p, n)}" data-name="${n}" />`
      )
      .join("");
    $("#d-thumbs").querySelectorAll("img").forEach((img) => {
      img.onclick = () => show(img.dataset.name);
    });
  };
  show(images[0]);

  $("#d-facts").innerHTML = `
    <dt>状态</dt><dd>${statusLabel[p.status] || p.status}</dd>
    <dt>耗时</dt><dd>${p.elapsed_sec == null ? "—" : p.elapsed_sec + " s"}</dd>
    <dt>入口</dt><dd>${p.entry}</dd>
    <dt>GPU</dt><dd>${(p.gpu || "NVIDIA Tesla T4").replace(/\u0000/g, "").trim()}</dd>
    <dt>更新</dt><dd>${p.updated_at || "—"}</dd>
  `;
  $("#d-log").textContent = (p.log || "暂无日志").replace(/\u0000/g, "");
  renderLesson(id);
}

function closeDrawer() {
  $("#drawer").hidden = true;
  document.body.style.overflow = "";
}

$("#close").onclick = closeDrawer;
$("#drawer").addEventListener("click", (e) => {
  if (e.target.id === "drawer") closeDrawer();
});
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#zoom").hidden) {
    closeZoom();
    return;
  }
  closeDrawer();
});
const zoom = {
  scale: 1,
  x: 0,
  y: 0,
  dragging: false,
  px: 0,
  py: 0,
  min: 0.25,
  max: 8,
};

function bindDiagramZoom(root) {
  root.querySelectorAll(".diagram").forEach((box) => {
    box.addEventListener("click", (e) => {
      if (e.target.closest("a")) return;
      const svg = box.querySelector("svg");
      if (!svg) return;
      openZoom(svg, box.dataset.title || "流程图");
    });
  });
}

function applyZoom() {
  const canvas = $("#zoom-canvas");
  canvas.style.transform = `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.scale})`;
  $("#zoom-level").textContent = Math.round(zoom.scale * 100) + "%";
}

function fitZoom() {
  const stage = $("#zoom-stage");
  const svg = $("#zoom-canvas svg");
  if (!svg) return;
  const pad = 80;
  let w = 800;
  let h = 500;
  try {
    const box = svg.getBBox();
    w = Math.max(box.width, 1);
    h = Math.max(box.height, 1);
  } catch (err) {
    w = svg.clientWidth || w;
    h = svg.clientHeight || h;
  }
  const sx = (stage.clientWidth - pad) / w;
  const sy = (stage.clientHeight - pad) / h;
  zoom.scale = Math.min(Math.max(Math.min(sx, sy), zoom.min), 3);
  zoom.x = (stage.clientWidth - w * zoom.scale) / 2;
  zoom.y = (stage.clientHeight - h * zoom.scale) / 2;
  applyZoom();
}

function openZoom(svg, title) {
  const clone = svg.cloneNode(true);
  clone.removeAttribute("width");
  clone.removeAttribute("height");
  clone.removeAttribute("style");
  clone.style.maxWidth = "none";
  clone.style.width = "auto";
  clone.style.height = "auto";
  $("#zoom-title").textContent = title;
  $("#zoom-canvas").innerHTML = "";
  $("#zoom-canvas").appendChild(clone);
  $("#zoom").hidden = false;
  zoom.scale = 1.6;
  zoom.x = 40;
  zoom.y = 40;
  applyZoom();
  requestAnimationFrame(() => requestAnimationFrame(fitZoom));
}

function closeZoom() {
  $("#zoom").hidden = true;
  $("#zoom-canvas").innerHTML = "";
}

function zoomBy(factor, cx, cy) {
  const stage = $("#zoom-stage");
  const rect = stage.getBoundingClientRect();
  const px = cx == null ? rect.width / 2 : cx - rect.left;
  const py = cy == null ? rect.height / 2 : cy - rect.top;
  const next = Math.min(zoom.max, Math.max(zoom.min, zoom.scale * factor));
  const k = next / zoom.scale;
  zoom.x = px - (px - zoom.x) * k;
  zoom.y = py - (py - zoom.y) * k;
  zoom.scale = next;
  applyZoom();
}

$("#zoom-close").onclick = closeZoom;
$("#zoom").querySelectorAll("[data-z]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const act = btn.dataset.z;
    if (act === "in") zoomBy(1.25);
    else if (act === "out") zoomBy(0.8);
    else if (act === "reset") {
      zoom.scale = 1;
      zoom.x = 40;
      zoom.y = 40;
      applyZoom();
    } else if (act === "fit") fitZoom();
  });
});

$("#zoom-stage").addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    zoomBy(e.deltaY < 0 ? 1.12 : 0.9, e.clientX, e.clientY);
  },
  { passive: false }
);

$("#zoom-stage").addEventListener("dblclick", (e) => {
  zoomBy(1.45, e.clientX, e.clientY);
});
$("#zoom-stage").addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  zoom.dragging = true;
  zoom.px = e.clientX - zoom.x;
  zoom.py = e.clientY - zoom.y;
  $("#zoom-stage").classList.add("dragging");
  $("#zoom-stage").setPointerCapture(e.pointerId);
});
$("#zoom-stage").addEventListener("pointermove", (e) => {
  if (!zoom.dragging) return;
  zoom.x = e.clientX - zoom.px;
  zoom.y = e.clientY - zoom.py;
  applyZoom();
});
$("#zoom-stage").addEventListener("pointerup", () => {
  zoom.dragging = false;
  $("#zoom-stage").classList.remove("dragging");
});
$("#zoom-stage").addEventListener("pointercancel", () => {
  zoom.dragging = false;
  $("#zoom-stage").classList.remove("dragging");
});

let pinch = 0;
$("#zoom-stage").addEventListener(
  "touchstart",
  (e) => {
    if (e.touches.length === 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      pinch = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    }
  },
  { passive: true }
);
$("#zoom-stage").addEventListener(
  "touchmove",
  (e) => {
    if (e.touches.length !== 2 || !pinch) return;
    e.preventDefault();
    const a = e.touches[0];
    const b = e.touches[1];
    const dist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    const midX = (a.clientX + b.clientX) / 2;
    const midY = (a.clientY + b.clientY) / 2;
    zoomBy(dist / pinch, midX, midY);
    pinch = dist;
  },
  { passive: false }
);

$("#q").addEventListener("input", (e) => {
  state.q = e.target.value;
  renderGrid();
});
$("#status").addEventListener("change", (e) => {
  state.status = e.target.value;
  renderGrid();
});

Promise.all([
  fetch("data.json", { cache: "no-store" }).then((r) => r.json()),
  fetch("lessons.json", { cache: "no-store" }).then((r) => r.json()),
])
  .then(([data, lessons]) => {
    state.data = data;
    state.lessons = lessons;
    render();
  })
  .catch(() => {
    $("#updated").textContent = "无法加载 data.json / lessons.json";
  });
