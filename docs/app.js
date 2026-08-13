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
        `<div class="diagram"><h4>${escapeHtml(d.title)}</h4><pre class="mermaid" id="m-${id}-${i}">${String(d.code).replace(/&/g, "&amp;").replace(/</g, "&lt;")}</pre></div>`
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
  if (window.mermaid) {
    window.mermaid.run({ querySelector: "#d-lesson .mermaid" }).catch(() => {});
  }
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
  if (e.key === "Escape") closeDrawer();
});
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
