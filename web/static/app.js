(() => {
  "use strict";
  const D = window.SWEEP;
  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };
  const plain = (html) => html.replace(/<[^>]+>/g, "");
  const modelBy = Object.fromEntries(D.models.map((m) => [m.key, m]));
  const taskBy = Object.fromEntries(D.tasks.map((t) => [t.id, t]));
  const pct = (x) => `${Math.round(100 * x)}%`;
  const speedLabel = (s) => `${s} m/s`;

  // Lazy clips: a video gets its source when it scrolls into view and pauses when it leaves. Slides of a carousel
  // that are off to the side are clipped by the carousel, so they count as out of view.
  const watcher = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target;
      v.dataset.visible = e.isIntersecting ? "1" : "";
      if (e.isIntersecting) {
        if (v.dataset.src && v.getAttribute("src") !== v.dataset.src) v.src = v.dataset.src;
        v.play().catch(() => {});
      } else if (!v.paused) {
        v.pause();
      }
    }
  }, { rootMargin: "120px 0px", threshold: 0.2 });
  const lazyVideo = (src, label) => {
    const v = el("video");
    Object.assign(v, { muted: true, loop: true, playsInline: true, preload: "none" });
    v.setAttribute("muted", "");
    v.setAttribute("playsinline", "");
    v.setAttribute("aria-label", label);
    v.dataset.src = src;
    watcher.observe(v);
    return v;
  };
  const setVideoSource = (v, src) => {
    if (v.dataset.src === src) return;
    v.dataset.src = src;
    if (v.dataset.visible) {
      v.src = src;
      v.play().catch(() => {});
    } else if (v.getAttribute("src")) {
      v.removeAttribute("src");
      v.load();
    }
  };
  // Unobserve, pause and unload the videos of a subtree that is about to be replaced, so no old media is retained.
  const releaseVideos = (node) => {
    for (const v of node.querySelectorAll("video")) {
      watcher.unobserve(v);
      v.pause();
      v.removeAttribute("src");
      v.load();
    }
  };
  const segmented = (node, values, label, onPick) => {
    const buttons = values.map((value) => {
      const b = el("button", "button is-small", label(value));
      b.type = "button";
      b.addEventListener("click", () => onPick(value));
      node.append(b);
      return [value, b];
    });
    return (current) => buttons.forEach(([v, b]) => {
      b.classList.toggle("is-selected", v === current);
      b.classList.toggle("is-dark", v === current);
      b.setAttribute("aria-pressed", String(v === current));
    });
  };
  const carousel = (selector, show) => window.bulmaCarousel.attach(selector, {
    slidesToShow: show, slidesToScroll: 1, loop: true, infinite: false, pagination: true, navigationKeys: false,
    breakpoints: [{ changePoint: 560, slidesToShow: 1, slidesToScroll: 1 }, { changePoint: 900, slidesToShow: 2, slidesToScroll: 1 },
      { changePoint: 1100, slidesToShow: 3, slidesToScroll: 1 }],
  });

  // ------------------------------------------------------------ teaser: three synced clips
  const speeds = D.curves.speeds;
  {
    const vids = [...document.querySelectorAll("#teaser video")];
    let ended = 0, started = false;
    const restart = () => { ended = 0; vids.forEach((x) => { x.currentTime = 0; x.play().catch(() => {}); }); };
    for (const v of vids) {
      v.src = v.dataset.src;
      v.addEventListener("ended", () => { ended += 1; if (ended === vids.length) restart(); });
    }
    new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !started) { started = true; restart(); }
      else if (!e.isIntersecting && started) { started = false; vids.forEach((x) => x.pause()); }
    }, { threshold: 0.3 }).observe($("#teaser"));
  }

  // ------------------------------------------------------------ tasks: carousel of a few, all of them on request
  {
    const clipSpeeds = D.gallerySpeeds;
    let speed = clipSpeeds[0];
    const mark = { 20: "†", 40: "‡", 80: "" };
    const card = (t, cls) => {
      const c = el("article", cls);
      const frame = el("div", "frame");
      const video = lazyVideo("", `${plain(t.name)}, scripted expert`);
      frame.append(video);
      c.append(frame, el("h4", null, t.name + mark[t.subset]), el("p", "meta", t.category), el("p", "instr", `“${t.instruction}”`));
      return { t, c, video };
    };
    const cards = D.carouselTasks.map((id) => card(taskBy[id], "item"));
    const strip = $("#task-carousel");
    cards.forEach((x) => strip.append(x.c));

    let gallery = null, category = null;
    const paint = segmented($("#task-speed"), clipSpeeds, (s) => `${speedLabel(s)} ${s === clipSpeeds[0] ? "(slow)" : "(fast)"}`, (s) => { speed = s; update(); });
    function update() {
      paint(speed);
      for (const x of cards.concat(gallery ? gallery.cards : [])) {
        setVideoSource(x.video, `static/vid/tasks/${x.t.id}-${speed}.mp4`);
      }
      if (gallery) {
        gallery.chips.forEach(([name, b]) => {
          b.classList.toggle("is-dark", name === (category || "All"));
          b.setAttribute("aria-pressed", String(name === (category || "All")));
        });
        let shown = 0;
        for (const x of gallery.cards) {
          const on = !category || x.t.category === category;
          x.c.hidden = !on;
          shown += on;
        }
        $("#gallery-count").textContent = `${shown} task${shown === 1 ? "" : "s"} at ${speedLabel(speed)}.`;
      }
    }
    // The 80 cards are built the first time the list is opened, so none of their clips load before that.
    $("#all-tasks").addEventListener("toggle", (ev) => {
      if (!ev.target.open || gallery) return;
      const chips = [["All", D.tasks.length], ...D.categories.map((c) => [c.name, c.count])].map(([name, n]) => {
        const b = el("button", "button is-small is-rounded", `${name}<span class="n">${n}</span>`);
        b.type = "button";
        b.addEventListener("click", () => { category = category === name || name === "All" ? null : name; update(); });
        $("#category-chips").append(b);
        return [name, b];
      });
      gallery = { chips, cards: D.tasks.map((t) => card(t, "gallery-card")) };
      gallery.cards.forEach((x) => $("#gallery").append(x.c));
      update();
    });
    update();
    carousel("#task-carousel", 4);
  }

  // ------------------------------------------------------------ speed curves
  // SVG is written as markup: the HTML parser puts <svg> content in the SVG namespace without a namespace URL.
  // The viewBox follows the container width (340 to 1100 units), so the 12-unit labels stay readable on phones.
  {
    let W, H, svg, layer, cursor, tip;
    const M = { l: 52, r: 14, t: 10, b: 42 };
    const x0 = 0.09, x1 = 0.41, y1 = 0.7;
    const X = (s) => M.l + ((s - x0) / (x1 - x0)) * (W - M.l - M.r);
    const Y = (r) => H - M.b - (r / y1) * (H - M.t - M.b);
    const yTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    const box = $("#chart");
    const chosen = new Set(["pi05", "fastwam"]);
    let input = "both";

    function frame() {
      W = Math.round(Math.min(1100, Math.max(340, box.clientWidth)));
      H = Math.round(Math.max(260, W * 0.3));
      const xTicks = W < 600 ? [0.1, 0.2, 0.3, 0.4] : [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4];
      const midY = (M.t + H - M.b) / 2;
      box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Success rate versus belt speed">
        <g class="grid">${yTicks.map((r) => `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(r)}" y2="${Y(r)}"></line>`).join("")}</g>
        <g class="axis">${yTicks.map((r) => `<text x="${M.l - 8}" y="${Y(r) + 4}" text-anchor="end">${pct(r)}</text>`).join("")}
          ${xTicks.map((s) => `<text x="${X(s)}" y="${H - M.b + 18}" text-anchor="middle">${s.toFixed(2)}</text>`).join("")}</g>
        <text class="axis-title" x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">Belt speed (m/s)</text>
        <text class="axis-title" x="14" y="${midY}" text-anchor="middle" transform="rotate(-90 14 ${midY})">Success rate</text>
        <g class="layer"></g>
        <line class="cursor" y1="${M.t}" y2="${H - M.b}" visibility="hidden"></line>
      </svg><div class="tip" hidden></div>`;
      svg = $("svg", box); layer = $(".layer", box); cursor = $(".cursor", box); tip = $(".tip", box);
      svg.addEventListener("pointermove", hover);
      svg.addEventListener("pointerdown", hover);
      svg.addEventListener("pointerleave", hideTip);
    }

    const chips = $("#model-chips");
    const allChip = el("button", "button is-small is-rounded", "All models");
    allChip.type = "button";
    allChip.addEventListener("click", () => {
      if (chosen.size === D.models.length) { chosen.clear(); chosen.add(D.models[0].key); } else D.models.forEach((m) => chosen.add(m.key));
      draw();
    });
    chips.append(allChip);
    const modelChips = D.models.map((m) => {
      const b = el("button", "button is-small is-rounded", `<span class="sw" style="background:${m.color}"></span>${m.label}`);
      b.type = "button";
      b.addEventListener("click", () => {
        if (chosen.has(m.key) && chosen.size > 1) chosen.delete(m.key); else chosen.add(m.key);
        draw();
      });
      chips.append(b);
      return [m.key, b];
    });
    const inputButtons = [...document.querySelectorAll("#input-toggle button")];
    inputButtons.forEach((b) => b.addEventListener("click", () => { input = b.dataset.input; draw(); }));

    const visible = () => D.curves.series.filter((s) => chosen.has(s.model) && (input === "both" || s.input === input));
    function hideTip() { tip.hidden = true; cursor.setAttribute("visibility", "hidden"); }
    function draw() {
      const press = (b, on) => { b.classList.toggle("is-dark", on); b.setAttribute("aria-pressed", String(on)); };
      press(allChip, chosen.size === D.models.length);
      modelChips.forEach(([k, b]) => press(b, chosen.has(k)));
      inputButtons.forEach((b) => press(b, b.dataset.input === input));
      layer.innerHTML = visible().map((s) => {
        const color = modelBy[s.model].color;
        const up = speeds.map((v, i) => `${X(v)},${Y(Math.min(y1, s.hi[i]))}`);
        const dn = speeds.map((v, i) => `${X(v)},${Y(Math.max(0, s.lo[i]))}`).reverse();
        const line = speeds.map((v, i) => `${X(v)},${Y(s.rate[i])}`).join("L");
        const dots = speeds.map((v, i) => `<circle cx="${X(v)}" cy="${Y(s.rate[i])}" r="3" stroke="${color}" fill="${color}"></circle>`).join("");
        return `<g class="series ${s.input}"><path class="band" d="M${up.join("L")}L${dn.join("L")}Z" fill="${color}"></path>` +
          `<path class="line" d="M${line}" stroke="${color}"></path>${dots}</g>`;
      }).join("");
      hideTip();
    }
    function hover(ev) {
      const rect = svg.getBoundingClientRect();
      const sx = ((ev.clientX - rect.left) / rect.width) * W;
      if (sx < M.l - 10 || sx > W - M.r + 10) { hideTip(); return; }
      let i = 0;
      speeds.forEach((v, k) => { if (Math.abs(X(v) - sx) < Math.abs(X(speeds[i]) - sx)) i = k; });
      cursor.setAttribute("x1", X(speeds[i]));
      cursor.setAttribute("x2", X(speeds[i]));
      cursor.setAttribute("visibility", "visible");
      const rows = visible().slice().sort((a, b) => b.rate[i] - a.rate[i])
        .map((s) => `<div class="row"><span><span class="sw" style="background:${modelBy[s.model].color}"></span>${modelBy[s.model].label}${s.input === "hist" ? " + past" : ", current"}</span><span>${pct(s.rate[i])} (${Math.round(100 * s.lo[i])}–${pct(s.hi[i])})</span></div>`);
      tip.innerHTML = `<b>${speeds[i].toFixed(Math.round(speeds[i] * 1000) % 10 ? 3 : 2)} m/s</b>${rows.join("")}`;
      tip.hidden = false;
      const px = (X(speeds[i]) / W) * rect.width;
      const left = px + 14 + tip.offsetWidth > rect.width ? px - 14 - tip.offsetWidth : px + 14;
      tip.style.left = `${Math.max(0, left)}px`;
      tip.style.top = `${Math.max(0, ev.clientY - rect.top - tip.offsetHeight / 2)}px`;
    }
    frame();
    draw();
    let lastWidth = box.clientWidth;
    window.addEventListener("resize", () => {
      if (Math.abs(box.clientWidth - lastWidth) > 40) { lastWidth = box.clientWidth; frame(); draw(); }
    });
  }

  // ------------------------------------------------------------ failures: one task, 8 models x 3 speeds
  {
    const qSpeeds = [...new Set(D.qual.map((e) => e.speed))].sort();
    const byKey = Object.fromEntries(D.qual.map((e) => [`${e.model}|${e.task}|${e.speed}`, e]));
    let task = D.qualTasks[0];
    const paint = segmented($("#qual-tabs"), D.qualTasks, (t) => taskBy[t].name, (t) => { task = t; update(); });
    const node = $("#qual-matrix");
    const narrow = window.matchMedia("(max-width: 768px)");
    const cell = (e) => {
      const fig = el("figure", "cell");
      const frame = el("div", "frame");
      frame.append(lazyVideo(`static/vid/fail/${e.file}`, `${plain(modelBy[e.model].label)}, ${plain(taskBy[e.task].name)}, ${e.speed} m/s: ${D.outcomes[e.outcome]}`));
      fig.append(frame, el("span", `label ${e.outcome}`, D.outcomes[e.outcome]));
      if (e.replayed) fig.append(el("span", "rerender", "re-rendered"));
      return fig;
    };
    function update() {
      paint(task);
      releaseVideos(node);
      node.replaceChildren();
      // Wide screens: speeds down, models across. Phones: models down, speeds across.
      const [rows, cols] = narrow.matches ? [D.models.map((m) => m.key), qSpeeds] : [qSpeeds, D.models.map((m) => m.key)];
      const head = (k) => (modelBy[k] ? modelBy[k].label : speedLabel(k));
      node.style.setProperty("--cols", cols.length);
      node.append(el("div", "corner"));
      cols.forEach((c) => node.append(el("div", "colhead", head(c))));
      for (const r of rows) {
        node.append(el("div", "rowhead", head(r)));
        cols.forEach((c) => {
          const [m, s] = narrow.matches ? [r, c] : [c, r];
          node.append(cell(byKey[`${m}|${task}|${s}`]));
        });
      }
      const shown = D.qual.filter((e) => e.task === task);
      const counts = Object.keys(D.outcomes).map((k) => [k, shown.filter((e) => e.outcome === k).length]).filter(([, n]) => n);
      const early = shown.filter((e) => e.speed === qSpeeds[0] && e.outcome === "early").length;
      const late = shown.filter((e) => e.speed === qSpeeds[qSpeeds.length - 1] && e.outcome === "late").length;
      $("#qual-count").textContent = `${shown.length} episodes: ${counts.map(([k, n]) => `${n} ${D.outcomes[k].toLowerCase()}`).join(", ")}. ` +
        `Too early at ${speedLabel(qSpeeds[0])}: ${early} of ${D.models.length} policies; too late at ${speedLabel(qSpeeds[qSpeeds.length - 1])}: ${late} of ${D.models.length}.`;
      $("#rerender-note").hidden = !shown.some((e) => e.replayed);
    }
    narrow.addEventListener("change", update);
    update();
  }

  // ------------------------------------------------------------ current vs past frames: one row per speed, one column per model and input
  {
    const cSpeeds = [...new Set(D.collapse.map((e) => e.speed))].sort();
    const rows = [...new Set(D.collapse.map((e) => `${e.model}|${e.input}`))].sort((a, b) => {
      const [ma, ia] = a.split("|"), [mb, ib] = b.split("|");
      return D.models.findIndex((m) => m.key === ma) - D.models.findIndex((m) => m.key === mb) || (ia === "orig" ? -1 : 1) - (ib === "orig" ? -1 : 1);
    });
    const byKey = Object.fromEntries(D.collapse.map((e) => [`${e.model}|${e.input}|${e.task}|${e.speed}`, e]));
    const inputLabel = { orig: "current frame", hist: "+ past frames" };
    const outcomeLabel = { success: "Success", failure: "Failure" };
    const withInput = { orig: "the current frame", hist: "past frames" };
    let task = D.collapseTasks[0];
    const paint = segmented($("#collapse-tabs"), D.collapseTasks, (t) => taskBy[t].name, (t) => { task = t; update(); });
    const node = $("#collapse-matrix");
    function update() {
      paint(task);
      releaseVideos(node);
      node.replaceChildren();
      node.style.setProperty("--cols", rows.length);
      node.append(el("div", "corner"));
      for (const r of rows) {
        const [m, input] = r.split("|");
        node.append(el("div", "colhead", `${modelBy[m].label}<span class="sub">${inputLabel[input]}</span>`));
      }
      for (const s of cSpeeds) {
        node.append(el("div", "rowhead", speedLabel(s)));
        for (const r of rows) {
          const [m, input] = r.split("|");
          const e = byKey[`${r}|${task}|${s}`];
          const fig = el("figure", "cell");
          const frame = el("div", "frame");
          frame.append(lazyVideo(`static/vid/collapse/${e.file}`,
            `${plain(modelBy[m].label)}, ${inputLabel[input]}, ${plain(taskBy[task].name)}, ${speedLabel(s)}: ${outcomeLabel[e.outcome]}`));
          fig.append(frame, el("span", `label ${e.outcome}`, outcomeLabel[e.outcome]));
          node.append(fig);
        }
      }
      const odd = D.collapse.filter((e) => e.task === task && e.differs);
      const note = $("#collapse-note");
      note.hidden = !odd.length;
      note.textContent = odd.map((e) => `${plain(modelBy[e.model].label)} with ${withInput[e.input]} at ${speedLabel(e.speed)}: ` +
        `this recording ${e.outcome === "success" ? "succeeds" : "fails"}, while the same episode ` +
        `${e.outcome === "success" ? "failed" : "succeeded"} in the evaluation.`).join(" ");
    }
    update();
  }

  // ------------------------------------------------------------ real robot: the trials of the paper's figure, as video
  // One row per belt speed; the two clips of a row start together, a clip that ends first holds its last frame, and
  // both restart when both have ended. The tag appears from the video time of the gripper's first close.
  const inputTag = $("#real-input-tag").content.firstElementChild;
  for (const pair of D.realFigure) {
    const row = el("div", "real-row");
    row.append(el("h4", null, speedLabel(pair.speed)));
    const grid = el("div", "real-pair");
    const vids = [];
    for (const p of pair.panels) {
      const fig = el("figure", "cell");
      const frame = el("div", "frame");
      const v = el("video");
      Object.assign(v, { muted: true, playsInline: true, preload: "metadata" });
      v.setAttribute("muted", "");
      v.setAttribute("playsinline", "");
      v.setAttribute("aria-label", `${plain(modelBy[p.model].label)}, past frames, ${pair.speed} m/s: ${p.note}`);
      v.dataset.src = `static/vid/real/${p.file}`;
      const tag = el("span", "close-tag", "gripper closes");
      tag.hidden = true;
      if (p.close !== null) {
        const show = () => { tag.hidden = v.currentTime < p.close; };
        ["timeupdate", "seeked", "play", "ended"].forEach((e) => v.addEventListener(e, show));
        const tick = () => { show(); v.requestVideoFrameCallback(tick); };
        if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(tick);
      }
      frame.append(v, inputTag.cloneNode(true), tag);
      fig.append(frame, el("span", "who", `${modelBy[p.model].label}, + past frames`), el("span", "fig-note", p.note));
      grid.append(fig);
      vids.push(v);
    }
    row.append(grid);
    $("#real-figure").append(row);
    let ended = 0, started = false;
    const restart = () => { ended = 0; vids.forEach((x) => { x.currentTime = 0; x.play().catch(() => {}); }); };
    vids.forEach((v) => v.addEventListener("ended", () => { ended += 1; if (ended === vids.length) restart(); }));
    new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !started) {
        started = true;
        vids.forEach((x) => { if (!x.getAttribute("src")) x.src = x.dataset.src; });
        restart();
      } else if (!e.isIntersecting && started) {
        started = false;
        vids.forEach((x) => x.pause());
      }
    }, { threshold: 0.3 }).observe(row);
  }
})();
