/* MyTorcha docs: highlighter, copy buttons, menu, and three small widgets.
   Everything here is progressive enhancement: the pages read fine without JS. */
document.documentElement.classList.add("js");

(function () {
  "use strict";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ------------------------------------------------------------------ *
   * tiny Python highlighter
   * ------------------------------------------------------------------ */
  const KW = new Set(("False None True and as assert break class continue def del elif else except " +
    "finally for from global if import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
  const BUILTIN = new Set("print len range zip enumerate sum max min abs round int float list dict set tuple isinstance super type sorted".split(" "));
  const TOKEN = /(#[^\n]*)|((?:[rRbBfF]{1,2})?(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(\b\d+(?:\.\d*)?(?:e[+-]?\d+)?\b|\.\d+\b)|([A-Za-z_]\w*)/g;
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function highlightPython(src) {
    let out = "", last = 0, m;
    TOKEN.lastIndex = 0;
    while ((m = TOKEN.exec(src))) {
      out += esc(src.slice(last, m.index));
      const t = m[0];
      if (m[1]) out += `<span class="tok-com">${esc(t)}</span>`;
      else if (m[2]) out += `<span class="tok-str">${esc(t)}</span>`;
      else if (m[3]) out += `<span class="tok-num">${esc(t)}</span>`;
      else if (KW.has(t)) out += `<span class="tok-kw">${t}</span>`;
      else if (BUILTIN.has(t) || src[TOKEN.lastIndex] === "(") out += `<span class="tok-fn">${t}</span>`;
      else out += t;
      last = TOKEN.lastIndex;
    }
    return out + esc(src.slice(last));
  }

  function highlightShell(src) {
    return src.split("\n").map((line) => {
      const i = line.indexOf("#");
      if (i < 0) return esc(line);
      return esc(line.slice(0, i)) + `<span class="tok-com">${esc(line.slice(i))}</span>`;
    }).join("\n");
  }

  function setupCode() {
    document.querySelectorAll("pre > code").forEach((code) => {
      const text = code.textContent;
      if (code.classList.contains("lang-python")) code.innerHTML = highlightPython(text);
      else if (code.classList.contains("lang-bash")) code.innerHTML = highlightShell(text);
      const box = code.closest(".code");
      if (!box || box.classList.contains("output") || !navigator.clipboard) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy";
      btn.textContent = "Copy";
      btn.setAttribute("aria-label", "Copy code to clipboard");
      btn.addEventListener("click", () => {
        navigator.clipboard.writeText(text).then(() => {
          btn.textContent = "Copied";
          setTimeout(() => (btn.textContent = "Copy"), 1400);
        }, () => (btn.textContent = "Press Ctrl+C"));
      });
      box.appendChild(btn);
    });
  }

  function setupMenu() {
    const btn = document.querySelector(".menu-btn");
    const nav = document.getElementById("sidebar");
    if (!btn || !nav) return;
    btn.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      btn.setAttribute("aria-expanded", String(open));
      btn.textContent = open ? "Close" : "Chapters";
    });
  }

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /* ------------------------------------------------------------------ *
   * widget 1: a live computation graph with forward + backward
   *   z1 = w*x ; z = z1 + b ; a = tanh(z) ; d = a - t ; L = d^2
   * ------------------------------------------------------------------ */
  function graphWidget(root) {
    const ids = ["x", "w", "b", "t"];
    const inputs = Object.fromEntries(ids.map((k) => [k, root.querySelector(`[name=${k}]`)]));
    const outs = Object.fromEntries(ids.map((k) => [k, root.querySelector(`output[for=g-${k}]`)]));
    const svg = root.querySelector("svg");
    const readout = root.querySelector(".readout");
    const lrSel = root.querySelector("[name=lr]");
    const initial = Object.fromEntries(ids.map((k) => [k, inputs[k].value]));
    let steps = 0;

    const N = {
      x: { x: 20, y: 18, label: "x", leaf: true },
      w: { x: 20, y: 108, label: "w", leaf: true },
      b: { x: 160, y: 198, label: "b", leaf: true },
      z1: { x: 175, y: 63, label: "w·x" },
      z: { x: 320, y: 120, label: "z" },
      a: { x: 460, y: 120, label: "a = tanh(z)" },
      t: { x: 460, y: 220, label: "target t", leaf: true },
      d: { x: 600, y: 170, label: "d = a − t" },
      L: { x: 740, y: 170, label: "L = d²" },
    };
    const E = [["x", "z1", "*"], ["w", "z1", "*"], ["z1", "z", "+"], ["b", "z", "+"], ["z", "a", "tanh"],
      ["a", "d", "−"], ["t", "d", "−"], ["d", "L", "**2"]];
    const W = 124, H = 56;

    function compute() {
      const v = Object.fromEntries(ids.map((k) => [k, parseFloat(inputs[k].value)]));
      // forward
      v.z1 = v.w * v.x; v.z = v.z1 + v.b; v.a = Math.tanh(v.z); v.d = v.a - v.t; v.L = v.d * v.d;
      // backward: start with dL/dL = 1, then each node's local derivative times the incoming gradient
      const g = { L: 1 };
      g.d = 2 * v.d * g.L;
      g.a = 1 * g.d; g.t = -1 * g.d;
      g.z = (1 - v.a * v.a) * g.a;
      g.z1 = g.z; g.b = g.z;
      g.w = v.x * g.z1; g.x = v.w * g.z1;
      return { v, g };
    }

    const f = (n) => (Math.abs(n) < 5e-5 ? "0.0000" : n.toFixed(4));

    function draw() {
      const { v, g } = compute();
      let s = "";
      for (const [a, b, op] of E) {
        const A = N[a], B = N[b];
        const x1 = A.x + W, y1 = A.y + H / 2, x2 = B.x, y2 = B.y + H / 2;
        const mx = (x1 + x2) / 2;
        s += `<path d="M${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}"/>`;
      }
      for (const [k, n] of Object.entries(N)) {
        s += `<g class="gnode${n.leaf ? " leaf" : ""}" transform="translate(${n.x} ${n.y})">` +
          `<rect width="${W}" height="${H}" rx="7"/>` +
          `<text class="gname" x="8" y="17">${n.label}</text>` +
          `<text x="8" y="34">val ${f(v[k])}</text>` +
          `<text class="ggrad" x="8" y="49">grad ${f(g[k])}</text></g>`;
      }
      const ops = { z1: "*", z: "+", a: "tanh", d: "−", L: "²" };
      for (const [k, op] of Object.entries(ops)) {
        const n = N[k];
        s += `<g class="gop" transform="translate(${n.x - (op.length > 2 ? 20 : 12)} ${n.y + H / 2})"><circle r="${op.length > 2 ? 19 : 11}"/>` +
          `<text text-anchor="middle" dy="4"${op.length > 2 ? ' style="font-size:11px"' : ""}>${op}</text></g>`;
      }
      svg.innerHTML = s;
      ids.forEach((k) => (outs[k].value = parseFloat(inputs[k].value).toFixed(2)));
      readout.innerHTML = `loss <b>${v.L.toFixed(4)}</b> · dL/dw <b>${f(g.w)}</b> · dL/db <b>${f(g.b)}</b> · steps taken <b>${steps}</b>`;
    }

    function clamp(input, value) {
      input.value = Math.max(parseFloat(input.min), Math.min(parseFloat(input.max), value)).toFixed(2);
    }

    ids.forEach((k) => inputs[k].addEventListener("input", draw));
    root.querySelector("[data-act=step]").addEventListener("click", () => {
      const { g } = compute();
      const lr = parseFloat(lrSel.value);
      clamp(inputs.w, parseFloat(inputs.w.value) - lr * g.w); // w.data -= lr * w.grad
      clamp(inputs.b, parseFloat(inputs.b.value) - lr * g.b);
      steps++;
      draw();
    });
    root.querySelector("[data-act=reset]").addEventListener("click", () => {
      ids.forEach((k) => (inputs[k].value = initial[k]));
      steps = 0;
      draw();
    });
    draw();
  }

  /* ------------------------------------------------------------------ *
   * widget 2: optimizer race on a 2-D loss surface
   * ------------------------------------------------------------------ */
  const SURFACES = {
    valley: {
      name: "long valley",
      f: (x, y) => 0.05 * x * x + 2 * y * y,
      g: (x, y) => [0.1 * x, 4 * y],
      box: [-5, 5, -3.75, 3.75],
      start: [-4.6, 2.2],
      min: [0, 0],
      lrs: [0.12, 0.02, 0.03, 0.1], // SGD, momentum, RMSprop, Adam
    },
    rosenbrock: {
      name: "Rosenbrock banana",
      f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2,
      g: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)],
      box: [-2, 2, -1, 3],
      start: [-1.6, 2.6],
      min: [1, 1],
      lrs: [0.004, 0.001, 0.03, 0.05], // the banana is ~100x more curved: SGD must tiptoe
    },
  };
  // optimizer update rules; base learning rates live on each surface (they differ by family and by curvature)
  const OPTS = [
    { name: "SGD", color: "#2563eb", lr: 0.12, make: () => ({}),
      step: (p, g, s, lr) => p.map((pi, i) => pi - lr * g[i]) },
    { name: "SGD + momentum 0.9", color: "#16a34a", lr: 0.02, make: () => ({ v: [0, 0] }),
      step: (p, g, s, lr) => p.map((pi, i) => { s.v[i] = 0.9 * s.v[i] + g[i]; return pi - lr * s.v[i]; }) },
    { name: "RMSprop", color: "#9333ea", lr: 0.03, make: () => ({ sq: [0, 0] }),
      step: (p, g, s, lr) => p.map((pi, i) => { s.sq[i] = 0.99 * s.sq[i] + 0.01 * g[i] * g[i]; return pi - lr * g[i] / (Math.sqrt(s.sq[i]) + 1e-8); }) },
    { name: "Adam", color: "#ea580c", lr: 0.1, make: () => ({ m: [0, 0], v: [0, 0], t: 0 }),
      step: (p, g, s, lr) => {
        s.t++;
        return p.map((pi, i) => {
          s.m[i] = 0.9 * s.m[i] + 0.1 * g[i];
          s.v[i] = 0.999 * s.v[i] + 0.001 * g[i] * g[i];
          const mh = s.m[i] / (1 - 0.9 ** s.t), vh = s.v[i] / (1 - 0.999 ** s.t);
          return pi - lr * mh / (Math.sqrt(vh) + 1e-8);
        });
      } },
  ];
  const MAX_STEPS = 300;

  function raceWidget(root) {
    const canvas = root.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const surfSel = root.querySelector("[name=surface]");
    const lrIn = root.querySelector("[name=lrmul]");
    const lrOut = root.querySelector("output[for=r-lr]");
    const legend = root.querySelector(".legend");
    const playBtn = root.querySelector("[data-act=play]");
    const Wc = canvas.width, Hc = canvas.height;
    let surf, runners, bg, timer = null, step = 0;

    const toPx = ([x, y]) => [(x - surf.box[0]) / (surf.box[1] - surf.box[0]) * Wc, (surf.box[3] - y) / (surf.box[3] - surf.box[2]) * Hc];

    function hexToRgb(c) {
      const d = document.createElement("div");
      d.style.color = c;
      document.body.appendChild(d);
      const m = getComputedStyle(d).color.match(/\d+/g).map(Number);
      d.remove();
      return m;
    }

    function background() {
      const img = ctx.createImageData(Wc, Hc);
      const c0 = hexToRgb(css("--bg")), c1 = hexToRgb(css("--line")), c2 = hexToRgb(css("--accent-soft"));
      const levels = new Float32Array(Wc * Hc);
      for (let j = 0; j < Hc; j++) {
        for (let i = 0; i < Wc; i++) {
          const x = surf.box[0] + (i / Wc) * (surf.box[1] - surf.box[0]);
          const y = surf.box[3] - (j / Hc) * (surf.box[3] - surf.box[2]);
          levels[j * Wc + i] = Math.log(surf.f(x, y) + 1e-3) * 2.2;
        }
      }
      let lo = Infinity, hi = -Infinity;
      levels.forEach((l) => { lo = Math.min(lo, l); hi = Math.max(hi, l); });
      for (let k = 0; k < levels.length; k++) {
        const band = Math.floor(levels[k]);
        const right = k % Wc < Wc - 1 ? Math.floor(levels[k + 1]) : band;
        const down = k + Wc < levels.length ? Math.floor(levels[k + Wc]) : band;
        const tNorm = (levels[k] - lo) / (hi - lo);
        const edge = band !== right || band !== down;
        const base = c0.map((v, ch) => v + (c2[ch] - v) * (1 - tNorm) * 0.9);
        const col = edge ? c1 : base;
        img.data.set([col[0], col[1], col[2], 255], k * 4);
      }
      return img;
    }

    function reset() {
      stop();
      surf = SURFACES[surfSel.value];
      bg = background();
      step = 0;
      runners = OPTS.map((o, i) => ({ ...o, lr: surf.lrs[i], p: surf.start.slice(), s: o.make(), path: [surf.start.slice()], dead: false }));
      draw();
    }

    function advance(n) {
      const mul = parseFloat(lrIn.value);
      for (let k = 0; k < n && step < MAX_STEPS; k++, step++) {
        for (const r of runners) {
          if (r.dead) continue;
          const g = surf.g(r.p[0], r.p[1]);
          r.p = r.step(r.p, g, r.s, r.lr * mul);
          if (!isFinite(r.p[0]) || !isFinite(r.p[1]) || Math.abs(r.p[0]) > 1e3 || Math.abs(r.p[1]) > 1e3) r.dead = true;
          else r.path.push(r.p.slice());
        }
      }
    }

    function draw() {
      ctx.putImageData(bg, 0, 0);
      const [mx, my] = toPx(surf.min);
      ctx.fillStyle = css("--text");
      ctx.beginPath(); ctx.arc(mx, my, 5, 0, 7); ctx.fill();
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText("minimum", mx + 8, my - 8);
      for (const r of runners) {
        ctx.strokeStyle = r.color; ctx.lineWidth = 2.2; ctx.beginPath();
        r.path.forEach((pt, i) => { const [px, py] = toPx(pt); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
        ctx.stroke();
        const [px, py] = toPx(r.path[r.path.length - 1]);
        ctx.fillStyle = r.color; ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 7); ctx.fill();
      }
      ctx.fillStyle = css("--muted");
      ctx.fillText(`step ${step} / ${MAX_STEPS}`, 10, Hc - 12);
      const mul = parseFloat(lrIn.value);
      lrOut.value = `× ${mul.toFixed(2)}`;
      legend.innerHTML = runners.map((r) => {
        const loss = r.dead ? "diverged" : `loss ${surf.f(r.p[0], r.p[1]).toExponential(2)}`;
        return `<li><i style="background:${r.color}"></i>${r.name} <span class="val">lr ${(r.lr * mul).toPrecision(2)} · ${loss}</span></li>`;
      }).join("");
    }

    function stop() {
      if (timer) cancelAnimationFrame(timer);
      timer = null;
      playBtn.textContent = "Play";
    }

    function play() {
      if (timer) return stop();
      if (step >= MAX_STEPS) reset();
      if (reduceMotion) { advance(MAX_STEPS); draw(); return; } // no animation: jump to the end
      playBtn.textContent = "Pause";
      const tick = () => {
        advance(2); draw();
        timer = step < MAX_STEPS ? requestAnimationFrame(tick) : (stop(), null);
      };
      timer = requestAnimationFrame(tick);
    }

    playBtn.addEventListener("click", play);
    root.querySelector("[data-act=step]").addEventListener("click", () => { stop(); advance(10); draw(); });
    root.querySelector("[data-act=reset]").addEventListener("click", reset);
    surfSel.addEventListener("change", reset);
    lrIn.addEventListener("input", reset);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { bg = background(); draw(); });
    reset();
  }

  /* ------------------------------------------------------------------ *
   * widget 3: convolution visualiser
   * ------------------------------------------------------------------ */
  const IMAGE = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 9, 9, 9, 9, 9, 0],
    [0, 0, 0, 0, 9, 0, 0],
    [0, 0, 0, 9, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
  ];
  const KERNELS = {
    vertical: { w: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
    horizontal: { w: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] },
    blur: { w: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => v / 9)), label: "1/9" },
    sharpen: { w: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
    identity: { w: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
  };

  function convWidget(root) {
    const kSel = root.querySelector("[name=kernel]");
    const sSel = root.querySelector("[name=stride]");
    const pSel = root.querySelector("[name=padding]");
    const inGrid = root.querySelector(".g-in"), kGrid = root.querySelector(".g-k"), outGrid = root.querySelector(".g-out");
    const calc = root.querySelector(".conv-calc"), sizeEl = root.querySelector(".conv-size");
    let active = [0, 0], Ho = 0, Wo = 0, X = [], out = [], k = KERNELS.vertical, s = 1;

    const shade = (v, max) => {
      const t = Math.max(-1, Math.min(1, v / (max || 1)));
      return t >= 0 ? `rgba(234, 88, 12, ${0.08 + 0.6 * t})` : `rgba(37, 99, 235, ${0.08 + 0.6 * -t})`;
    };
    const fmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));

    function padded(p) {
      const n = IMAGE.length + 2 * p;
      return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
        const r = i - p, c = j - p;
        return r >= 0 && c >= 0 && r < IMAGE.length && c < IMAGE.length ? IMAGE[r][c] : null;
      }));
    }

    // rebuild every grid: only when a setting changes
    function build() {
      k = KERNELS[kSel.value]; s = +sSel.value;
      const p = +pSel.value;
      X = padded(p);
      const n = X.length;
      Ho = Wo = Math.floor((IMAGE.length + 2 * p - 3) / s) + 1;
      sizeEl.textContent = `output size = ⌊(7 + 2·${p} − 3) / ${s}⌋ + 1 = ${Ho}`;
      active = [Math.min(active[0], Ho - 1), Math.min(active[1], Wo - 1)];
      out = [];
      for (let i = 0; i < Ho; i++) {
        out.push([]);
        for (let j = 0; j < Wo; j++) {
          let acc = 0;
          for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) acc += (X[i * s + a][j * s + b] || 0) * k.w[a][b];
          out[i].push(acc);
        }
      }
      const outMax = Math.max(...out.flat().map(Math.abs));
      inGrid.style.gridTemplateColumns = `repeat(${n}, auto)`;
      inGrid.innerHTML = X.map((row, i) => row.map((v, j) =>
        `<div class="cell${v === null ? " pad" : ""}" data-i="${i}" data-j="${j}" style="background:${v ? shade(v, 9) : ""}">${v === null ? 0 : v}</div>`).join("")).join("");
      kGrid.style.gridTemplateColumns = "repeat(3, auto)";
      kGrid.innerHTML = k.w.map((row) => row.map((v) => `<div class="cell" style="background:${shade(v, 1)}">${k.label || fmt(v)}</div>`).join("")).join("");
      outGrid.style.gridTemplateColumns = `repeat(${Wo}, auto)`;
      outGrid.innerHTML = out.map((row, i) => row.map((v, j) =>
        `<button type="button" class="cell" data-i="${i}" data-j="${j}" ` +
        `aria-label="output row ${i} column ${j}: ${fmt(v)}" style="background:${shade(v, outMax)}">${fmt(v)}</button>`).join("")).join("");
      highlight();
    }

    // move the highlight only: no DOM rebuild, so hovering is stable
    function highlight() {
      const [ai, aj] = active;
      inGrid.querySelectorAll(".cell").forEach((c) => {
        const i = +c.dataset.i, j = +c.dataset.j;
        c.classList.toggle("hl", i >= ai * s && i < ai * s + 3 && j >= aj * s && j < aj * s + 3);
      });
      outGrid.querySelectorAll(".cell").forEach((c) => c.classList.toggle("active", +c.dataset.i === ai && +c.dataset.j === aj));
      const terms = [];
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        const x = X[ai * s + a][aj * s + b] || 0, w = k.w[a][b];
        if (x !== 0 && w !== 0) terms.push(`${fmt(x)}·${k.label || fmt(w)}`);
      }
      calc.textContent = `out[${ai}][${aj}] = Σ window × kernel = ${terms.length ? terms.join(" + ") : "0"} = ${fmt(out[ai][aj])}`;
    }

    function pick(e) {
      const c = e.target.closest("button.cell");
      if (!c) return;
      const next = [+c.dataset.i, +c.dataset.j];
      if (next[0] === active[0] && next[1] === active[1]) return;
      active = next;
      highlight();
    }
    outGrid.addEventListener("mouseover", pick);
    outGrid.addEventListener("focusin", pick);
    outGrid.addEventListener("click", pick);
    root.querySelector("[data-act=next]").addEventListener("click", () => {
      let [i, j] = active;
      j++;
      if (j >= Wo) { j = 0; i = (i + 1) % Ho; }
      active = [i, j];
      highlight();
    });
    [kSel, sSel, pSel].forEach((el) => el.addEventListener("change", build));
    build();
  }

  document.addEventListener("DOMContentLoaded", () => {
    setupCode();
    setupMenu();
    document.querySelectorAll("[data-widget]").forEach((el) => {
      const kind = el.dataset.widget;
      const noscript = el.querySelector(".noscript");
      if (noscript) noscript.remove();
      el.querySelectorAll("[hidden]").forEach((h) => h.removeAttribute("hidden"));
      if (kind === "graph") graphWidget(el);
      else if (kind === "race") raceWidget(el);
      else if (kind === "conv") convWidget(el);
    });
  });
})();
