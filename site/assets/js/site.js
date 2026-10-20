/* CASCADE project site: progressive enhancement only. Every section reads fine without this file. */
(function () {
  "use strict";

  var root = document.documentElement;
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- theme ---------- */
  var themeBtn = document.querySelector(".theme-toggle");
  function syncThemeLabel() {
    if (!themeBtn) return;
    var dark = root.getAttribute("data-theme") === "dark";
    themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }
  syncThemeLabel();
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("cascade-theme", next); } catch (e) {}
      syncThemeLabel();
    });
  }
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var onSchemeChange = function (e) {
      var saved = null;
      try { saved = localStorage.getItem("cascade-theme"); } catch (err) {}
      if (saved) return;
      root.setAttribute("data-theme", e.matches ? "dark" : "light");
      syncThemeLabel();
    };
    if (mq.addEventListener) mq.addEventListener("change", onSchemeChange);
  }

  /* ---------- mobile menu ---------- */
  var nav = document.querySelector(".site-nav");
  var menuBtn = document.querySelector(".menu-toggle");
  if (nav && menuBtn) {
    menuBtn.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    });
    nav.querySelectorAll(".nav-links a").forEach(function (a) {
      a.addEventListener("click", function () {
        nav.classList.remove("open");
        menuBtn.setAttribute("aria-expanded", "false");
      });
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && nav.classList.contains("open")) {
        nav.classList.remove("open");
        menuBtn.setAttribute("aria-expanded", "false");
        menuBtn.focus();
      }
    });
  }

  /* ---------- active nav link ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".nav-links a"));
  var sections = links.map(function (a) { return document.querySelector(a.getAttribute("href")); }).filter(Boolean);
  if ("IntersectionObserver" in window && sections.length) {
    var current = null;
    var navIo = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) current = en.target.id;
      });
      links.forEach(function (a) {
        if (a.getAttribute("href") === "#" + current) a.setAttribute("aria-current", "true");
        else a.removeAttribute("aria-current");
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach(function (s) { navIo.observe(s); });
  }

  /* ---------- hero trace ----------
     Every row is always visible (that is the static frame). Playback only moves a highlight
     cursor down the rows and leaves it on the verdict. */
  var trace = document.getElementById("trace");
  var replay = document.getElementById("trace-replay");
  var traceTimers = [];
  function playTrace() {
    if (!trace || reduceMotion) return;
    var rows = Array.prototype.slice.call(trace.querySelectorAll("li"));
    traceTimers.forEach(clearTimeout);
    traceTimers = [];
    rows.forEach(function (r, i) {
      traceTimers.push(setTimeout(function () {
        rows.forEach(function (o) { o.classList.remove("cursor"); });
        r.classList.add("cursor");
        if (i === rows.length - 1) traceTimers.push(setTimeout(function () { r.classList.remove("cursor"); }, 1400));
      }, 400 + i * 520));
    });
  }
  if (replay) {
    if (reduceMotion) replay.hidden = true;
    replay.addEventListener("click", playTrace);
  }
  playTrace();

  /* ---------- pipeline stepper ---------- */
  var stepper = document.getElementById("stepper");
  if (stepper) {
    var tabs = Array.prototype.slice.call(stepper.querySelectorAll(".step-tab"));
    var panels = tabs.map(function (t) { return document.getElementById(t.getAttribute("aria-controls")); });
    var progress = stepper.querySelector(".rail-progress");
    var pulse = stepper.querySelector(".rail-pulse");
    var count = document.getElementById("step-count");
    var playBtn = document.getElementById("step-play");
    panels.forEach(function (pn) { pn.setAttribute("role", "tabpanel"); });
    var idx = 0, timer = null, playing = false, userTouched = false;
    var n = tabs.length;

    function railPos(i) {
      /* centre of column i inside the padded rail, as a CSS calc() */
      return "calc(var(--rail-pad, 18px) + (100% - 2 * var(--rail-pad, 18px)) * " + ((2 * i + 1) / (2 * n)) + ")";
    }
    function select(i, focus) {
      idx = (i + n) % n;
      tabs.forEach(function (t, k) {
        var on = k === idx;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        t.classList.toggle("done", k < idx);
        panels[k].hidden = !on;
      });
      var span = "(100% - 2 * var(--rail-pad, 18px)) * " + (idx / n);
      if (progress) progress.style.width = "calc(" + span + ")";
      if (pulse) pulse.style.left = railPos(idx);
      if (count) count.textContent = "Step " + (idx + 1) + " of " + n;
      if (focus) tabs[idx].focus();
    }
    function stop() {
      playing = false;
      clearInterval(timer);
      if (playBtn) { playBtn.textContent = "Play"; playBtn.setAttribute("aria-pressed", "false"); }
    }
    function start() {
      if (reduceMotion) return;
      playing = true;
      if (playBtn) { playBtn.textContent = "Pause"; playBtn.setAttribute("aria-pressed", "true"); }
      clearInterval(timer);
      timer = setInterval(function () { select(idx + 1); }, 5200);
    }
    tabs.forEach(function (t, k) {
      t.addEventListener("click", function () { userTouched = true; stop(); select(k); });
      t.addEventListener("keydown", function (e) {
        var key = e.key, next = null;
        if (key === "ArrowRight" || key === "ArrowDown") next = idx + 1;
        else if (key === "ArrowLeft" || key === "ArrowUp") next = idx - 1;
        else if (key === "Home") next = 0;
        else if (key === "End") next = n - 1;
        if (next !== null) { e.preventDefault(); userTouched = true; stop(); select(next, true); }
      });
    });
    document.getElementById("step-prev").addEventListener("click", function () { userTouched = true; stop(); select(idx - 1); });
    document.getElementById("step-next").addEventListener("click", function () { userTouched = true; stop(); select(idx + 1); });
    if (playBtn) {
      if (reduceMotion) playBtn.hidden = true;
      playBtn.addEventListener("click", function () { userTouched = true; if (playing) stop(); else start(); });
    }
    stepper.addEventListener("focusin", function (e) { if (e.target.closest(".step-panel")) stop(); });
    select(0);
    /* autoplay once, while the stepper is on screen, until the reader touches it */
    if ("IntersectionObserver" in window && !reduceMotion) {
      var sio = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting && !userTouched && !playing) start();
          else if (!en.isIntersecting && playing && !userTouched) stop();
        });
      }, { threshold: 0.45 });
      sio.observe(stepper);
    }
  }

  /* ---------- generic tablist (get started) ---------- */
  document.querySelectorAll(".tabs").forEach(function (box) {
    var tabEls = Array.prototype.slice.call(box.querySelectorAll('[role="tab"]'));
    tabEls.forEach(function (t) { document.getElementById(t.getAttribute("aria-controls")).setAttribute("role", "tabpanel"); });
    function show(i, focus) {
      tabEls.forEach(function (t, k) {
        var on = k === i;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      });
      if (focus) tabEls[i].focus();
    }
    tabEls.forEach(function (t, k) {
      t.addEventListener("click", function () { show(k); });
      t.addEventListener("keydown", function (e) {
        var i = tabEls.indexOf(document.activeElement), next = null;
        if (e.key === "ArrowRight") next = (i + 1) % tabEls.length;
        else if (e.key === "ArrowLeft") next = (i - 1 + tabEls.length) % tabEls.length;
        else if (e.key === "Home") next = 0;
        else if (e.key === "End") next = tabEls.length - 1;
        if (next !== null) { e.preventDefault(); show(next, true); }
      });
    });
    show(0);
  });

  /* ---------- filters ---------- */
  function wireFilter(groupSel, attr, itemsSel, itemAttr, onChange) {
    var btns = Array.prototype.slice.call(document.querySelectorAll(groupSel + " [" + attr + "]"));
    var items = Array.prototype.slice.call(document.querySelectorAll(itemsSel));
    btns.forEach(function (b) {
      b.addEventListener("click", function () {
        var v = b.getAttribute(attr);
        btns.forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        var shown = 0;
        items.forEach(function (it) {
          var on = v === "all" || it.getAttribute(itemAttr) === v;
          it.hidden = !on;
          if (on) shown++;
        });
        if (onChange) onChange(shown);
      });
    });
  }
  var catCount = document.getElementById("catalog-count");
  wireFilter("#catalog .catalog-head", "data-filter", "#catalog .tool-list li", "data-kind", function (k) {
    if (catCount) catCount.textContent = k + (k === 1 ? " tool" : " tools");
  });
  wireFilter(".stack-filters", "data-tier", ".stack-group", "data-tier");

  /* ---------- copy buttons ---------- */
  document.querySelectorAll(".code").forEach(function (block) {
    var pre = block.querySelector("pre");
    if (!pre || !navigator.clipboard) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy";
    btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy command");
    btn.addEventListener("click", function () {
      var clone = pre.cloneNode(true);
      clone.querySelectorAll(".c").forEach(function (c) { c.remove(); });
      var text = clone.textContent.split("\n").map(function (l) { return l.replace(/\s+$/, ""); }).join("\n").trim();
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = "Copied";
        btn.setAttribute("data-done", "");
        setTimeout(function () { btn.textContent = "Copy"; btn.removeAttribute("data-done"); }, 1600);
      });
    });
    block.appendChild(btn);
    block.classList.add("has-copy");
  });
  /* ---------- scroll reveal ----------
     Content is visible by default. Only elements that start below the viewport get the hidden
     .reveal-pending state, and only when motion is allowed and IntersectionObserver exists.
     Anything in or above the viewport (also after a resize, e.g. a full-page screenshot tool
     growing the window) is shown at once, without a transition. This runs last, after the
     stepper and tab widgets have hidden their inactive panels, so positions are final. */
  var pending = [];
  if ("IntersectionObserver" in window && !reduceMotion) {
    var vh = window.innerHeight || document.documentElement.clientHeight;
    document.querySelectorAll(".reveal").forEach(function (el) {
      if (el.getBoundingClientRect().top > vh) { el.classList.add("reveal-pending"); pending.push(el); }
    });
    var settle = function (el) {
      el.addEventListener("transitionend", function done() {
        el.classList.remove("reveal-pending", "is-visible");
        el.removeEventListener("transitionend", done);
      });
    };
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        var el = en.target;
        io.unobserve(el);
        if (el.classList.contains("reveal-pending")) { el.classList.add("is-visible"); settle(el); }
      });
    }, { rootMargin: "0px 0px -6% 0px", threshold: 0.04 });
    pending.forEach(function (el) { io.observe(el); });
    var showInView = function () {
      var h = window.innerHeight || document.documentElement.clientHeight;
      pending = pending.filter(function (el) {
        if (!el.classList.contains("reveal-pending")) return false;
        if (el.getBoundingClientRect().top < h) { el.classList.remove("reveal-pending"); io.unobserve(el); return false; }
        return true;
      });
    };
    window.addEventListener("resize", showInView);
    window.addEventListener("load", showInView);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(showInView);
    window.addEventListener("beforeprint", function () {
      pending.forEach(function (el) { el.classList.remove("reveal-pending", "is-visible"); });
      pending = [];
    });
  }

})();
