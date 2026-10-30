// One README screenshot (or the demo video's frames) of the instance run.ps1 starts.
//   node shoot.mjs <url> <home|transcribe|history> <out.png> [arg]
//   node shoot.mjs <url> video <manifest.json> [arg]
//   transcribe, video: arg = where the slider's right end goes, in seconds (100)
//   history:           arg = the entry to open ("Client onboarding call")
// video: the transcribe flow with a drawn mouse pointer, saved as numbered PNG
// frames next to the manifest, which lists each frame's time and when the waits
// start and end (demo.py gif speeds those up and makes the GIF).
// Needs puppeteer-core (PUPPETEER_FROM, a folder whose node_modules has it) and
// Chrome (CHROME). Light theme, scale 1; screenshots are 1320 px wide.
import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(process.env.PUPPETEER_FROM || "C:/Work/tools/");
const puppeteer = require("puppeteer-core");

const CHROME =
  process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
// The demo folder is a long temp path; the field shows the kind a reader has.
const SHOWN_FOLDER = String.raw`C:\Users\you\Videos`;
const WIDTH = 1320;
// The video: no sidebar, a window the GIF can shrink to 960 px and stay legible.
const VIDEO = { width: 1200, height: 760 };
const [url, mode, out, extra] = process.argv.slice(2);
if (!url || !mode || !out) {
  console.error("usage: node shoot.mjs <url> <home|transcribe|history|video> <out> [arg]");
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--hide-scrollbars", "--force-prefers-color-scheme=light"],
});
const page = await browser.newPage();
await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);
await page.setViewport(
  mode === "video"
    ? { ...VIDEO, deviceScaleFactor: 1 }
    : { width: WIDTH, height: 900, deviceScaleFactor: 1 },
);
await page.goto(url, { waitUntil: "networkidle2", timeout: 60000 });

async function waitText(text, timeout = 60000) {
  await page.waitForFunction(
    (t) => document.body.textContent.includes(t),
    { timeout, polling: 500 },
    text,
  );
}

async function idle() {
  // Streamlit shows its status widget while a script run is going.
  await sleep(600);
  await page.waitForFunction(
    () => !document.querySelector('[data-testid="stStatusWidget"]'),
    { timeout: 600000, polling: 300 },
  );
  await sleep(400);
}

async function clickText(selector, text) {
  // Waits for the element: a rerun may still be drawing it.
  await page.waitForFunction(
    (sel, t) => [...document.querySelectorAll(sel)].some((e) => e.textContent.includes(t)),
    { timeout: 60000, polling: 300 },
    selector,
    text,
  );
  await page.evaluate(
    (sel, t) => {
      const el = [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(t));
      el.scrollIntoView({ block: "center" });
      el.click();
    },
    selector,
    text,
  );
}

async function maskFolder() {
  // Draws the short path over the field's text, again after every rerun (React
  // writes the real value back whenever it redraws the field). Nothing is typed,
  // so the app keeps the real folder.
  await page.evaluate((shown) => {
    const apply = () => {
      const box = [...document.querySelectorAll('[data-testid="stTextInput"]')].find((b) =>
        b.textContent.includes("Recordings folder"),
      );
      const input = box?.querySelector("input");
      if (!input) return;
      input.style.webkitTextFillColor = "transparent";
      const holder = input.parentElement;
      if (getComputedStyle(holder).position === "static") holder.style.position = "relative";
      let label = holder.querySelector(":scope > .demo-folder");
      if (!label) {
        label = document.createElement("span");
        label.className = "demo-folder";
        label.textContent = shown;
        holder.appendChild(label);
      }
      const cs = getComputedStyle(input);
      Object.assign(label.style, {
        position: "absolute",
        left: `${input.offsetLeft + parseFloat(cs.paddingLeft)}px`,
        top: `${input.offsetTop}px`,
        height: `${input.offsetHeight}px`,
        display: "flex",
        alignItems: "center",
        pointerEvents: "none",
        whiteSpace: "nowrap",
        font: cs.font,
        color: cs.color,
      });
    };
    apply();
    setInterval(apply, 30);
  }, SHOWN_FOLDER);
  await sleep(100);
}

async function fullHeight() {
  // Streamlit scrolls inside its own container, so a full-page screenshot
  // would stop at the window: make the window as tall as the page instead.
  const height = await page.evaluate(() => {
    const main = document.querySelector('[data-testid="stMain"]');
    const view = document.querySelector('[data-testid="stAppViewContainer"]');
    return Math.max(main?.scrollHeight ?? 0, view?.scrollHeight ?? 0, document.body.scrollHeight);
  });
  await page.setViewport({ width: WIDTH, height: Math.ceil(height) + 10, deviceScaleFactor: 1 });
  await sleep(1200);
}

async function finish() {
  await page.evaluate(() => {
    const main = document.querySelector('[data-testid="stMain"]');
    if (main) main.scrollTop = 0;
    window.scrollTo(0, 0);
  });
  await sleep(500);
  await page.screenshot({ path: out });
  console.log("saved", out);
}

async function openRecordingList() {
  await waitText("Choose a recording");
  // Not the first select box: the sidebar's title model comes before it.
  const handle = await page.evaluateHandle(() =>
    [...document.querySelectorAll('[data-testid="stSelectbox"]')]
      .find((b) => b.textContent.includes("Choose a recording"))
      .querySelector('div[data-baseweb="select"]'),
  );
  await handle.click();
  await page.waitForSelector('[role="option"]');
}

async function moveSliderEnd(end) {
  const value = () =>
    page.evaluate(() => {
      const thumbs = document.querySelectorAll('[data-testid="stSlider"] [role="slider"]');
      return thumbs[thumbs.length - 1].getAttribute("aria-valuetext") || "";
    });
  const seconds = (text) => text.split(":").reduce((acc, part) => acc * 60 + Number(part), 0);
  await page.evaluate(() => {
    const thumbs = document.querySelectorAll('[data-testid="stSlider"] [role="slider"]');
    thumbs[thumbs.length - 1].focus();
  });
  // Page Down = 10 s, arrow = 1 s. A press can get lost while the page reruns,
  // so read the value back and go again until it is there.
  for (let round = 0; round < 6; round++) {
    const left = seconds(await value()) - end;
    if (left === 0) break;
    const [big, small] = left > 0 ? ["PageDown", "ArrowLeft"] : ["PageUp", "ArrowRight"];
    for (let i = 0; i < Math.floor(Math.abs(left) / 10); i++) {
      await page.keyboard.press(big);
      await sleep(40);
    }
    for (let i = 0; i < Math.abs(left) % 10; i++) {
      await page.keyboard.press(small);
      await sleep(40);
    }
    await sleep(800);
  }
  await idle();
  if (seconds(await value()) !== end) throw new Error(`slider end is ${await value()}`);
}

// --- The video. Headless Chrome shows no pointer, so one is drawn that follows
// the real mouse events; frames are taken as screenshots with their own times
// (Chrome's screencast stopped sending frames once a run started).
const mouse = { x: VIDEO.width * 0.62, y: VIDEO.height * 0.55 };

async function drawPointer() {
  await page.evaluate(() => {
    const style = document.createElement("style");
    style.textContent = `
      #demo-pointer { position: fixed; left: 0; top: 0; z-index: 2147483647;
        pointer-events: none; }
      .demo-click { position: fixed; z-index: 2147483646; pointer-events: none;
        width: 36px; height: 36px; margin: -18px 0 0 -18px; border-radius: 50%;
        background: rgba(255, 75, 75, 0.35); animation: demo-click 0.5s ease-out forwards; }
      @keyframes demo-click { from { transform: scale(0.3); opacity: 1; }
        to { transform: scale(1.4); opacity: 0; } }`;
    document.head.appendChild(style);
    const pointer = document.createElement("div");
    pointer.id = "demo-pointer";
    pointer.innerHTML =
      '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M5 3v16l4.2-4.1 2.9 6.6 ' +
      '2.6-1.1-2.9-6.5h5.9z" fill="#1f2a44" stroke="#fff" stroke-width="1.6" ' +
      'stroke-linejoin="round"/></svg>';
    document.body.appendChild(pointer);
    const place = (x, y) => (pointer.style.transform = `translate(${x - 5}px, ${y - 3}px)`);
    addEventListener("mousemove", (e) => place(e.clientX, e.clientY), true);
    addEventListener(
      "mousedown",
      (e) => {
        const ring = document.createElement("div");
        ring.className = "demo-click";
        ring.style.left = `${e.clientX}px`;
        ring.style.top = `${e.clientY}px`;
        document.body.appendChild(ring);
        setTimeout(() => ring.remove(), 600);
      },
      true,
    );
  });
}

function startFrames(dir) {
  mkdirSync(dir, { recursive: true });
  const grab = { every: 100, running: true, frames: [], started: Date.now() };
  grab.done = (async () => {
    while (grab.running) {
      const begun = Date.now();
      const file = `f${String(grab.frames.length).padStart(5, "0")}.png`;
      await page.screenshot({
        path: path.join(dir, file),
        captureBeyondViewport: false,
        optimizeForSpeed: true,
      });
      grab.frames.push({ file, t: (begun - grab.started) / 1000 });
      await sleep(Math.max(0, grab.every - (Date.now() - begun)));
    }
  })();
  return grab;
}

async function glide(to, ms = 650) {
  const from = { ...mouse };
  const steps = Math.max(8, Math.round(ms / 25));
  for (let i = 1; i <= steps; i++) {
    const t = 1 - (1 - i / steps) ** 3; // ease out
    mouse.x = from.x + (to.x - from.x) * t;
    mouse.y = from.y + (to.y - from.y) * t;
    await page.mouse.move(mouse.x, mouse.y);
    await sleep(ms / steps);
  }
}

async function centerOf(selector, text, block = "nearest") {
  await page.waitForFunction(
    (sel, t) => [...document.querySelectorAll(sel)].some((e) => e.textContent.includes(t)),
    { timeout: 60000, polling: 300 },
    selector,
    text,
  );
  await page.evaluate(
    (sel, t, b) => {
      const el = [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(t));
      el.scrollIntoView({ behavior: "smooth", block: b });
    },
    selector,
    text,
    block,
  );
  await sleep(700);
  return page.evaluate(
    (sel, t) => {
      const el = [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(t));
      const r = el.getBoundingClientRect();
      return { x: r.left + Math.min(r.width / 2, 120), y: r.top + r.height / 2 };
    },
    selector,
    text,
  );
}

async function clickOn(point, pause = 250) {
  await glide(point);
  await sleep(pause);
  await page.mouse.click(point.x, point.y);
}

async function scrollMain(top) {
  await page.evaluate(
    (y) => document.querySelector('[data-testid="stMain"]')?.scrollTo({ top: y, behavior: "smooth" }),
    top,
  );
  await sleep(900);
}

async function thumbs() {
  return page.evaluate(() =>
    [...document.querySelectorAll('[data-testid="stSlider"] [role="slider"]')].map((el) => {
      const r = el.getBoundingClientRect();
      const text = el.getAttribute("aria-valuetext") || "";
      const seconds = text.split(":").reduce((acc, part) => acc * 60 + Number(part), 0);
      return { x: r.left + r.width / 2, y: r.top + r.height / 2, seconds };
    }),
  );
}

async function recordVideo(end) {
  // Fold the sidebar away: the flow is in the main area.
  await page.evaluate(() =>
    document.querySelector('[data-testid="stSidebarCollapseButton"] button')?.click(),
  );
  await sleep(800);
  await maskFolder();
  await drawPointer();
  await page.mouse.move(mouse.x, mouse.y);
  const frames = startFrames(path.join(path.dirname(out), "frames"));
  // Stretches timed apart from the rest: [from, to, at most this long] for waits,
  // [from, to, null] for pauses that play at real speed (time to read).
  const squeezed = [];
  const now = () => (Date.now() - frames.started) / 1000;
  const squeeze = async (longest, work) => {
    const from = now();
    await work();
    squeezed.push([from, now(), longest]);
  };
  const hold = async (ms) => {
    const from = now();
    await sleep(ms);
    squeezed.push([from, now(), null]);
  };
  await sleep(600);

  // 1. The folder's recordings, each with its length; pick the meeting.
  await waitText("Choose a recording");
  const list = await page.evaluate(() => {
    const box = [...document.querySelectorAll('[data-testid="stSelectbox"]')]
      .find((b) => b.textContent.includes("Choose a recording"))
      .querySelector('div[data-baseweb="select"]');
    const r = box.getBoundingClientRect();
    return { x: r.left + r.width * 0.4, y: r.top + r.height / 2 };
  });
  await clickOn(list);
  await page.waitForSelector('[role="option"]');
  await hold(1300); // the list, each recording with its length
  await clickOn(await centerOf('[role="option"]', "Sprint planning"), 400);
  frames.every = 300;
  await squeeze(0.8, async () => {
    await waitText("Transcribe from", 180000);
    await idle();
  });
  frames.every = 100;

  // 2. The picture of the sound; drag the right end back to where the talk stops.
  await centerOf('[data-testid="stSlider"]', "Transcribe from", "start");
  await page.evaluate(() =>
    document.querySelector('[data-testid="stMain"]')?.scrollBy({ top: -60, behavior: "smooth" }),
  );
  await hold(1100); // the picture of the whole recording
  const [left, right] = await thumbs();
  const perSecond = (right.x - left.x) / (right.seconds - left.seconds);
  await glide(right);
  await sleep(300);
  await page.mouse.down();
  await glide({ x: left.x + end * perSecond, y: right.y }, 1700);
  await page.mouse.up();
  await squeeze(0.4, idle);
  for (let round = 0; round < 3; round++) {
    const at = (await thumbs())[1].seconds;
    if (at === end) break;
    const key = at > end ? "ArrowLeft" : "ArrowRight";
    for (let i = 0; i < Math.abs(at - end); i++) await page.keyboard.press(key);
    await squeeze(0.4, idle);
  }
  await hold(2400); // the caption: what is sent, what is left out

  // 3. Local engine with timestamps, then Start; follow the progress under it.
  await clickOn(await centerOf('[data-testid="stRadio"] label', "Local (offline)", "center"));
  await squeeze(0.4, idle);
  await clickOn(await centerOf('[data-testid="stCheckbox"] label', "Include timestamps"));
  await squeeze(0.4, idle);
  const start = await centerOf("button", "Start Transcription", "center");
  await clickOn(start, 400);
  await sleep(1000);
  await centerOf("button", "Start Transcription", "center");
  await hold(600); // the progress under Start
  frames.every = 400;
  await squeeze(2.5, async () => {
    await waitText("Download Transcript", 600000);
    await idle();
  });
  frames.every = 100;

  // 4. The transcript: its first line says which part was transcribed.
  await scrollMain(130);
  const note = await page.evaluate(() => {
    const area = [...document.querySelectorAll("textarea")].find((t) =>
      t.value.includes("Transcribed"),
    );
    const r = area.getBoundingClientRect();
    return { x: r.left + r.width * 0.8, y: r.top + 28 }; // past the line's end
  });
  await glide(note, 900);
  await hold(3200); // the first line: which part was transcribed
  const finished = now();
  frames.running = false;
  await frames.done;
  // speed: everything not squeezed plays this much faster than it was recorded.
  const manifest = { size: VIDEO, end: finished, speed: 1.6, squeezed, frames: frames.frames };
  writeFileSync(out, JSON.stringify(manifest, null, 1));
  console.log("saved", out, frames.frames.length, "frames,", finished.toFixed(1), "s");
}

await waitText("Video & Audio Transcription");
await idle();

if (mode === "home") {
  await openRecordingList();
  await sleep(500);
  await maskFolder();
  await finish();
} else if (mode === "transcribe") {
  await openRecordingList();
  await clickText('[role="option"]', "Sprint planning");
  await idle();
  await waitText("Transcribe from", 180000);
  await idle();
  await moveSliderEnd(Number(extra || 100));
  await clickText('[data-testid="stRadio"] label', "Local (offline)");
  await idle();
  await clickText('[data-testid="stCheckbox"] label', "Include timestamps");
  await idle();
  await clickText("button", "Start Transcription");
  await sleep(2000);
  await waitText("Download Transcript", 600000);
  await idle();
  await fullHeight();
  await maskFolder();
  await finish();
} else if (mode === "history") {
  await clickText('button[role="tab"]', "History");
  await idle();
  await clickText('[data-testid="stExpander"] summary', extra || "Client onboarding call");
  await idle();
  await fullHeight();
  await finish();
} else if (mode === "video") {
  await recordVideo(Number(extra || 100));
} else {
  throw new Error(`unknown mode ${mode}`);
}
await browser.close();
