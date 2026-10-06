// محرك السعرات في المتصفح: نفس منطق src/predict.py و src/portion.py و src/nutrition.py بالظبط.
// الموديل بيشتغل على جهاز المستخدم (onnxruntime-web)، والصورة مش بتتبعت لأي سيرفر.

let session = null, cv = null, DATA = null;

// Python round() بيقرّب للزوجي عند النص (2.5 ← 2)، فبنعمل زيه عشان النتايج تطابق
function pyRound(x, nd = 0) {
  const m = 10 ** nd, v = x * m, f = Math.floor(v), d = v - f;
  const r = Math.abs(d - 0.5) < 1e-9 ? (f % 2 === 0 ? f : f + 1) : Math.round(v);
  return r / m;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src; s.async = true; s.onload = resolve; s.onerror = () => reject(new Error("فشل تحميل " + src));
    document.head.appendChild(s);
  });
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("فشل تحميل " + url);
  const total = +res.headers.get("Content-Length") || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress && onProgress(got / total);
  }
  const out = new Uint8Array(got);
  let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function loadOpenCV() {
  await loadScript("vendor/opencv.js");
  let c = window.cv;
  if (c && typeof c.then === "function") c = await c;
  if (!c.Mat) await new Promise((r) => { c.onRuntimeInitialized = r; });
  return c;
}

/** بيحمّل الموديل و OpenCV وجدول السعرات. onProgress(نسبة من 0 لـ 1) */
export async function loadEngine(onProgress) {
  const parts = { model: 0, cv: 0 };
  const report = () => onProgress && onProgress(0.8 * parts.model + 0.2 * parts.cv);
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", location.href).href;
  ort.env.wasm.numThreads = 1;

  const [modelBytes, data, cvLib] = await Promise.all([
    fetchWithProgress("model/model.onnx", (p) => { parts.model = p; report(); }),
    fetch("model/foods.json").then((r) => r.json()),
    loadOpenCV().then((c) => { parts.cv = 1; report(); return c; }),
  ]);
  session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  DATA = data; cv = cvLib;
  return DATA;
}

export function foodsList() {
  return Object.entries(DATA.foods).map(([food, v]) => ({ food, name_ar: v.name_ar }))
    .sort((a, b) => a.name_ar.localeCompare(b.name_ar, "ar"));
}

// ---------------- فك الصورة ----------------
const MAX_DECODE = 4096;
function decode(bitmap) {
  // بكسلات الصورة الأصلية من غير أي تنعيم أو تحويل ألوان (زي PIL)
  const s = Math.min(1, MAX_DECODE / Math.max(bitmap.width, bitmap.height));
  const w = Math.round(bitmap.width * s), h = Math.round(bitmap.height * s);
  const c = new OffscreenCanvas(w, h), ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// ---------------- 1) التصنيف (src/predict.py) ----------------
// نفس خوارزمية PIL.Image.resize(BILINEAR) بالظبط (antialias + fixed point)،
// عشان الموديل يشوف الصورة زي ما اتدرب عليها في torchvision.
const PREC = 22, ONE = 2 ** PREC, HALF = 2 ** (PREC - 1);
function coeffs(inSize, outSize) {
  const scale = inSize / outSize, fscale = Math.max(scale, 1), support = fscale, ss = 1 / fscale;
  const bounds = [], weights = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    const k = []; let ww = 0;
    for (let x = 0; x < xmax - xmin; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss), w = t < 1 ? 1 - t : 0;
      k.push(w); ww += w;
    }
    bounds.push(xmin);
    weights.push(k.map((w) => { const v = ww ? w / ww : 0; return Math.trunc(v < 0 ? v * ONE - 0.5 : v * ONE + 0.5); }));
  }
  return { bounds, weights };
}
const clip8 = (v) => { const r = Math.floor(v / ONE); return r < 0 ? 0 : r > 255 ? 255 : r; };

function pilResize(src, sw, sh, dw, dh) {  // src: RGB Uint8Array
  let cur = src, w = sw;
  if (dw !== sw) {  // أفقي الأول زي PIL
    const { bounds, weights } = coeffs(sw, dw), out = new Uint8Array(dw * sh * 3);
    for (let y = 0; y < sh; y++) for (let x = 0; x < dw; x++) {
      const k = weights[x], b = bounds[x];
      for (let c = 0; c < 3; c++) {
        let s = HALF;
        for (let i = 0; i < k.length; i++) s += cur[(y * w + b + i) * 3 + c] * k[i];
        out[(y * dw + x) * 3 + c] = clip8(s);
      }
    }
    cur = out; w = dw;
  }
  if (dh !== sh) {
    const { bounds, weights } = coeffs(sh, dh), out = new Uint8Array(w * dh * 3);
    for (let y = 0; y < dh; y++) {
      const k = weights[y], b = bounds[y];
      for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
        let s = HALF;
        for (let i = 0; i < k.length; i++) s += cur[((b + i) * w + x) * 3 + c] * k[i];
        out[(y * w + x) * 3 + c] = clip8(s);
      }
    }
    cur = out;
  }
  return cur;
}

function preprocess(img) {
  // Resize(256) للضلع الأصغر ← CenterCrop(224) ← Normalize (زي torchvision بالظبط)
  const { img_size: S, resize: R, mean, std } = DATA.config;
  const w = img.width, h = img.height;
  const [nw, nh] = w <= h ? [R, Math.trunc(R * h / w)] : [Math.trunc(R * w / h), R];
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) { rgb[i * 3] = img.data[i * 4]; rgb[i * 3 + 1] = img.data[i * 4 + 1]; rgb[i * 3 + 2] = img.data[i * 4 + 2]; }
  const small = pilResize(rgb, w, h, nw, nh);
  const top = pyRound((nh - S) / 2), left = pyRound((nw - S) / 2);
  const out = new Float32Array(3 * S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const si = ((y + top) * nw + (x + left)) * 3, di = y * S + x;
    for (let c = 0; c < 3; c++) out[c * S * S + di] = (small[si + c] / 255 - mean[c]) / std[c];
  }
  return new ort.Tensor("float32", out, [1, 3, S, S]);
}

async function classify(img, topK = 5) {
  const out = await session.run({ image: preprocess(img) });
  const logits = out.logits.data;
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e, i) => ({ food: DATA.classes[i], confidence: pyRound(e / sum, 4) }))
    .sort((a, b) => b.confidence - a.confidence).slice(0, topK);
}

// ---------------- 2) تقدير الكمية (src/portion.py) ----------------
function toRGBMat(img) {
  // نفس الصورة الأصلية ← cv.resize لـ 512 (INTER_LINEAR زي بايثون)
  const P = DATA.config.portion;
  const rgba = cv.matFromImageData(img), rgb = new cv.Mat();
  cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB); rgba.delete();
  const s = P.max_side / Math.max(rgb.cols, rgb.rows);
  if (s < 1) {
    const small = new cv.Mat();
    cv.resize(rgb, small, new cv.Size(Math.floor(rgb.cols * s), Math.floor(rgb.rows * s)), 0, 0, cv.INTER_LINEAR);
    rgb.delete(); return small;
  }
  return rgb;
}

function findPlate(rgb) {
  const gray = new cv.Mat(), circles = new cv.Mat();
  cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
  cv.medianBlur(gray, gray, 5);
  const short = Math.min(gray.rows, gray.cols);
  cv.HoughCircles(gray, circles, cv.HOUGH_GRADIENT, 1.2, short, 100, 40,
    Math.floor(short * 0.25), Math.floor(short * 0.62));
  let best = null;
  for (let i = 0; i < circles.cols; i++) {
    const [x, y, r] = [0, 1, 2].map((k) => pyRound(circles.data32F[i * 3 + k]));
    if (!best || r > best[2]) best = [x, y, r];
  }
  gray.delete(); circles.delete();
  return best;
}

function median(values) {  // زي np.median على أرقام من 0 لـ 255
  const hist = new Uint32Array(256);
  for (const v of values) hist[v]++;
  const n = values.length, at = (k) => { let c = 0; for (let i = 0; i < 256; i++) { c += hist[i]; if (c > k) return i; } };
  return n % 2 ? at((n - 1) / 2) : (at(n / 2 - 1) + at(n / 2)) / 2;
}

function segmentFood(rgb, [x, y, r]) {
  const P = DATA.config.portion, W = rgb.cols, H = rgb.rows, N = W * H;
  const labMat = new cv.Mat();
  cv.cvtColor(rgb, labMat, cv.COLOR_RGB2Lab);
  const lab = labMat.data;
  const inside = new Uint8Array(N), rim = [[], [], []], rimIdx = [];
  for (let yy = 0; yy < H; yy++) for (let xx = 0; xx < W; xx++) {
    const i = yy * W + xx, d = Math.sqrt((xx - x) ** 2 + (yy - y) ** 2);
    if (d < r * 0.97) inside[i] = 1;
    if (d > r * 0.90 && d < r * 0.98) { rimIdx.push(i); for (let c = 0; c < 3; c++) rim[c].push(lab[i * 3 + c]); }
  }
  const plate = rim.map(median);
  const diff = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    diff[i] = Math.hypot(lab[i * 3] - plate[0], lab[i * 3 + 1] - plate[1], lab[i * 3 + 2] - plate[2]);
  }
  labMat.delete();

  // تأكيد إنه طبق فعلًا: أغلب حافته لازم تكون لون واحد تقريبًا
  let uniform = 0; for (const i of rimIdx) if (diff[i] < P.color_diff_threshold) uniform++;
  if (!rimIdx.length || uniform / rimIdx.length < P.min_rim_uniformity) return null;

  const mask = new cv.Mat(H, W, cv.CV_8UC1);
  for (let i = 0; i < N; i++) mask.data[i] = diff[i] > P.color_diff_threshold && inside[i] ? 1 : 0;
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(7, 7));
  cv.morphologyEx(mask, mask, cv.MORPH_OPEN, kernel);
  cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, kernel);
  kernel.delete();
  const out = Uint8Array.from(mask.data); mask.delete();
  return out;
}

function estimateFoodArea(img) {
  const P = DATA.config.portion;
  const rgb = toRGBMat(img);
  try {
    const circle = findPlate(rgb);
    if (!circle) return null;
    const mask = segmentFood(rgb, circle);
    if (!mask) return null;
    let count = 0; for (const v of mask) count += v;
    const platePx = Math.PI * circle[2] ** 2, coverage = count / platePx;
    if (coverage < P.min_coverage || coverage > P.max_coverage) return null;
    const cm2PerPx = (Math.PI * (P.plate_diameter_cm / 2) ** 2) / platePx;
    return {
      food_area_cm2: count * cm2PerPx, coverage, circle, mask,
      width: rgb.cols, height: rgb.rows, pixels: Uint8Array.from(rgb.data),
    };
  } finally { rgb.delete(); }
}

/** صورة توضيحية: الطبق بالأخضر والأكل متلوّن بالبرتقالي (زي visualize في بايثون) */
export function visualize(p) {
  const c = document.createElement("canvas");
  c.width = p.width; c.height = p.height;
  const ctx = c.getContext("2d"), id = ctx.createImageData(p.width, p.height);
  for (let i = 0; i < p.width * p.height; i++) {
    let [r, g, b] = [p.pixels[i * 3], p.pixels[i * 3 + 1], p.pixels[i * 3 + 2]];
    if (p.mask[i]) { r = (r + 255) / 2; g = (g + 140) / 2; b = b / 2; }
    id.data.set([r, g, b, 255], i * 4);
  }
  ctx.putImageData(id, 0, 0);
  ctx.strokeStyle = "rgb(0,200,0)"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(p.circle[0], p.circle[1], p.circle[2], 0, 2 * Math.PI); ctx.stroke();
  return c.toDataURL("image/jpeg", 0.85);
}

// ---------------- 3) حساب السعرات (src/nutrition.py) ----------------
function portionGrams(food, size, grams, area) {
  const C = DATA.config, info = DATA.foods[food];
  if (grams) return [grams, "user"];
  if (size in C.size_multiplier) return [info.serving_g * C.size_multiplier[size], "size"];
  const density = C.grams_per_cm2[info.shape];
  if (area && density) {
    const g = Math.min(Math.max(area * density, info.serving_g * C.min_factor), info.serving_g * C.max_factor);
    return [g, "image"];
  }
  if (area && density == null) return [info.serving_g, "bowl"];
  return [info.serving_g, "default"];
}

export function calculate(preds, { size = null, grams = null, area = null } = {}) {
  const C = DATA.config, top = preds[0];
  const cands = top.confidence >= C.confidence_threshold ? [top]
    : preds.slice(0, 3).filter((p) => p.food in DATA.foods);
  const total = cands.reduce((s, p) => s + p.confidence, 0) || 1;
  let kcal = 0;
  for (const p of cands) {
    const [g] = portionGrams(p.food, size, grams, area);
    kcal += (p.confidence / total) * DATA.foods[p.food].kcal_per_100g * g / 100;
  }
  const info = DATA.foods[top.food];
  const [portion, source] = portionGrams(top.food, size, grams, area);
  const m = source === "image" ? C.error_margin_image : C.error_margin;
  return {
    food: top.food, name_ar: info.name_ar, confidence: top.confidence,
    portion_g: pyRound(portion), portion_source: source,
    food_area_cm2: area ? pyRound(area) : null,
    kcal_per_100g: info.kcal_per_100g, calories: pyRound(kcal),
    calories_range: [pyRound(kcal * (1 - m)), pyRound(kcal * (1 + m))],
    uncertain: cands.length > 1,
    alternatives: preds.slice(1, 3)
      .filter((p) => p.food in DATA.foods && p.confidence >= C.min_alternative_conf)
      .map((p) => ({ food: p.food, name_ar: DATA.foods[p.food].name_ar, confidence: p.confidence })),
  };
}

/** التحليل الكامل: صورة ← صنف ← كمية ← سعرات */
export async function analyze(file, { size = null, grams = null } = {}) {
  const bitmap = await createImageBitmap(file, {
    imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none",
  });
  try {
    const img = decode(bitmap);
    const preds = await classify(img);
    const portion = estimateFoodArea(img);
    // زي بايثون: مساحة الصورة بتتستخدم بس في الوضع التلقائي
    const area = !grams && !size && portion ? portion.food_area_cm2 : null;
    const result = calculate(preds, { size, grams, area });
    return { result, portion, preds };
  } finally { bitmap.close(); }
}
