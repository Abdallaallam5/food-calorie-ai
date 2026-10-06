import { loadEngine, analyze, calculate, foodsList, visualize } from "./engine.js";

const $ = (id) => document.getElementById(id);
const SOURCE = {
  user: "حسب الوزن اللي كتبته",
  size: "حسب حجم الطبق اللي اخترته",
  image: "متقدّرة من الصورة",
  bowl: "الحصة المعتادة، لأن عمق الشوربة مش باين في الصورة",
  default: "الحصة المعتادة، لأن مفيش طبق واضح في الصورة",
};
let file = null, size = "auto", last = null, ready = false;

// ---- تحميل الموديل أول ما الصفحة تفتح ----
loadEngine((p) => { $("loadingBar").style.width = Math.round(p * 100) + "%"; })
  .then(() => {
    ready = true;
    $("loading").hidden = true;
    foodsList().forEach((f) => $("fix").add(new Option(`${f.name_ar} — ${f.food.replaceAll("_", " ")}`, f.food)));
    updateButton();
  })
  .catch((e) => {
    console.error(e);
    $("loadingText").textContent = "⚠️ مش قادر يحمّل الموديل. اتأكد من النت واعمل refresh للصفحة.";
  });

function updateButton() { $("go").disabled = !(file && ready); }

// ---- اختيار الصورة ----
const drop = $("drop"), input = $("file");
drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
input.addEventListener("change", () => input.files[0] && setFile(input.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));

function setFile(f) {
  if (!f.type.startsWith("image/")) return showError("الملف ده مش صورة");
  file = f;
  $("preview").src = URL.createObjectURL(f);
  $("preview").hidden = false; $("change").hidden = false; $("hint").hidden = true;
  $("result").hidden = true; showError(); updateButton();
}

// ---- الحجم والوزن ----
$("sizes").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  size = b.dataset.size;
  $("sizes").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  $("grams").value = "";
});
$("grams").addEventListener("input", () => {
  if ($("grams").value) $("sizes").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", false));
});

function options() {
  const grams = parseFloat($("grams").value);
  if (grams > 0 && grams <= 5000) return { grams, size: null };
  return { grams: null, size: size === "auto" ? null : size };
}

// ---- التحليل ----
$("go").addEventListener("click", async () => {
  if (!file || !ready) return;
  const btn = $("go");
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>بيحلل الصورة…';
  showError();
  await new Promise((r) => setTimeout(r, 30)); // نسيب المتصفح يرسم الـ spinner
  try {
    const { result, portion } = await analyze(file, options());
    last = {
      area: portion ? portion.food_area_cm2 : null,
      coverage: portion ? portion.coverage : null,
      visualization: portion ? visualize(portion) : null,
    };
    render(result);
  } catch (e) {
    console.error(e);
    showError("حصلت مشكلة في تحليل الصورة، جرّب صورة تانية");
  } finally {
    btn.textContent = "احسب السعرات"; updateButton();
  }
});

function render(r) {
  $("foodAr").textContent = r.name_ar;
  $("foodEn").textContent = r.food.replaceAll("_", " ");
  if (r.confidence == null) {
    $("conf").innerHTML = "✔️ انت اللي اخترت الصنف";
  } else {
    const p = Math.round(r.confidence * 100);
    $("conf").innerHTML = `ثقة الموديل: <b>${p}%</b><div class="bar"><i style="width:${p}%"></i></div>`;
  }
  animate($("kcal"), r.calories);
  $("range").textContent = `غالبًا بين ${r.calories_range[0]} و ${r.calories_range[1]} سعر`;
  $("portion").textContent = `~${r.portion_g} جم`;
  $("per100").textContent = `${Math.round(r.kcal_per_100g)} سعر`;
  $("source").textContent = "📏 الكمية " + SOURCE[r.portion_source] +
    (r.portion_source === "image" && r.food_area_cm2 ? ` (مساحة الأكل ~${r.food_area_cm2} سم²)` : "");
  $("warn").hidden = !r.uncertain;

  const alts = r.alternatives || [];
  $("altsBox").hidden = !alts.length;
  $("alts").innerHTML = "";
  alts.forEach((a) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = `${a.name_ar} (${Math.round(a.confidence * 100)}%)`;
    b.onclick = () => pick(a.food);
    $("alts").appendChild(b);
  });

  if (last && last.visualization) {
    $("shot").src = last.visualization;
    $("shotCap").innerHTML = `<span class="dot" style="background:#00c800"></span>الطبق
      <span class="dot" style="background:#ff8c00"></span>الأكل · الأكل واخد ${Math.round(last.coverage * 100)}% من الطبق`;
  } else {
    $("shot").src = $("preview").src;
    $("shotCap").textContent = "";
  }
  $("fix").value = "";
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

// المستخدم اختار الصنف الصح بنفسه
function pick(food) {
  if (!food || !last) return;
  const opt = options();
  const r = calculate([{ food, confidence: 1 }], { ...opt, area: !opt.grams && !opt.size ? last.area : null });
  r.confidence = null;
  render(r);
}
$("fix").addEventListener("change", (e) => pick(e.target.value));

function animate(el, to) {
  const t0 = performance.now(), d = 600;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / d);
    el.textContent = Math.round(to * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function showError(msg) { $("error").hidden = !msg; $("error").textContent = msg || ""; }
