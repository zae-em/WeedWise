const el = id => document.getElementById(id);
const asset = name => new URL("./release/" + name, import.meta.url).href;
const pct = value => (value * 100).toFixed(1) + "%";
let metadata, selected, previewUrl;
let generation = 0;
const tabs = [el("upload-tab"), el("examples-tab")];
const gallery = el("example-gallery");

function clearSelection() {
  const request = ++generation;
  selected?.bitmap.close();
  selected = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  el("preview").removeAttribute("src");
  el("preview").hidden = el("selection").hidden = el("output").hidden = true;
  el("result-empty").hidden = false;
  el("run").disabled = true;
  gallery.querySelector('[aria-pressed="true"]')?.setAttribute("aria-pressed", "false");
  return request;
}

function showTab(tab) {
  if (tab.getAttribute("aria-selected") === "true") return;
  clearSelection();
  el("upload").value = "";
  for (const item of tabs) {
    const active = item === tab;
    item.setAttribute("aria-selected", String(active));
    item.tabIndex = active ? 0 : -1;
    el(item.getAttribute("aria-controls")).hidden = !active;
  }
  if (metadata) el("status").textContent = tab === tabs[1] ? "Select an example photo." : "Choose a photo to upload.";
}

for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => showTab(tab));
  tab.addEventListener("keydown", event => {
    const next = {ArrowLeft: 1 - index, ArrowRight: 1 - index, Home: 0, End: 1}[event.key];
    if (next === undefined) return;
    event.preventDefault();
    showTab(tabs[next]);
    tabs[next].focus();
  });
}

function followPhotoLink() {
  if (location.hash === "#examples") {
    showTab(tabs[1]);
    el("identify").scrollIntoView();
  } else if (location.hash === "#identify") {
    showTab(tabs[0]);
  }
}

window.addEventListener("hashchange", followPhotoLink);
for (const link of document.querySelectorAll('a[href="#examples"]')) {
  link.addEventListener("click", () => {
    showTab(tabs[1]);
    el("identify").scrollIntoView();
  });
}
followPhotoLink();

async function choose(blob, description, expected, request) {
  if (!blob.size || blob.size > 12 * 1024 * 1024) throw new Error("Choose an image under 12 MB.");
  if (!["image/jpeg", "image/png", "image/webp"].includes(blob.type)) throw new Error("Choose a JPG, PNG or WebP file.");
  const bitmap = await createImageBitmap(blob, {imageOrientation: "from-image"});
  if (request !== generation) {
    bitmap.close();
    return;
  }
  if (Math.min(bitmap.width, bitmap.height) < 32 || bitmap.width * bitmap.height > 40_000_000) {
    bitmap.close();
    throw new Error("Choose an image at least 32 pixels wide and tall and under 40 megapixels.");
  }
  selected = {bitmap, expected};
  previewUrl = URL.createObjectURL(blob);
  el("preview").src = previewUrl;
  el("selection").textContent = description;
  el("preview").hidden = el("selection").hidden = false;
  el("run").disabled = !metadata;
  el("status").textContent = "Photo ready.";
}

async function encodePhoto(bitmap) {
  const {size} = metadata;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d", {willReadFrequently: true});
  const ratio = size / Math.max(bitmap.width, bitmap.height);
  const width = Math.max(1, Math.round(bitmap.width * ratio));
  const height = Math.max(1, Math.round(bitmap.height * ratio));
  ctx.fillStyle = "#7f7f7f";
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, Math.floor((size - width) / 2), Math.floor((size - height) / 2), width, height);
  const {data} = ctx.getImageData(0, 0, size, size);
  const rgb = new Uint8Array(3 * size * size);
  for (let pixel = 0; pixel < size * size; pixel++) {
    rgb.set(data.subarray(pixel * 4, pixel * 4 + 3), pixel * 3);
  }
  const compressed = new Blob([rgb]).stream().pipeThrough(new CompressionStream("gzip"));
  const bytes = new Uint8Array(await new Response(compressed).arrayBuffer());
  return btoa(Array.from(bytes, byte => String.fromCharCode(byte)).join(""));
}

function renderResult(probabilities, photo) {
  if (probabilities.length !== metadata.classes.length || probabilities.some(p => !Number.isFinite(p))) {
    throw new Error("The model returned invalid scores.");
  }
  const ranked = metadata.classes.map((record, index) => ({...record, confidence: probabilities[index]}))
    .sort((a, b) => b.confidence - a.confidence);
  const best = ranked[0];
  const crop = ranked.filter(r => r.role === "crop").reduce((sum, r) => sum + r.confidence, 0);
  const review = best.confidence < metadata.threshold;
  el("flag").textContent = review ? "Needs review" : "Suggested match";
  el("flag").dataset.review = String(review);
  el("prediction").textContent = best.name;
  el("scientific").textContent = best.scientific;
  el("roles").textContent = "Crop " + pct(crop) + " · Weed " + pct(1 - crop);
  const rank = ranked.findIndex(r => r.label === photo.expected) + 1;
  el("comparison").textContent = !rank ? "" : rank <= 5 ? "Dataset label ranks " + rank + " of 5." : "Dataset label is outside the five closest matches.";
  el("ranking").replaceChildren(...ranked.slice(0, 5).map(record => {
    const item = document.createElement("li");
    const line = document.createElement("span");
    const name = document.createElement("b");
    const confidence = document.createElement("small");
    const bar = document.createElement("progress");
    name.textContent = record.name;
    confidence.textContent = pct(record.confidence);
    line.append(name, confidence);
    bar.max = 1;
    bar.value = record.confidence;
    bar.setAttribute("aria-label", record.name + " confidence");
    item.append(line, bar);
    return item;
  }));
  el("result-empty").hidden = true;
  el("output").hidden = false;
}

function listSpecies() {
  const query = el("search").value.trim().toLowerCase();
  const role = el("role").value;
  const matches = metadata.classes.filter(r => (!role || r.role === role)
    && [r.name, r.label, r.scientific, r.family].some(field => field.toLowerCase().includes(query)))
    .sort((a, b) => a.name.localeCompare(b.name));
  el("species-count").textContent = matches.length + " of " + metadata.classes.length + " supported species";
  el("species-list").replaceChildren(...matches.map(record => {
    const item = document.createElement("li");
    const name = document.createElement("b");
    const scientific = document.createElement("i");
    const details = document.createElement("small");
    name.textContent = record.name;
    scientific.textContent = record.scientific;
    details.textContent = record.role + " · " + record.family;
    item.append(name, scientific, details);
    return item;
  }));
}

el("upload").addEventListener("change", async event => {
  const [file] = event.target.files;
  if (!file) return;
  const request = clearSelection();
  try {
    await choose(file, file.name, null, request);
  } catch (error) {
    if (request === generation) el("status").textContent = error.message;
  }
});
function buildGallery() {
  const classes = new Map(metadata.classes.map(record => [record.label, record]));
  gallery.replaceChildren(...metadata.examples.map(sample => {
    const record = classes.get(sample.label);
    const button = document.createElement("button");
    const image = document.createElement("img");
    const name = document.createElement("span");
    button.type = "button";
    button.className = "example-photo";
    button.dataset.example = sample.file;
    button.setAttribute("aria-pressed", "false");
    image.src = asset(sample.file);
    image.alt = "";
    image.width = image.height = 224;
    image.loading = "lazy";
    name.textContent = record.name;
    button.append(image, name);
    button.addEventListener("click", async () => {
      const request = clearSelection();
      button.setAttribute("aria-pressed", "true");
      el("status").textContent = "Loading example…";
      try {
        const response = await fetch(asset(sample.file));
        if (!response.ok) throw new Error("The example photo could not be loaded.");
        await choose(await response.blob(), "Selected example · " + record.name, sample.label, request);
      } catch (error) {
        if (request === generation) {
          button.setAttribute("aria-pressed", "false");
          el("status").textContent = error.message;
        }
      }
    });
    return button;
  }));
}
el("run").addEventListener("click", async () => {
  const photo = selected;
  el("run").disabled = true;
  el("status").textContent = "Running prediction…";
  try {
    const pixels = await encodePhoto(photo.bitmap);
    if (photo !== selected) return;
    const response = await fetch("./api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({pixels}),
      signal: AbortSignal.timeout(45_000),
    });
    const prediction = await response.json();
    if (!response.ok) throw new Error(prediction.error || "The prediction service is unavailable.");
    if (photo !== selected) return;
    renderResult(prediction.probabilities, photo);
    el("status").textContent = "Prediction complete.";
  } catch (error) {
    console.error(error);
    if (photo === selected) el("status").textContent = "The prediction could not be completed. Please try again.";
  } finally {
    if (photo === selected) el("run").disabled = false;
  }
});

try {
  const response = await fetch(asset("metadata.json"));
  if (!response.ok) throw new Error("Release metadata is missing.");
  metadata = await response.json();
  buildGallery();
  el("run").disabled = !selected;
  el("status").textContent = selected ? "Photo ready." :
    tabs[1].getAttribute("aria-selected") === "true" ? "Select an example photo." : "Choose a photo to upload.";
  el("search").addEventListener("input", listSpecies);
  el("role").addEventListener("change", listSpecies);
  listSpecies();
} catch (error) {
  console.error(error);
  el("status").textContent = "V2 is unavailable. Refresh to retry.";
}
