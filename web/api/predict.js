import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { gunzipSync } from "node:zlib";

const engine = join(process.cwd(), "engine");
let loaded;

function loadModel() {
  loaded ??= (async () => {
    const metadata = JSON.parse(await readFile(join(engine, "metadata.json"), "utf8"));
    const runtime = await import(pathToFileURL(join(engine, "ort.wasm.min.mjs")).href);
    runtime.env.wasm.numThreads = 1;
    runtime.env.wasm.wasmPaths = pathToFileURL(engine + "/").href;
    runtime.env.wasm.wasmBinary = await readFile(join(engine, "ort-wasm-simd-threaded.wasm"));
    const model = Buffer.concat(await Promise.all(metadata.model.chunks.map(name => readFile(join(engine, name)))));
    if (model.length !== metadata.model.size || createHash("sha256").update(model).digest("hex") !== metadata.model.sha256) {
      throw new Error("Model file integrity check failed.");
    }
    const session = await runtime.InferenceSession.create(model, { executionProviders: ["wasm"] });
    return { metadata, session, Tensor: runtime.Tensor };
  })().catch(error => {
    loaded = null;
    throw error;
  });
  return loaded;
}

export default async function predict(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Send a photo using POST." });
  }
  const encoded = req.body?.pixels;
  if (typeof encoded !== "string" || !encoded.length || encoded.length > 240_000) {
    return res.status(400).json({ error: "Invalid image payload." });
  }
  try {
    const { metadata, session, Tensor } = await loadModel();
    const { size, mean, std } = metadata;
    const count = size * size;
    let rgb;
    try {
      rgb = gunzipSync(Buffer.from(encoded, "base64"), { maxOutputLength: count * 3 });
    } catch {
      return res.status(400).json({ error: "Invalid compressed image." });
    }
    if (rgb.length !== count * 3) return res.status(400).json({ error: "Invalid image dimensions." });
    const input = new Float32Array(rgb.length);
    for (let channel = 0; channel < 3; channel++) {
      for (let pixel = 0; pixel < count; pixel++) {
        input[channel * count + pixel] = (rgb[pixel * 3 + channel] / 255 - mean[channel]) / std[channel];
      }
    }
    const output = await session.run({ image: new Tensor("float32", input, [1, 3, size, size]) });
    return res.status(200).json({ probabilities: Array.from(output.probabilities.data) });
  } catch (error) {
    console.error("Inference failed", error.message);
    return res.status(503).json({ error: "The model could not run. Try again shortly." });
  }
}
