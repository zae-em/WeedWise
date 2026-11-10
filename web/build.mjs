import { createHash } from "node:crypto";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const path = relative => fileURLToPath(new URL(relative, import.meta.url));
await mkdir(path("./dist/engine"), { recursive: true });
await mkdir(path("./dist/public/release"), { recursive: true });
const sourceModel = path("./public/release/model.onnx");
const model = await readFile(sourceModel);
const metadata = JSON.parse(await readFile(path("./public/release/metadata.json"), "utf8"));
for (const filename of ["index.html", "main.js", "style.css"]) {
  await cp(path("./public/" + filename), path("./dist/public/" + filename));
}
for (const filename of ["metadata.json", ...metadata.examples.map(example => example.file)]) {
  await cp(path("./public/release/" + filename), path("./dist/public/release/" + filename));
}
for (const filename of ["api", "vercel.json", "public/.vercelignore"]) {
  await cp(path("./" + filename), path("./dist/" + filename.split("/").at(-1)), { recursive: true });
}
await writeFile(path("./dist/package.json"), '{"private":true,"type":"module","engines":{"node":"24.x"}}');
const chunks = [];
const partSize = 1024 * 1024;
for (let offset = 0; offset < model.length; offset += partSize) {
  const name = "model-" + chunks.length + ".bin";
  await writeFile(path("./dist/engine/" + name), model.subarray(offset, offset + partSize));
  chunks.push(name);
}
metadata.model = { chunks, size: model.length, sha256: createHash("sha256").update(model).digest("hex") };
await writeFile(path("./dist/engine/metadata.json"), JSON.stringify(metadata));
for (const filename of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  await cp(path("./node_modules/onnxruntime-web/dist/" + filename), path("./dist/engine/" + filename));
}
