// Render the existing vector logo to lossless home-screen icons.
const fs = require("node:fs/promises");
const path = require("node:path");
const {chromium} = require("playwright");

(async () => {
  const publicDir = path.resolve(__dirname, "../../web/public");
  const svg = await fs.readFile(path.join(publicDir, "weedwise-logo.svg"), "utf8");
  const output = path.join(publicDir, "icons");
  await fs.mkdir(output, {recursive: true});
  const browser = await chromium.launch({channel: "msedge", headless: true});
  try {
    const page = await browser.newPage({deviceScaleFactor: 1});
    for (const [filename, size, maskable] of [
      ["icon-192.png", 192, false],
      ["icon-512.png", 512, false],
      ["icon-maskable-512.png", 512, true],
      ["apple-touch-icon.png", 180, false]
    ]) {
      let artwork = svg.replace('width="64" height="64"', `width="${size}" height="${size}"`);
      if (maskable) {
        artwork = artwork.replace(/(<svg[^>]*>)/, '$1<g transform="translate(12.8 12.8) scale(.6)">')
          .replace("</svg>", "</g></svg>");
      }
      await page.setViewportSize({width: size, height: size});
      await page.setContent(`<html style="background:#D5E8C3"><body style="margin:0;line-height:0">${artwork}</body></html>`);
      await page.screenshot({path: path.join(output, filename)});
      console.log(`Rendered ${filename} (${size}×${size})`);
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
