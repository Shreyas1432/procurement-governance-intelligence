const { chromium } = require("playwright");

async function main() {
  const [htmlPath, pngPath] = process.argv.slice(2);
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 720 }, deviceScaleFactor: 2 });
  await page.goto(`file://${htmlPath}`, { waitUntil: "networkidle" });
  await page.waitForSelector(".js-plotly-plot", { timeout: 30000 });
  await page.screenshot({ path: pngPath, fullPage: true });
  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
