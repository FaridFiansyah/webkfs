import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join, parse } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectDir = join(scriptDir, "..");
const outDir = join(projectDir, "public", "ort-wasm");

let searchDir = projectDir;
let distDir = "";
const rootPath = parse(projectDir).root;

while (true) {
  const candidate = join(searchDir, "node_modules", "onnxruntime-web", "dist");
  if (existsSync(candidate)) {
    distDir = candidate;
    break;
  }

  if (searchDir === rootPath) {
    break;
  }

  searchDir = dirname(searchDir);
}

if (!existsSync(distDir)) {
  console.warn("onnxruntime-web belum terinstall, lewati copy wasm.");
  process.exit(0);
}

mkdirSync(outDir, { recursive: true });

for (const file of readdirSync(distDir)) {
  if (file.endsWith(".wasm")) {
    copyFileSync(join(distDir, file), join(outDir, file));
  }
}
