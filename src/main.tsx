import React, { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Download,
  FileImage,
  ImageUp,
  LoaderCircle,
  Play,
  RotateCcw,
  SlidersHorizontal
} from "lucide-react";
import * as ort from "onnxruntime-web/wasm";
import "./styles.css";

const MODEL_URL = "/bestkfslast.onnx";
const MODEL_NAME = "bestkfslast.onnx";
const MODEL_SIZE = 640;
const CLASS_NAMES = [
  "FB1",
  "FB10",
  "FB12",
  "FB13",
  "FB2",
  "FB3",
  "FB4",
  "FB5",
  "FB6",
  "FB7",
  "FB8",
  "FB9",
  "FR1",
  "FR10",
  "FR11",
  "FR12",
  "FR13",
  "FR14",
  "FR15",
  "FR2",
  "FR3",
  "FR4",
  "FR5",
  "FR6",
  "FR7",
  "FR8",
  "FR9",
  "Fb11",
  "Fb14",
  "Fb15",
  "RB1",
  "RB10",
  "RB11",
  "RB13",
  "RB14",
  "RB2",
  "RB3",
  "RB4",
  "RB5",
  "RB6",
  "RB7",
  "RB8",
  "RB9",
  "RR1",
  "RR10",
  "RR11",
  "RR12",
  "RR13",
  "RR14",
  "RR15",
  "RR2",
  "RR3",
  "RR4",
  "RR5",
  "RR6",
  "RR7",
  "RR8",
  "RR9",
  "Rb12",
  "Rb15"
];
const CLASS_COLORS = CLASS_NAMES.map((_, index) => `hsl(${(index * 47) % 360} 72% 38%)`);

type Box = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
};

type PredictionSummary = {
  score: number;
  classId: number;
};

type LoadedImage = {
  fileName: string;
  bitmap: ImageBitmap;
  width: number;
  height: number;
};

type LetterboxInfo = {
  ratio: number;
  padX: number;
  padY: number;
};

let sessionPromise: Promise<ort.InferenceSession> | null = null;

function getAssetUrl(path: string) {
  return new URL(path, window.location.origin).href;
}

function getSession() {
  if (!sessionPromise) {
    ort.env.wasm.wasmPaths = {
      mjs: getAssetUrl("/ort-wasm/ort-wasm-simd-threaded.mjs"),
      wasm: getAssetUrl("/ort-wasm/ort-wasm-simd-threaded.wasm")
    };
    ort.env.wasm.numThreads = 1;
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    });
  }

  return sessionPromise as Promise<ort.InferenceSession>;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

function getShortErrorMessage(error: unknown) {
  const message = getErrorMessage(error).replace(/\s+/g, " ").trim();

  return message.length > 96 ? `${message.slice(0, 96)}...` : message || "error tidak diketahui";
}

function preprocess(bitmap: ImageBitmap) {
  const canvas = document.createElement("canvas");
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("Canvas tidak tersedia.");
  }

  const ratio = Math.min(MODEL_SIZE / bitmap.width, MODEL_SIZE / bitmap.height);
  const scaledWidth = Math.round(bitmap.width * ratio);
  const scaledHeight = Math.round(bitmap.height * ratio);
  const padX = Math.floor((MODEL_SIZE - scaledWidth) / 2);
  const padY = Math.floor((MODEL_SIZE - scaledHeight) / 2);

  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, padX, padY, scaledWidth, scaledHeight);

  const pixels = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const input = new Float32Array(1 * 3 * MODEL_SIZE * MODEL_SIZE);
  const planeSize = MODEL_SIZE * MODEL_SIZE;

  for (let i = 0; i < planeSize; i += 1) {
    const pixelIndex = i * 4;
    input[i] = pixels[pixelIndex] / 255;
    input[planeSize + i] = pixels[pixelIndex + 1] / 255;
    input[planeSize * 2 + i] = pixels[pixelIndex + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]),
    letterbox: { ratio, padX, padY }
  };
}

function intersectionOverUnion(a: Box, b: Box) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - intersection;

  return union <= 0 ? 0 : intersection / union;
}

function nonMaxSuppression(boxes: Box[], iouThreshold: number) {
  const selected: Box[] = [];

  for (const classId of CLASS_NAMES.keys()) {
    const candidates = boxes
      .filter((box) => box.classId === classId)
      .sort((a, b) => b.score - a.score);

    while (candidates.length > 0) {
      const best = candidates.shift();
      if (!best) {
        continue;
      }

      selected.push(best);

      for (let i = candidates.length - 1; i >= 0; i -= 1) {
        if (intersectionOverUnion(best, candidates[i]) > iouThreshold) {
          candidates.splice(i, 1);
        }
      }
    }
  }

  return selected.sort((a, b) => b.score - a.score);
}

function parseOutput(
  output: ort.Tensor,
  letterbox: LetterboxInfo,
  originalWidth: number,
  originalHeight: number,
  confidenceThreshold: number,
  iouThreshold: number
) {
  const data = output.data as Float32Array;
  const [, channels, anchors] = output.dims;
  const classCount = channels - 4;
  const boxes: Box[] = [];
  let bestPrediction: PredictionSummary | null = null;

  for (let i = 0; i < anchors; i += 1) {
    let classId = 0;
    let score = -Infinity;

    for (let c = 0; c < classCount; c += 1) {
      const classScore = data[(4 + c) * anchors + i];
      if (classScore > score) {
        score = classScore;
        classId = c;
      }
    }

    if (!bestPrediction || score > bestPrediction.score) {
      bestPrediction = { score, classId };
    }

    if (score < confidenceThreshold) {
      continue;
    }

    const cx = data[i];
    const cy = data[anchors + i];
    const width = data[anchors * 2 + i];
    const height = data[anchors * 3 + i];

    const x1 = (cx - width / 2 - letterbox.padX) / letterbox.ratio;
    const y1 = (cy - height / 2 - letterbox.padY) / letterbox.ratio;
    const x2 = (cx + width / 2 - letterbox.padX) / letterbox.ratio;
    const y2 = (cy + height / 2 - letterbox.padY) / letterbox.ratio;

    boxes.push({
      x1: clamp(x1, 0, originalWidth),
      y1: clamp(y1, 0, originalHeight),
      x2: clamp(x2, 0, originalWidth),
      y2: clamp(y2, 0, originalHeight),
      score,
      classId
    });
  }

  return {
    boxes: nonMaxSuppression(boxes, iouThreshold),
    bestPrediction
  };
}

function drawResult(canvas: HTMLCanvasElement, image: LoadedImage, boxes: Box[]) {
  canvas.width = image.width;
  canvas.height = image.height;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Canvas tidak tersedia.");
  }

  ctx.drawImage(image.bitmap, 0, 0, image.width, image.height);
  const lineWidth = Math.max(2, Math.round(Math.min(image.width, image.height) / 220));
  const fontSize = Math.max(13, Math.round(Math.min(image.width, image.height) / 36));

  for (const box of boxes) {
    const color = CLASS_COLORS[box.classId] ?? "#111827";
    const label = `${CLASS_NAMES[box.classId] ?? `class_${box.classId}`} ${(box.score * 100).toFixed(1)}%`;
    const x = box.x1;
    const y = box.y1;
    const width = Math.max(1, box.x2 - box.x1);
    const height = Math.max(1, box.y2 - box.y1);

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x, y, width, height);

    ctx.font = `600 ${fontSize}px Inter, Arial, sans-serif`;
    const metrics = ctx.measureText(label);
    const labelWidth = metrics.width + 12;
    const labelHeight = fontSize + 9;
    const labelY = y - labelHeight >= 0 ? y - labelHeight : y;

    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, labelWidth, labelHeight);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 6, labelY + fontSize + 2);
  }
}

function App() {
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [status, setStatus] = useState("Siap");
  const [isBusy, setIsBusy] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.5);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const classSummary = useMemo(() => {
    return CLASS_NAMES.map((name, classId) => ({
      name,
      count: boxes.filter((box) => box.classId === classId).length,
      color: CLASS_COLORS[classId]
    }));
  }, [boxes]);

  useEffect(() => {
    getSession()
      .then(() => setStatus("Model siap"))
      .catch((error) => {
        console.error(error);
        setStatus(`Model gagal: ${getShortErrorMessage(error)}`);
      });
  }, []);

  const loadFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setStatus("File bukan gambar");
      return;
    }

    setIsBusy(true);
    setBoxes([]);
    setStatus("Memuat gambar");

    try {
      const bitmap = await createImageBitmap(file);
      const loaded: LoadedImage = {
        fileName: file.name,
        bitmap,
        width: bitmap.width,
        height: bitmap.height
      };

      setImage(loaded);
      const canvas = canvasRef.current;
      if (canvas) {
        drawResult(canvas, loaded, []);
      }
      setStatus("Gambar siap");
    } catch (error) {
      console.error(error);
      setStatus("Gambar gagal dimuat");
    } finally {
      setIsBusy(false);
    }
  }, []);

  const runDetection = useCallback(async () => {
    if (!image || !canvasRef.current) {
      setStatus("Pilih gambar");
      return;
    }

    setIsBusy(true);
    setStatus("Deteksi berjalan");

    try {
      const session = await getSession();
      const { tensor, letterbox } = preprocess(image.bitmap);
      const inputName = session.inputNames[0];
      const outputName = session.outputNames[0];
      const result = await session.run({ [inputName]: tensor });
      const detection = parseOutput(
        result[outputName],
        letterbox,
        image.width,
        image.height,
        confidence,
        iou
      );
      const detectedBoxes = detection.boxes;

      drawResult(canvasRef.current, image, detectedBoxes);
      setBoxes(detectedBoxes);
      if (detectedBoxes.length > 0) {
        setStatus(`${detectedBoxes.length} objek`);
      } else if (detection.bestPrediction) {
        const bestName = CLASS_NAMES[detection.bestPrediction.classId] ?? `class_${detection.bestPrediction.classId}`;
        setStatus(`0 objek, max ${bestName} ${(detection.bestPrediction.score * 100).toFixed(1)}%`);
      } else {
        setStatus("0 objek");
      }
    } catch (error) {
      console.error(error);
      setStatus(`Gagal: ${getShortErrorMessage(error)}`);
    } finally {
      setIsBusy(false);
    }
  }, [confidence, image, iou]);

  const reset = useCallback(() => {
    setImage(null);
    setBoxes([]);
    setStatus("Siap");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  const download = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) {
      return;
    }

    const link = document.createElement("a");
    const baseName = image.fileName.replace(/\.[^/.]+$/, "");
    link.href = canvas.toDataURL("image/png");
    link.download = `${baseName}-detected.png`;
    link.click();
  }, [image]);

  const onInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        void loadFile(file);
      }
    },
    [loadFile]
  );

  const onDrop = useCallback(
    (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      setIsDragging(false);
      const file = event.dataTransfer.files?.[0];
      if (file) {
        void loadFile(file);
      }
    },
    [loadFile]
  );

  return (
    <main className="app-shell">
      <section className="workspace">
        <aside className="side-panel">
          <div className="brand-row">
            <div className="brand-mark">
              <FileImage size={20} aria-hidden="true" />
            </div>
            <div>
              <h1>ABU26 Detector</h1>
              <p>{MODEL_NAME}</p>
            </div>
          </div>

          <label
            className={`drop-zone ${isDragging ? "is-dragging" : ""}`}
            onDragEnter={() => setIsDragging(true)}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
          >
            <input ref={fileInputRef} type="file" accept="image/*" onChange={onInputChange} />
            <ImageUp size={30} aria-hidden="true" />
            <span>{image ? image.fileName : "Pilih gambar"}</span>
          </label>

          <div className="control-group">
            <div className="control-title">
              <SlidersHorizontal size={16} aria-hidden="true" />
              <span>Parameter</span>
            </div>

            <label className="range-row">
              <span>Confidence</span>
              <strong>{confidence.toFixed(2)}</strong>
              <input
                type="range"
                min="0.01"
                max="0.95"
                step="0.01"
                value={confidence}
                onChange={(event) => setConfidence(Number(event.target.value))}
              />
            </label>

            <label className="range-row">
              <span>IoU</span>
              <strong>{iou.toFixed(2)}</strong>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={iou}
                onChange={(event) => setIou(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="button-grid">
            <button className="primary-button" type="button" disabled={!image || isBusy} onClick={runDetection}>
              {isBusy ? <LoaderCircle className="spin" size={18} aria-hidden="true" /> : <Play size={18} aria-hidden="true" />}
              <span>Deteksi</span>
            </button>
            <button type="button" disabled={!image || isBusy} onClick={download} title="Download">
              <Download size={18} aria-hidden="true" />
            </button>
            <button type="button" disabled={!image || isBusy} onClick={reset} title="Reset">
              <RotateCcw size={18} aria-hidden="true" />
            </button>
          </div>

          <div className="status-box">
            <span>Status</span>
            <strong>{status}</strong>
          </div>

          <div className="class-list" aria-label="Ringkasan class">
            {classSummary.map((item) => (
              <div className="class-row" key={item.name}>
                <span style={{ backgroundColor: item.color }} />
                <p>{item.name}</p>
                <strong>{item.count}</strong>
              </div>
            ))}
          </div>
        </aside>

        <section className="canvas-panel">
          {!image && (
            <div className="empty-state">
              <FileImage size={38} aria-hidden="true" />
              <span>Belum ada gambar</span>
            </div>
          )}
          <canvas ref={canvasRef} className={image ? "result-canvas is-visible" : "result-canvas"} />
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
