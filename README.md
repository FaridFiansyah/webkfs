# ABU26 Detector Web

Web statis untuk upload gambar, menjalankan model ONNX `bestforest25.onnx`, lalu menampilkan gambar dengan bounding box.

Folder ini sudah siap dijadikan satu repository GitHub.

## Jalankan Lokal

```bash
npm install
npm run dev
```

Buka `http://localhost:5173/`.

## Build Production

```bash
npm run build
```

Output production ada di `dist/`.

## Deploy ke Vercel

Project ini sudah memakai Vite dan `vercel.json`, jadi bisa langsung deploy dari root folder:

```bash
vercel --prod
```

Vercel akan menjalankan `npm install` dan `npm run build`, lalu menyajikan folder `dist`.

## Update Model

Vercel tidak menjalankan PyTorch `.pt` secara langsung. Model perlu tersedia sebagai ONNX di folder `public/` agar inferensi bisa berjalan di browser. Model aktif saat ini adalah `public/bestforest25.onnx`, hasil ekspor dari `bestforest25.pt`.

```bash
python scripts/export_bestforest25.py
```
