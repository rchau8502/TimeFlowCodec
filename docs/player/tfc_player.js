// TimeFlowCodec WebAssembly Player Runtime

class TFCPlayer {
  constructor() {
    this.canvas = document.getElementById("viewCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.playBtn = document.getElementById("playBtn");
    this.restartBtn = document.getElementById("restartBtn");
    this.slider = document.getElementById("timeSlider");
    this.statusText = document.getElementById("statusText");
    this.fpsBadge = document.getElementById("fpsBadge");
    this.metaStats = document.getElementById("metaStats");
    this.dropOverlay = document.getElementById("dropOverlay");
    this.fileInput = document.getElementById("fileInput");
    this.loadDemoBtn = document.getElementById("loadDemoBtn");
    this.loadDemoOverlayBtn = document.getElementById("loadDemoOverlayBtn");

    this.wasmInstance = null;
    this.wasmMemory = null;
    this.isPlaying = false;
    this.currentFrame = 0;
    this.totalFrames = 0;
    this.width = 0;
    this.height = 0;
    this.fps = 30.0;
    this.lastFrameTime = 0;
    this.rafId = null;

    this.initEventListeners();
    this.loadWasm();
  }

  async loadWasm() {
    try {
      this.statusText.textContent = "Loading libtfc.wasm...";
      const response = await fetch("libtfc.wasm");
      if (!response.ok) {
        throw new Error(`Failed to fetch libtfc.wasm (${response.status})`);
      }
      const wasmBytes = await response.arrayBuffer();
      const importObject = {
        wasi_snapshot_preview1: new Proxy({}, {
          get: (target, prop) => (...args) => 0
        }),
        env: {
          abort: () => console.error("WASM aborted"),
        }
      };
      const { instance } = await WebAssembly.instantiate(wasmBytes, importObject);
      this.wasmInstance = instance;
      this.wasmMemory = instance.exports.memory;
      this.statusText.textContent = "WASM core loaded. Ready for .tfc file.";
    } catch (err) {
      console.warn("WASM binary load note:", err.message);
      this.statusText.textContent = "Drop a .tfc file to start.";
    }
  }

  initEventListeners() {
    this.dropOverlay.addEventListener("click", () => this.fileInput.click());
    this.fileInput.addEventListener("change", (e) => {
      if (e.target.files.length) this.handleFile(e.target.files[0]);
    });

    window.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.dropOverlay.classList.add("dragover");
    });
    window.addEventListener("dragleave", (e) => {
      e.preventDefault();
      this.dropOverlay.classList.remove("dragover");
    });
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      this.dropOverlay.classList.remove("dragover");
      if (e.dataTransfer.files.length) this.handleFile(e.dataTransfer.files[0]);
    });

    this.playBtn.addEventListener("click", () => this.togglePlay());
    this.restartBtn.addEventListener("click", () => this.seek(0));
    this.slider.addEventListener("input", (e) => this.seek(parseInt(e.target.value, 10)));

    const handleDemo = (e) => {
      e.stopPropagation();
      this.loadDemo();
    };
    if (this.loadDemoBtn) this.loadDemoBtn.addEventListener("click", handleDemo);
    if (this.loadDemoOverlayBtn) this.loadDemoOverlayBtn.addEventListener("click", handleDemo);
  }

  async loadDemo() {
    try {
      this.statusText.textContent = "Loading bundled demo.tfc...";
      const resp = await fetch("demo.tfc");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      this.decodeTfc(new Uint8Array(buf));
      this.play();
    } catch (err) {
      console.error(err);
      this.statusText.textContent = `Demo load failed: ${err.message}`;
    }
  }

  async handleFile(file) {
    this.statusText.textContent = `Reading ${file.name}...`;
    const buffer = await file.arrayBuffer();
    this.decodeTfc(new Uint8Array(buffer));
  }

  decodeTfc(bytes) {
    if (!this.wasmInstance) {
      alert("WASM core not initialized.");
      return;
    }
    const exports = this.wasmInstance.exports;
    const inPtr = exports.tfc_wasm_alloc_in(bytes.byteLength);
    const wasmView = new Uint8Array(this.wasmMemory.buffer, inPtr, bytes.byteLength);
    wasmView.set(bytes);

    this.statusText.textContent = "Decoding TFC bitstream...";
    const t0 = performance.now();
    const status = exports.tfc_wasm_decode(bytes.byteLength);
    const decodeTime = (performance.now() - t0).toFixed(1);

    if (status !== 0) {
      this.statusText.textContent = `Decode failed with error code: ${status}`;
      return;
    }

    this.width = exports.tfc_wasm_get_width();
    this.height = exports.tfc_wasm_get_height();
    this.totalFrames = exports.tfc_wasm_get_num_frames();

    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.slider.max = this.totalFrames - 1;
    this.slider.disabled = false;
    this.playBtn.disabled = false;
    this.restartBtn.disabled = false;
    this.dropOverlay.style.display = "none";

    this.statusText.textContent = `Decoded ${this.totalFrames} frames in ${decodeTime}ms`;
    this.metaStats.textContent = `${this.width}x${this.height} | ${this.totalFrames} frames`;
    this.renderFrame(0);
  }

  renderFrame(frameIdx) {
    if (frameIdx < 0 || frameIdx >= this.totalFrames) return;
    this.currentFrame = frameIdx;
    this.slider.value = frameIdx;

    const exports = this.wasmInstance.exports;
    const framePtr = exports.tfc_wasm_get_frame_ptr(frameIdx);
    if (!framePtr) return;

    const rgbView = new Uint8Array(this.wasmMemory.buffer, framePtr, this.width * this.height * 3);
    const imgData = this.ctx.createImageData(this.width, this.height);
    const data = imgData.data;

    let srcIdx = 0;
    let dstIdx = 0;
    const totalPixels = this.width * this.height;

    for (let p = 0; p < totalPixels; ++p) {
      data[dstIdx] = rgbView[srcIdx];
      data[dstIdx + 1] = rgbView[srcIdx + 1];
      data[dstIdx + 2] = rgbView[srcIdx + 2];
      data[dstIdx + 3] = 255;
      srcIdx += 3;
      dstIdx += 4;
    }

    this.ctx.putImageData(imgData, 0, 0);
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    if (this.currentFrame >= this.totalFrames - 1) {
      this.currentFrame = 0;
    }
    this.isPlaying = true;
    this.playBtn.textContent = "Pause";
    this.lastFrameTime = performance.now();
    this.tick();
  }

  pause() {
    this.isPlaying = false;
    this.playBtn.textContent = "Play";
    if (this.rafId) cancelAnimationFrame(this.rafId);
  }

  seek(frameIdx) {
    this.pause();
    this.renderFrame(frameIdx);
  }

  tick() {
    if (!this.isPlaying) return;
    const now = performance.now();
    const frameInterval = 1000.0 / this.fps;

    if (now - this.lastFrameTime >= frameInterval) {
      this.currentFrame++;
      if (this.currentFrame >= this.totalFrames) {
        this.currentFrame = this.totalFrames - 1;
        this.pause();
        return;
      }
      this.renderFrame(this.currentFrame);
      this.lastFrameTime = now;
    }
    this.rafId = requestAnimationFrame(() => this.tick());
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.player = new TFCPlayer();
});
