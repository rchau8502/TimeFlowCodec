"""Tests for WebAssembly Decoder Build and Node.js Runtime Verification."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WASM_FILE = REPO_ROOT / "web" / "libtfc.wasm"


def test_wasm_binary_exists_and_valid():
    """Verify libtfc.wasm exists and has valid WebAssembly magic header."""
    assert WASM_FILE.exists(), f"WebAssembly binary {WASM_FILE} not found"
    data = WASM_FILE.read_bytes()
    assert len(data) > 1024, "WASM binary too small"
    # WebAssembly binary magic: '\0asm', version 1: 0x01 0x00 0x00 0x00
    assert data[:4] == b"\x00asm", "Invalid WebAssembly magic number"
    assert data[4:8] == b"\x01\x00\x00\x00", "Invalid WebAssembly version"


def test_wasm_node_execution():
    """Verify that libtfc.wasm loads and exports expected decoder API in Node.js."""
    node_script = """
    const fs = require('fs');
    const wasmBuffer = fs.readFileSync(process.argv[1]);
    const wasiMock = {
      wasi_snapshot_preview1: new Proxy({}, {
        get: (target, prop) => (...args) => 0
      })
    };
    WebAssembly.instantiate(wasmBuffer, wasiMock).then(({ instance }) => {
      const exp = instance.exports;
      if (typeof exp.tfc_wasm_alloc_in !== 'function') process.exit(1);
      if (typeof exp.tfc_wasm_decode !== 'function') process.exit(2);
      if (typeof exp.tfc_wasm_get_width !== 'function') process.exit(3);
      if (typeof exp.tfc_wasm_get_rgb !== 'function') process.exit(4);
      process.exit(0);
    }).catch(err => {
      console.error(err);
      process.exit(10);
    });
    """
    res = subprocess.run(
        ["node", "-e", node_script, str(WASM_FILE)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Node.js WASM execution failed: {res.stderr}"
