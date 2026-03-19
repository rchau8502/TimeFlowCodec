# -*- mode: python -*-

block_cipher = None

import platform

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

for pkg in ("imageio", "imageio_ffmpeg"):
    _d, _b, _h = collect_all(pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

target_arch = None
if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
    target_arch = "arm64"

info_plist = {
    "CFBundleDisplayName": "TimeFlowCodec",
    "CFBundleName": "TimeFlowCodec",
    "CFBundleShortVersionString": "0.2.0",
    "CFBundleVersion": "0.2.0",
    "LSMinimumSystemVersion": "12.0",
    "NSHighResolutionCapable": True,
}

a = Analysis(
    ['gui.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['timeflowcodec'] + hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TimeFlowCodec',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
)

app = BUNDLE(
    exe,
    name='TimeFlowCodec.app',
    icon=None,
    bundle_identifier='com.timeflowcodec.gui',
    info_plist=info_plist,
    manifest=None,
    resources=[],
    target_arch=target_arch,
)
