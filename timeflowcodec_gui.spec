# -*- mode: python -*-

block_cipher = None

import platform

from PyInstaller.utils.hooks import collect_all
from timeflowcodec.version import __version__

datas = [("demo_assets/demo_input.mp4", "demo_assets"), ("demo_assets/demo_input.tfc", "demo_assets")]
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
    "CFBundleShortVersionString": __version__,
    "CFBundleVersion": __version__,
    "LSMinimumSystemVersion": "13.0",
    "NSHighResolutionCapable": True,
    "CFBundleDocumentTypes": [{
        "CFBundleTypeName": "TimeFlowCodec video",
        "CFBundleTypeRole": "Viewer",
        "LSHandlerRank": "Alternate",
        "LSItemContentTypes": ["com.timeflowcodec.video"],
    }],
    "UTExportedTypeDeclarations": [{
        "UTTypeIdentifier": "com.timeflowcodec.video",
        "UTTypeDescription": "TimeFlowCodec video",
        "UTTypeConformsTo": ["public.data"],
        "UTTypeTagSpecification": {"public.filename-extension": ["tfc"]},
    }],
}

a = Analysis(
    ['gui.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['timeflowcodec', 'timeflowcodec.player_decode'] + hiddenimports,
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
    [],
    exclude_binaries=True,
    name='TimeFlowCodec',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
)

collection = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='TimeFlowCodec',
)

if platform.system() == "Darwin":
    app = BUNDLE(
        collection,
        name='TimeFlowCodec.app',
        icon=None,
        bundle_identifier='com.timeflowcodec.gui',
        info_plist=info_plist,
        manifest=None,
        resources=[],
        target_arch=target_arch,
    )
