#!/usr/bin/env bash
set -euo pipefail

APP_NAME="TimeFlowCodec"
DIST_DIR="dist"
APP_PATH="$DIST_DIR/$APP_NAME.app"
DMG_PATH="$DIST_DIR/${APP_NAME}_macbook_installer.dmg"
ARCH_NAME="$(uname -m)"

echo "Building $APP_NAME for macOS architecture: $ARCH_NAME"

python3 -m pip install --upgrade pyinstaller "imageio[ffmpeg]" PySide6 zstandard
pyinstaller --noconfirm --clean timeflowcodec_gui.spec

if [[ ! -d "$APP_PATH" ]]; then
  echo "Build failed: $APP_PATH not found"
  exit 1
fi

if command -v hdiutil >/dev/null 2>&1; then
  rm -f "$DMG_PATH"
  hdiutil create \
    -volname "$APP_NAME Installer" \
    -srcfolder "$APP_PATH" \
    -ov \
    -format UDZO \
    "$DMG_PATH"
  echo "Created DMG: $DMG_PATH"
else
  echo "hdiutil not found; skipping DMG creation"
fi

echo "macOS build complete: $APP_PATH"
