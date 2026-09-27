# Standalone TimeFlowCodec player

The user approved continuing with the proposed standalone-player bundle. Deliver
local implementation and packaging support, not a public upload or release.

Use the existing decoder to render RGB directly in a PySide6 window: Open,
Play/Pause, frame seeking, and editable FPS (30 default, clearly marked as assumed
because v1-v4 store no timestamps/audio). Decode in a child process, write a private
temporary NumPy cache, and memory-map it in the UI. Cancel/reopen/close must stop
that child and release the cache. A failed decode leaves a useful error and permits
another open. Loading never blocks Qt event processing.

Cap declared decoded RGB size at 256 MiB and input size at 64 MiB for this initial
short-clip player. These checks are not a complete hostile-container sandbox or a
process memory guarantee: payload validation remains roadmap item 2. No manual
conversion or lossy intermediate video. Playback timing follows a monotonic clock
and skips display frames when rendering falls behind; seeking uses frame indexes.

Ship timeflowplay and timeflowcodec-gui launchers in the wheel. Preserve existing
root gui imports/tests. Add playback access to the existing GUI and handle command
line file paths and macOS file-open events. The bundle advertises a .tfc document
type on macOS; file associations elsewhere remain platform installer work. Keep
Windows builds free of macOS-only BUNDLE execution.

Verify real decode subprocesses, native RGB output, playback/pause/seek/FPS,
cancellation, malformed-file recovery, file-open dispatch, wheel contents and
installed launchers. Run existing implemented codec tests separately from known
unfinished media/safety/GUI jobs tests. Build a local bundle if tooling permits;
report untested OS/signing/distribution limitations explicitly.
