# TimeFlowCodec container v5 (release candidate)

v5 preserves the existing v3/v4 segment codecs inside independent chunks. v1–v4
readers and explicit legacy encoding remain available. A v5 file requires the
updated TimeFlowCodec application; registering its extension does not add a decoder
to QuickTime, browsers, VLC, or arbitrary applications.

The file starts with ASCII `TFC5`, a little-endian uint32 JSON length (at most
16 KiB), then UTF-8 JSON. Required fields are positive integer `width`, `height`,
`num_frames`, `fps_num`, `fps_den`, and `chunk_frames`, plus boolean `has_audio`.
`version` is 5. Additional source-start fields are diagnostic metadata; arbitrary
source timeline offsets are not yet a validated release capability.

Each following record has `<4sQI`: four-byte tag, uint64 byte length, uint32 CRC32.
`VIDS` contains a full legacy v3/v4 file, limited to 128 MiB. Video records are
ordered and every record except the last contains `chunk_frames` frames. A single
optional `AUDI` record follows video and stores the first audio stream in Matroska
(up to 1 GiB), copied without re-encoding during compression. An empty `DONE`
record terminates the file, with no trailing bytes permitted. Record checksums,
frame totals, dimensions, ordering, and chunk boundaries are validated on decode.
CRC32 detects corruption; it does not authenticate files.

Encoding uses at most the requested window (default 16 frames), additionally
limited to 16 MiB of input RGB per chunk, or one frame when a frame exceeds that
budget. This is a chunk-size target, not a total process RAM guarantee: predictor
arrays, dictionaries, residuals, and compression workspaces also use memory.

Source probing checks every decoded video timestamp. Irregular/variable frame
rates are rejected with an explicit error; v5 does not silently normalize them.
A rational frame rate is retained and exported. Legacy files use 30 FPS unless
explicitly overridden. FPS override with audio is rejected. Encoding `max_frames`
with audio is rejected until soundtrack trimming is implemented and verified.
Only the first video and first audio stream are included; subtitles, extra tracks,
HDR metadata, and arbitrary timeline offsets are outside the current release scope.

The player creates an exact RGB cache on temporary disk before playback. v5 cache
creation and frame reads are bounded in memory; disk space scales as
`width × height × frame_count × 3` bytes plus audio and scratch space. Free space
is checked first, loading is cancellable, and closing clears the cache. Playback
uses source FPS and Qt audio playback. Startup latency still scales with duration;
instant startup and a bounded disk cache remain future work.

Encoding and export replace destinations only after success. A failed or cancelled
operation removes its temporary output and preserves an existing destination.
