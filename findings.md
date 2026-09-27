# Findings

Prior evaluation: docs/evaluation/improvement-plan-2026-09-14.md.
Motion vector uint8 subtraction wraps negative offsets; warped samples can lie outside 0–255.
Legacy decoder reads entire plane payloads; chunked outer container required for bounded streaming.
Existing tests monkeypatch encoder.imageio readers and decoder writers; preserve this compatibility.
