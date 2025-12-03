PY=python
PIP=pip

install:
	$(PIP) install -e .[bench]
	$(PIP) install ruff pytest

lint:
	ruff check .

format:
	ruff check --select I --fix .

test:
	pytest

bench-smoke:
	mkdir -p /tmp/tfc_clips
	ffmpeg -y -f lavfi -i testsrc=size=320x240:rate=10 -t 2 /tmp/tfc_clips/gradient.mp4
	ffmpeg -y -f lavfi -i noise=alls=20:allf=t+u:size=320x240:rate=10 -t 2 /tmp/tfc_clips/noise.mp4
	$(PY) -m timeflowcodec.bench --clips /tmp/tfc_clips --out /tmp/tfc_out --codecs tfc --presets medium --crf 23

bench:
	$(PY) -m timeflowcodec.bench --clips clips --out bench_out --codecs tfc,x264,x265,av1 --presets fast,medium --crf 18,23,28

.PHONY: install lint format test bench bench-smoke
