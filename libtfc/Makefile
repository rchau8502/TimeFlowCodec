CC ?= clang
CFLAGS ?= -O3 -Wall -Wextra -std=c99 -Iinclude -DHAVE_ZLIB=1
LDFLAGS ?= -lz -lm

# Auto-detect zstd via pkg-config or standard paths
ZSTD_PKG_CFLAGS := $(shell pkg-config --cflags libzstd 2>/dev/null)
ZSTD_PKG_LIBS := $(shell pkg-config --libs libzstd 2>/dev/null)

ifneq ($(ZSTD_PKG_LIBS),)
  ZSTD_INC ?= $(ZSTD_PKG_CFLAGS) -DHAVE_ZSTD=1
  ZSTD_LIB ?= $(ZSTD_PKG_LIBS)
else
  ZSTD_INC ?= $(shell if [ -f /opt/homebrew/opt/zstd/include/zstd.h ]; then echo "-I/opt/homebrew/opt/zstd/include -DHAVE_ZSTD=1"; \
                      elif [ -f /opt/anaconda3/include/zstd.h ]; then echo "-I/opt/anaconda3/include -DHAVE_ZSTD=1"; \
                      elif [ -f /usr/include/zstd.h ] || [ -f /usr/local/include/zstd.h ]; then echo "-DHAVE_ZSTD=1"; fi)
  ZSTD_LIB ?= $(shell if [ -f /opt/homebrew/opt/zstd/lib/libzstd.dylib ] || [ -f /opt/homebrew/opt/zstd/lib/libzstd.a ]; then echo "-L/opt/homebrew/opt/zstd/lib -lzstd"; \
                      elif [ -f /opt/anaconda3/lib/libzstd.dylib ]; then echo "-L/opt/anaconda3/lib -lzstd"; \
                      elif [ -f /usr/include/zstd.h ] || [ -f /usr/local/include/zstd.h ] || [ -f /usr/lib/libzstd.so ] || [ -f /usr/local/lib/libzstd.so ] || [ -n "$$(ls /usr/lib/*-linux-gnu/libzstd.so 2>/dev/null)" ]; then echo "-lzstd"; fi)
endif

CFLAGS += $(ZSTD_INC)
LDFLAGS += $(ZSTD_LIB)

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

OBJS = $(BUILD_DIR)/tfc_decoder.o
TARGET_LIB = $(BUILD_DIR)/libtfc.a
TARGET_BIN = $(BUILD_DIR)/tfc_dec

.PHONY: all clean

all: $(TARGET_LIB) $(TARGET_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/tfc_decoder.o: $(SRC_DIR)/tfc_decoder.c $(INC_DIR)/tfc.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET_LIB): $(BUILD_DIR)/tfc_decoder.o
	ar rcs $@ $^

$(TARGET_BIN): $(SRC_DIR)/tfc_dec.c $(TARGET_LIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(TARGET_LIB) $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)

WASM_CC ?= /opt/homebrew/opt/llvm/bin/clang
WASM_LD ?= /opt/homebrew/bin/wasm-ld
WASI_SYSROOT ?= /opt/homebrew/share/wasi-sysroot
WASM_TARGET ?= ../web/libtfc.wasm

wasm: $(WASM_TARGET)

$(WASM_TARGET): $(SRC_DIR)/tfc_decoder.c $(SRC_DIR)/tfc_wasm.c $(SRC_DIR)/wasm_rt.c $(INC_DIR)/tfc.h | $(BUILD_DIR)
	$(WASM_CC) --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) -c -O3 -I$(INC_DIR) -DHAVE_ZLIB=0 -DHAVE_ZSTD=0 $(SRC_DIR)/tfc_decoder.c -o $(BUILD_DIR)/tfc_decoder_wasm.o
	$(WASM_CC) --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) -c -O3 -I$(INC_DIR) -DHAVE_ZLIB=0 -DHAVE_ZSTD=0 $(SRC_DIR)/tfc_wasm.c -o $(BUILD_DIR)/tfc_wasm.o
	$(WASM_CC) --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) -c -O3 $(SRC_DIR)/wasm_rt.c -o $(BUILD_DIR)/wasm_rt.o
	mkdir -p ../web
	$(WASM_LD) --no-entry --export-all -O3 $(BUILD_DIR)/tfc_decoder_wasm.o $(BUILD_DIR)/tfc_wasm.o $(BUILD_DIR)/wasm_rt.o -L$(WASI_SYSROOT)/lib/wasm32-wasip1 -lc -lm -o $(WASM_TARGET)

