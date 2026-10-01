CC ?= clang
CFLAGS ?= -O3 -Wall -Wextra -std=c99 -Iinclude -DHAVE_ZLIB=1
LDFLAGS ?= -lz

ZSTD_INC ?= $(shell if [ -f /opt/homebrew/opt/zstd/include/zstd.h ]; then echo "-I/opt/homebrew/opt/zstd/include -DHAVE_ZSTD=1"; elif [ -f /opt/anaconda3/include/zstd.h ]; then echo "-I/opt/anaconda3/include -DHAVE_ZSTD=1"; elif [ -f /usr/include/zstd.h ] || [ -f /usr/local/include/zstd.h ]; then echo "-DHAVE_ZSTD=1"; fi)
ZSTD_LIB ?= $(shell if [ -f /opt/homebrew/opt/zstd/lib/libzstd.dylib ] || [ -f /opt/homebrew/opt/zstd/lib/libzstd.a ]; then echo "-L/opt/homebrew/opt/zstd/lib -lzstd"; elif [ -f /opt/anaconda3/lib/libzstd.dylib ]; then echo "-L/opt/anaconda3/lib -lzstd"; elif [ -f /usr/lib/libzstd.so ] || [ -f /usr/local/lib/libzstd.so ]; then echo "-lzstd"; fi)

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
