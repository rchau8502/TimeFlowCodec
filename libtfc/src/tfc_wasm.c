#include "tfc.h"
#include <stdlib.h>
#include <string.h>

/* Global decoder state in WebAssembly linear memory */
static uint8_t *g_in_buf = NULL;
static size_t g_in_capacity = 0;
static uint8_t *g_out_rgb24 = NULL;
static size_t g_out_size = 0;
static tfc_header_t g_header;
static int g_is_decoded = 0;

__attribute__((visibility("default")))
uint8_t* tfc_wasm_alloc_in(size_t size) {
    if (size > g_in_capacity) {
        if (g_in_buf) free(g_in_buf);
        g_in_buf = (uint8_t*)malloc(size);
        g_in_capacity = size;
    }
    return g_in_buf;
}

__attribute__((visibility("default")))
void tfc_wasm_free(void *ptr) {
    if (ptr) free(ptr);
}

__attribute__((visibility("default")))
int tfc_wasm_decode(size_t in_size) {
    if (!g_in_buf || in_size < 8) return -1;

    if (g_out_rgb24) {
        free(g_out_rgb24);
        g_out_rgb24 = NULL;
        g_out_size = 0;
    }
    g_is_decoded = 0;

    tfc_status_t status = tfc_read_header(g_in_buf, in_size, &g_header);
    if (status != TFC_OK) return (int)status;

    size_t total_bytes = (size_t)g_header.width * g_header.height * g_header.num_frames * 3;
    g_out_rgb24 = (uint8_t *)malloc(total_bytes);
    if (!g_out_rgb24) return (int)TFC_ERR_OUT_OF_MEMORY;
    g_out_size = total_bytes;

    status = tfc_decode_v3_v4_memory(g_in_buf, in_size, g_out_rgb24, g_out_size, &g_header);
    if (status != TFC_OK) {
        free(g_out_rgb24);
        g_out_rgb24 = NULL;
        g_out_size = 0;
        return (int)status;
    }

    g_is_decoded = 1;
    return 0;
}

__attribute__((visibility("default")))
uint32_t tfc_wasm_get_width(void) {
    return g_is_decoded ? g_header.width : 0;
}

__attribute__((visibility("default")))
uint32_t tfc_wasm_get_height(void) {
    return g_is_decoded ? g_header.height : 0;
}

__attribute__((visibility("default")))
uint32_t tfc_wasm_get_num_frames(void) {
    return g_is_decoded ? g_header.num_frames : 0;
}

__attribute__((visibility("default")))
uint8_t* tfc_wasm_get_rgb(void) {
    return g_out_rgb24;
}

__attribute__((visibility("default")))
uint8_t* tfc_wasm_get_frame_ptr(uint32_t frame_idx) {
    if (!g_is_decoded || !g_out_rgb24 || frame_idx >= g_header.num_frames) return NULL;
    size_t frame_bytes = (size_t)g_header.width * g_header.height * 3;
    return g_out_rgb24 + (size_t)frame_idx * frame_bytes;
}

__attribute__((visibility("default")))
void tfc_wasm_cleanup(void) {
    if (g_in_buf) { free(g_in_buf); g_in_buf = NULL; g_in_capacity = 0; }
    if (g_out_rgb24) { free(g_out_rgb24); g_out_rgb24 = NULL; g_out_size = 0; }
    g_is_decoded = 0;
}
