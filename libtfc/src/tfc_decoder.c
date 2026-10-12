#include "tfc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if !defined(__wasm__) && (defined(HAVE_ZLIB) || defined(__APPLE__) || defined(__linux__))
#include <zlib.h>
#define TFC_HAS_ZLIB 1
#else
#define TFC_HAS_ZLIB 0
#endif

#if !defined(__wasm__) && defined(HAVE_ZSTD) && (HAVE_ZSTD != 0)
#include <zstd.h>
#define TFC_HAS_ZSTD 1
#else
#define TFC_HAS_ZSTD 0
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TFC_HAS_NEON 1
#else
#define TFC_HAS_NEON 0
#endif

static inline uint8_t clamp_u8(float v) {
    int iv = (int)roundf(v);
    if (iv < 0) return 0;
    if (iv > 255) return 255;
    return (uint8_t)iv;
}

static inline void yuv_to_rgb(float y, float u, float v, uint8_t *out_rgb) {
    float uf = u - 128.0f;
    float vf = v - 128.0f;
    out_rgb[0] = clamp_u8(y + 1.402f * vf);
    out_rgb[1] = clamp_u8(y - 0.344136f * uf - 0.714136f * vf);
    out_rgb[2] = clamp_u8(y + 1.772f * uf);
}

const char *tfc_status_to_string(tfc_status_t status) {
    switch (status) {
        case TFC_OK: return "Success";
        case TFC_ERR_INVALID_MAGIC: return "Invalid container magic";
        case TFC_ERR_UNSUPPORTED_VERSION: return "Unsupported container version";
        case TFC_ERR_CORRUPT_HEADER: return "Corrupt or truncated container header";
        case TFC_ERR_UNSUPPORTED_COMPRESSION: return "Unsupported payload compression";
        case TFC_ERR_CORRUPT_STREAM: return "Corrupt bitstream payload";
        case TFC_ERR_OUT_OF_MEMORY: return "Out of memory";
        case TFC_ERR_IO: return "I/O error";
        case TFC_ERR_BUFFER_TOO_SMALL: return "Destination buffer too small";
        case TFC_ERR_CHECKSUM_MISMATCH: return "CRC32 checksum mismatch";
        default: return "Unknown error";
    }
}

void tfc_free(void *ptr) {
    if (ptr) {
        free(ptr);
    }
}

static inline uint16_t read_u16_le(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static inline int16_t read_i16_le(const uint8_t *p) {
    return (int16_t)(p[0] | (p[1] << 8));
}

static inline uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t read_u64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= ((uint64_t)p[i]) << (8 * i);
    }
    return v;
}

static inline float read_f32_le(const uint8_t *p) {
    uint32_t u = read_u32_le(p);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static int parse_json_uint(const char *json, const char *key, uint32_t *out) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *pos = strstr(json, pattern);
    if (!pos) {
        snprintf(pattern, sizeof(pattern), "\"%s\" :", key);
        pos = strstr(json, pattern);
    }
    if (!pos) return 0;
    pos += strlen(pattern);
    while (*pos == ' ' || *pos == '\t' || *pos == '\r' || *pos == '\n') pos++;
    *out = (uint32_t)strtoul(pos, NULL, 10);
    return 1;
}

tfc_status_t tfc_read_header(const uint8_t *buffer, size_t size, tfc_header_t *header) {
    if (!buffer || !header) return TFC_ERR_CORRUPT_HEADER;
    if (size < 8) return TFC_ERR_CORRUPT_HEADER;

    memset(header, 0, sizeof(tfc_header_t));

    if (memcmp(buffer, "TFC5", 4) == 0) {
        uint32_t meta_len = read_u32_le(buffer + 4);
        if (size < 8 + meta_len) return TFC_ERR_CORRUPT_HEADER;
        char *json = (char *)malloc(meta_len + 1);
        if (!json) return TFC_ERR_OUT_OF_MEMORY;
        memcpy(json, buffer + 8, meta_len);
        json[meta_len] = '\0';

        header->version = 5;
        parse_json_uint(json, "width", &header->width);
        parse_json_uint(json, "height", &header->height);
        parse_json_uint(json, "num_frames", &header->num_frames);
        header->fps_num = 30;
        header->fps_den = 1;
        parse_json_uint(json, "fps_num", &header->fps_num);
        parse_json_uint(json, "fps_den", &header->fps_den);
        header->color_format = TFC_COLOR_RGB;
        header->payload_comp_type = TFC_COMP_NONE;
        header->tiling = 1;
        header->segment_count = 1;

        free(json);
        return TFC_OK;
    }

    if (memcmp(buffer, "TFC1", 4) == 0) {
        if (size < 40) return TFC_ERR_CORRUPT_HEADER;
        uint16_t version = read_u16_le(buffer + 4);
        uint16_t hdr_size = read_u16_le(buffer + 6);
        if (version != 3 && version != 4) return TFC_ERR_UNSUPPORTED_VERSION;
        if (hdr_size < 40) return TFC_ERR_CORRUPT_HEADER;

        header->version = version;
        header->width = read_u32_le(buffer + 8);
        header->height = read_u32_le(buffer + 12);
        header->num_frames = read_u32_le(buffer + 16);
        header->color_format = buffer[20];
        header->payload_comp_type = buffer[22];
        header->tiling = read_u16_le(buffer + 23);
        header->segment_count = read_u16_le(buffer + 25);
        header->fps_num = 30;
        header->fps_den = 1;
        return TFC_OK;
    }

    return TFC_ERR_INVALID_MAGIC;
}

static tfc_status_t decompress_payload(
    const uint8_t *in, size_t in_size,
    uint8_t comp_type,
    uint8_t **out, size_t *out_size
) {
    if (comp_type == TFC_COMP_NONE) {
        *out = (uint8_t *)malloc(in_size);
        if (!*out) return TFC_ERR_OUT_OF_MEMORY;
        memcpy(*out, in, in_size);
        *out_size = in_size;
        return TFC_OK;
    }
#if TFC_HAS_ZLIB
    if (comp_type == TFC_COMP_ZLIB) {
        /* Estimate 8x expansion buffer, double if needed */
        uLongf dest_cap = (uLongf)(in_size * 8 + 65536);
        uint8_t *dest = (uint8_t *)malloc(dest_cap);
        if (!dest) return TFC_ERR_OUT_OF_MEMORY;
        while (1) {
            uLongf dest_len = dest_cap;
            int ret = uncompress(dest, &dest_len, in, (uLong)in_size);
            if (ret == Z_OK) {
                *out = dest;
                *out_size = (size_t)dest_len;
                return TFC_OK;
            } else if (ret == Z_BUF_ERROR && dest_cap < 256 * 1024 * 1024) {
                dest_cap *= 2;
                uint8_t *new_dest = (uint8_t *)realloc(dest, dest_cap);
                if (!new_dest) {
                    free(dest);
                    return TFC_ERR_OUT_OF_MEMORY;
                }
                dest = new_dest;
            } else {
                free(dest);
                return TFC_ERR_CORRUPT_STREAM;
            }
        }
    }
#endif
#if TFC_HAS_ZSTD
    if (comp_type == TFC_COMP_ZSTD) {
        unsigned long long rSize = ZSTD_getFrameContentSize(in, in_size);
        size_t dest_cap;
        if (rSize == ZSTD_CONTENTSIZE_ERROR || rSize == ZSTD_CONTENTSIZE_UNKNOWN) {
            dest_cap = in_size * 8 + 65536;
        } else {
            dest_cap = (size_t)rSize;
        }
        uint8_t *dest = (uint8_t *)malloc(dest_cap);
        if (!dest) return TFC_ERR_OUT_OF_MEMORY;
        size_t dSize = ZSTD_decompress(dest, dest_cap, in, in_size);
        if (ZSTD_isError(dSize)) {
            free(dest);
            return TFC_ERR_CORRUPT_STREAM;
        }
        *out = dest;
        *out_size = dSize;
        return TFC_OK;
    }
#endif
    return TFC_ERR_UNSUPPORTED_COMPRESSION;
}

static tfc_status_t decode_rle_modes(
    const uint8_t *rle_buf, size_t rle_len,
    uint8_t *modes_out, uint32_t logical_count
) {
    uint32_t decoded_modes = 0;
    size_t offset = 0;

    while (offset + 3 <= rle_len && decoded_modes < logical_count) {
        uint16_t run_len = read_u16_le(rle_buf + offset);
        uint8_t run_val = rle_buf[offset + 2];
        offset += 3;

        uint8_t m0 = (run_val >> 0) & 0x3;
        uint8_t m1 = (run_val >> 2) & 0x3;
        uint8_t m2 = (run_val >> 4) & 0x3;
        uint8_t m3 = (run_val >> 6) & 0x3;

        for (uint16_t r = 0; r < run_len && decoded_modes < logical_count; ++r) {
            if (decoded_modes < logical_count) modes_out[decoded_modes++] = m0;
            if (decoded_modes < logical_count) modes_out[decoded_modes++] = m1;
            if (decoded_modes < logical_count) modes_out[decoded_modes++] = m2;
            if (decoded_modes < logical_count) modes_out[decoded_modes++] = m3;
        }
    }

    if (decoded_modes != logical_count) {
        return TFC_ERR_CORRUPT_STREAM;
    }
    return TFC_OK;
}

tfc_status_t tfc_decode_v3_v4_memory(
    const uint8_t *in_buffer,
    size_t in_size,
    uint8_t *out_rgb24,
    size_t out_capacity,
    tfc_header_t *out_header
) {
    tfc_header_t hdr;
    tfc_status_t status = tfc_read_header(in_buffer, in_size, &hdr);
    if (status != TFC_OK) return status;
    if (out_header) *out_header = hdr;

    size_t required = (size_t)hdr.width * hdr.height * hdr.num_frames * 3;
    if (out_capacity < required) return TFC_ERR_BUFFER_TOO_SMALL;

    uint32_t W = hdr.width;
    uint32_t H = hdr.height;
    uint32_t T = hdr.num_frames;
    uint32_t tile_size = hdr.tiling > 0 ? hdr.tiling : 1;

    uint8_t *plane_bufs[3] = {NULL, NULL, NULL};
    uint32_t plane_w[3], plane_h[3];

    for (int p = 0; p < 3; ++p) {
        if (hdr.color_format == TFC_COLOR_YUV420 && p > 0) {
            plane_w[p] = (W + 1) / 2;
            plane_h[p] = (H + 1) / 2;
        } else {
            plane_w[p] = W;
            plane_h[p] = H;
        }
        if (hdr.color_format != TFC_COLOR_RGB) {
            plane_bufs[p] = (uint8_t *)malloc((size_t)plane_w[p] * plane_h[p] * T);
            if (!plane_bufs[p]) {
                for (int k = 0; k < p; ++k) free(plane_bufs[k]);
                return TFC_ERR_OUT_OF_MEMORY;
            }
        }
    }

    size_t offset = 40; /* Size of v3/v4 container header */

    /* Process 3 planes: 0=R, 1=G, 2=B (or Y, U, V) */
    for (int plane = 0; plane < 3; ++plane) {
        if (offset + 4 > in_size) {
            for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
            return TFC_ERR_CORRUPT_HEADER;
        }
        uint32_t payload_len = read_u32_le(in_buffer + offset);
        offset += 4;
        if (offset + payload_len > in_size) {
            for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
            return TFC_ERR_CORRUPT_HEADER;
        }

        uint8_t *decompressed = NULL;
        size_t decomp_size = 0;
        status = decompress_payload(in_buffer + offset, payload_len, hdr.payload_comp_type, &decompressed, &decomp_size);
        if (status != TFC_OK) {
            for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
            return status;
        }
        offset += payload_len;

        const uint8_t *p = decompressed;
        size_t p_left = decomp_size;

        if (p_left < 4) {
            free(decompressed);
            for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
            return TFC_ERR_CORRUPT_STREAM;
        }
        uint32_t seg_count = read_u32_le(p);
        p += 4; p_left -= 4;

        uint32_t cur_W = plane_w[plane];
        uint32_t cur_H = plane_h[plane];
        uint32_t tiles_x = (cur_W + tile_size - 1) / tile_size;
        uint32_t tiles_y = (cur_H + tile_size - 1) / tile_size;
        uint32_t expected_logical = tiles_x * tiles_y;

        uint8_t *frame_base = (hdr.color_format == TFC_COLOR_RGB) ? (out_rgb24 + plane) : plane_bufs[plane];
        size_t frame_stride = (hdr.color_format == TFC_COLOR_RGB) ? ((size_t)H * W * 3) : ((size_t)cur_H * cur_W);
        size_t pixel_stride = (hdr.color_format == TFC_COLOR_RGB) ? 3 : 1;
        size_t row_stride = (hdr.color_format == TFC_COLOR_RGB) ? ((size_t)W * 3) : (size_t)cur_W;

        uint8_t *modes = (uint8_t *)malloc(expected_logical);
        float *intercept = (float *)malloc(expected_logical * sizeof(float));
        float *slope = (float *)malloc(expected_logical * sizeof(float));
        int *is_raw = (int *)malloc(expected_logical * sizeof(int));
        int *raw_idx_map = (int *)malloc(expected_logical * sizeof(int));
        if (!modes || !intercept || !slope || !is_raw || !raw_idx_map) {
            if (modes) free(modes);
            if (intercept) free(intercept);
            if (slope) free(slope);
            if (is_raw) free(is_raw);
            if (raw_idx_map) free(raw_idx_map);
            free(decompressed);
            for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
            return TFC_ERR_OUT_OF_MEMORY;
        }

        uint32_t frame_cursor = 0;
        for (uint32_t s = 0; s < seg_count; ++s) {
            uint32_t seg_hdr_size = (hdr.version == 4) ? 36 : 32;
            if (p_left < seg_hdr_size) {
                free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                free(decompressed);
                for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                return TFC_ERR_CORRUPT_STREAM;
            }
            uint32_t seg_len = read_u32_le(p);
            uint32_t logical_count = read_u32_le(p + 4);
            uint32_t mode_len = read_u32_le(p + 8);
            float b_scale = read_f32_le(p + 12);
            float raw_scale = read_f32_le(p + 16);
            uint32_t const_count = read_u32_le(p + 20);
            uint32_t linear_count = read_u32_le(p + 24);
            uint32_t raw_count = read_u32_le(p + 28);
            uint32_t has_mvs = (hdr.version == 4) ? read_u32_le(p + 32) : 0;
            p += seg_hdr_size; p_left -= seg_hdr_size;

            if (logical_count != expected_logical) {
                free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                free(decompressed);
                for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                return TFC_ERR_CORRUPT_STREAM;
            }
            if (p_left < mode_len) {
                free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                free(decompressed);
                for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                return TFC_ERR_CORRUPT_STREAM;
            }

            status = decode_rle_modes(p, mode_len, modes, logical_count);
            p += mode_len; p_left -= mode_len;
            if (status != TFC_OK) {
                free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                free(decompressed);
                for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                return status;
            }

            if (has_mvs) {
                size_t mvs_len = (size_t)(seg_len - 1) * logical_count;
                if (p_left < mvs_len) {
                    free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                    free(decompressed);
                    for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                    return TFC_ERR_CORRUPT_STREAM;
                }
                /* Skip motion vector block for reference linear prediction */
                p += mvs_len; p_left -= mvs_len;
            }

            /* Fast path: pure 1-frame CONST segment (e.g. from native spatial compaction) */
            if (seg_len == 1 && const_count == logical_count && linear_count == 0 && raw_count == 0) {
                if (p_left < const_count) {
                    free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                    free(decompressed);
                    for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                    return TFC_ERR_CORRUPT_STREAM;
                }
                const uint8_t *const_vals = p;
                p += const_count; p_left -= const_count;

                if (frame_cursor < T) {
                    uint8_t *frame_ptr = frame_base + (size_t)frame_cursor * frame_stride;
                    if (tile_size == 1) {
                        if (pixel_stride == 1) {
                            memcpy(frame_ptr, const_vals, logical_count);
                        } else if (pixel_stride == 3) {
                            uint32_t i = 0;
                            for (; i + 8 <= logical_count; i += 8) {
                                uint8_t *p_out = frame_ptr + (size_t)i * 3;
                                p_out[0]  = const_vals[i];
                                p_out[3]  = const_vals[i + 1];
                                p_out[6]  = const_vals[i + 2];
                                p_out[9]  = const_vals[i + 3];
                                p_out[12] = const_vals[i + 4];
                                p_out[15] = const_vals[i + 5];
                                p_out[18] = const_vals[i + 6];
                                p_out[21] = const_vals[i + 7];
                            }
                            for (; i < logical_count; ++i) {
                                frame_ptr[(size_t)i * 3] = const_vals[i];
                            }
                        }
                    } else {
                        for (uint32_t ty = 0; ty < tiles_y; ++ty) {
                            uint32_t y_start = ty * tile_size;
                            uint32_t y_end = (y_start + tile_size < cur_H) ? (y_start + tile_size) : cur_H;
                            for (uint32_t tx = 0; tx < tiles_x; ++tx) {
                                uint32_t idx = ty * tiles_x + tx;
                                uint8_t byte_val = const_vals[idx];
                                uint32_t x_start = tx * tile_size;
                                uint32_t x_end = (x_start + tile_size < cur_W) ? (x_start + tile_size) : cur_W;
                                uint32_t span = x_end - x_start;
                                if (pixel_stride == 1) {
                                    for (uint32_t y = y_start; y < y_end; ++y) {
                                        uint8_t *row_ptr = frame_ptr + (size_t)y * row_stride + x_start;
                                        memset(row_ptr, byte_val, span);
                                    }
                                } else {
                                    for (uint32_t y = y_start; y < y_end; ++y) {
                                        uint8_t *row_ptr = frame_ptr + (size_t)y * row_stride;
                                        for (uint32_t x = x_start; x < x_end; ++x) {
                                            row_ptr[(size_t)x * pixel_stride] = byte_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                frame_cursor += 1;
                continue;
            }

            memset(intercept, 0, logical_count * sizeof(float));
            memset(slope, 0, logical_count * sizeof(float));
            memset(is_raw, 0, logical_count * sizeof(int));
            memset(raw_idx_map, 0, logical_count * sizeof(int));

            /* Read CONST stream */
            if (p_left < const_count) {
                free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                free(decompressed);
                for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                return TFC_ERR_CORRUPT_STREAM;
            }
            const uint8_t *const_vals = p;
            p += const_count; p_left -= const_count;

            uint32_t ci = 0;
            for (uint32_t i = 0; i < logical_count; ++i) {
                if (modes[i] == TFC_MODE_CONST) {
                    if (ci < const_count) {
                        intercept[i] = (float)const_vals[ci++];
                    }
                }
            }

            /* Read LINEAR stream */
            size_t linear_bytes = linear_count + linear_count * 2;
            if (p_left < linear_bytes) {
                free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                free(decompressed);
                for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                return TFC_ERR_CORRUPT_STREAM;
            }
            const uint8_t *lin_a = p;
            const uint8_t *lin_b = p + linear_count;
            p += linear_bytes; p_left -= linear_bytes;

            uint32_t li = 0;
            for (uint32_t i = 0; i < logical_count; ++i) {
                if (modes[i] == TFC_MODE_LINEAR) {
                    if (li < linear_count) {
                        intercept[i] = (float)lin_a[li];
                        int16_t b_q = read_i16_le(lin_b + li * 2);
                        slope[i] = (float)b_q * b_scale;
                        li++;
                    }
                }
            }

            /* Read RAW stream */
            int16_t *raw_residuals = NULL;
            if (raw_count > 0) {
                size_t raw_bytes = raw_count + raw_count * 2 + (size_t)raw_count * seg_len * 2;
                if (p_left < raw_bytes) {
                    free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                    free(decompressed);
                    for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                    return TFC_ERR_CORRUPT_STREAM;
                }
                const uint8_t *r_a = p;
                const uint8_t *r_b = p + raw_count;
                const uint8_t *r_q = p + raw_count * 3;
                p += raw_bytes; p_left -= raw_bytes;

                raw_residuals = (int16_t *)malloc(raw_count * seg_len * sizeof(int16_t));
                if (!raw_residuals) {
                    free(modes); free(intercept); free(slope); free(is_raw); free(raw_idx_map);
                    free(decompressed);
                    for (int k = 0; k < 3; ++k) free(plane_bufs[k]);
                    return TFC_ERR_OUT_OF_MEMORY;
                }
                for (size_t k = 0; k < (size_t)raw_count * seg_len; ++k) {
                    raw_residuals[k] = read_i16_le(r_q + k * 2);
                }

                uint32_t ri = 0;
                for (uint32_t i = 0; i < logical_count; ++i) {
                    if (modes[i] == TFC_MODE_RAW) {
                        if (ri < raw_count) {
                            intercept[i] = (float)r_a[ri];
                            int16_t b_q = read_i16_le(r_b + ri * 2);
                            slope[i] = (float)b_q * b_scale;
                            is_raw[i] = 1;
                            raw_idx_map[i] = (int)ri;
                            ri++;
                        }
                    }
                }
            }

            /* Synthesize samples across time for this segment */
            for (uint32_t t = 0; t < seg_len && (frame_cursor + t) < T; ++t) {
                uint32_t f_idx = frame_cursor + t;
                uint8_t *frame_ptr = frame_base + (size_t)f_idx * frame_stride;

                if (tile_size == 1) {
#if TFC_HAS_NEON
                    if (pixel_stride == 1 && raw_count == 0) {
                        uint32_t i = 0;
                        float32x4_t vt = vdupq_n_f32((float)t);
                        for (; i + 8 <= logical_count; i += 8) {
                            float32x4_t vi0 = vld1q_f32(intercept + i);
                            float32x4_t vs0 = vld1q_f32(slope + i);
                            float32x4_t vf0 = vmlaq_f32(vi0, vs0, vt);
                            int32x4_t vi32_0 = vcvtnq_s32_f32(vf0);
                            int16x4_t vi16_0 = vqmovn_s32(vi32_0);

                            float32x4_t vi1 = vld1q_f32(intercept + i + 4);
                            float32x4_t vs1 = vld1q_f32(slope + i + 4);
                            float32x4_t vf1 = vmlaq_f32(vi1, vs1, vt);
                            int32x4_t vi32_1 = vcvtnq_s32_f32(vf1);
                            int16x4_t vi16_1 = vqmovn_s32(vi32_1);

                            uint8x8_t vu8 = vqmovun_s16(vcombine_s16(vi16_0, vi16_1));
                            vst1_u8(frame_ptr + i, vu8);
                        }
                        for (; i < logical_count; ++i) {
                            frame_ptr[i] = clamp_u8(intercept[i] + slope[i] * (float)t);
                        }
                    } else if (pixel_stride == 3 && raw_count == 0) {
                        uint32_t i = 0;
                        float32x4_t vt = vdupq_n_f32((float)t);
                        for (; i + 8 <= logical_count; i += 8) {
                            float32x4_t vi0 = vld1q_f32(intercept + i);
                            float32x4_t vs0 = vld1q_f32(slope + i);
                            float32x4_t vf0 = vmlaq_f32(vi0, vs0, vt);
                            int32x4_t vi32_0 = vcvtnq_s32_f32(vf0);
                            int16x4_t vi16_0 = vqmovn_s32(vi32_0);

                            float32x4_t vi1 = vld1q_f32(intercept + i + 4);
                            float32x4_t vs1 = vld1q_f32(slope + i + 4);
                            float32x4_t vf1 = vmlaq_f32(vi1, vs1, vt);
                            int32x4_t vi32_1 = vcvtnq_s32_f32(vf1);
                            int16x4_t vi16_1 = vqmovn_s32(vi32_1);

                            uint8x8_t vu8 = vqmovun_s16(vcombine_s16(vi16_0, vi16_1));
                            uint8_t bytes[8];
                            vst1_u8(bytes, vu8);

                            uint8_t *p_out = frame_ptr + (size_t)i * 3;
                            p_out[0]  = bytes[0];
                            p_out[3]  = bytes[1];
                            p_out[6]  = bytes[2];
                            p_out[9]  = bytes[3];
                            p_out[12] = bytes[4];
                            p_out[15] = bytes[5];
                            p_out[18] = bytes[6];
                            p_out[21] = bytes[7];
                        }
                        for (; i < logical_count; ++i) {
                            frame_ptr[(size_t)i * 3] = clamp_u8(intercept[i] + slope[i] * (float)t);
                        }
                    } else
#endif
                    {
                        for (uint32_t i = 0; i < logical_count; ++i) {
                            float v = intercept[i] + slope[i] * (float)t;
                            if (is_raw[i] && raw_residuals) {
                                v += (float)raw_residuals[raw_idx_map[i] * seg_len + t] * raw_scale;
                            }
                            frame_ptr[(size_t)i * pixel_stride] = clamp_u8(v);
                        }
                    }
                } else {
                    for (uint32_t ty = 0; ty < tiles_y; ++ty) {
                        uint32_t y_start = ty * tile_size;
                        uint32_t y_end = (y_start + tile_size < cur_H) ? (y_start + tile_size) : cur_H;

                        for (uint32_t tx = 0; tx < tiles_x; ++tx) {
                            uint32_t idx = ty * tiles_x + tx;
                            float v = intercept[idx] + slope[idx] * (float)t;
                            if (is_raw[idx] && raw_residuals) {
                                v += (float)raw_residuals[raw_idx_map[idx] * seg_len + t] * raw_scale;
                            }
                            uint8_t byte_val = clamp_u8(v);

                            uint32_t x_start = tx * tile_size;
                            uint32_t x_end = (x_start + tile_size < cur_W) ? (x_start + tile_size) : cur_W;
                            uint32_t span = x_end - x_start;

                            if (pixel_stride == 1) {
                                for (uint32_t y = y_start; y < y_end; ++y) {
                                    uint8_t *row_ptr = frame_ptr + (size_t)y * row_stride + x_start;
                                    memset(row_ptr, byte_val, span);
                                }
                            } else {
                                for (uint32_t y = y_start; y < y_end; ++y) {
                                    uint8_t *row_ptr = frame_ptr + (size_t)y * row_stride;
                                    for (uint32_t x = x_start; x < x_end; ++x) {
                                        row_ptr[(size_t)x * pixel_stride] = byte_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            frame_cursor += seg_len;
            if (raw_residuals) free(raw_residuals);
        }

        free(modes);
        free(intercept);
        free(slope);
        free(is_raw);
        free(raw_idx_map);
        free(decompressed);
    }

    /* Convert to RGB if needed */
    if (hdr.color_format == TFC_COLOR_YUV444) {
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t y = 0; y < H; ++y) {
                for (uint32_t x = 0; x < W; ++x) {
                    size_t idx = (size_t)t * H * W + y * W + x;
                    float y_val = (float)plane_bufs[0][idx];
                    float u_val = (float)plane_bufs[1][idx];
                    float v_val = (float)plane_bufs[2][idx];
                    yuv_to_rgb(y_val, u_val, v_val, out_rgb24 + idx * 3);
                }
            }
        }
    } else if (hdr.color_format == TFC_COLOR_YUV420) {
        uint32_t cw = plane_w[1];
        uint32_t ch = plane_h[1];
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t y = 0; y < H; ++y) {
                uint32_t yc = (y / 2 < ch) ? (y / 2) : (ch - 1);
                for (uint32_t x = 0; x < W; ++x) {
                    uint32_t xc = (x / 2 < cw) ? (x / 2) : (cw - 1);
                    size_t idx_y = (size_t)t * H * W + y * W + x;
                    size_t idx_c = (size_t)t * ch * cw + yc * cw + xc;
                    float y_val = (float)plane_bufs[0][idx_y];
                    float u_val = (float)plane_bufs[1][idx_c];
                    float v_val = (float)plane_bufs[2][idx_c];
                    yuv_to_rgb(y_val, u_val, v_val, out_rgb24 + idx_y * 3);
                }
            }
        }
    }

    for (int k = 0; k < 3; ++k) {
        if (plane_bufs[k]) free(plane_bufs[k]);
    }

    return TFC_OK;
}

tfc_status_t tfc_decode_file(
    const char *filepath,
    uint8_t **out_rgb24,
    size_t *out_size,
    tfc_header_t *out_header
) {
    if (!filepath || !out_rgb24 || !out_size || !out_header) {
        return TFC_ERR_IO;
    }

    FILE *f = fopen(filepath, "rb");
    if (!f) return TFC_ERR_IO;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 8) {
        fclose(f);
        return TFC_ERR_CORRUPT_HEADER;
    }

    uint8_t magic[4];
    if (fread(magic, 1, 4, f) != 4) {
        fclose(f);
        return TFC_ERR_IO;
    }
    fseek(f, 0, SEEK_SET);

    /* Case 1: TFC1 (v3/v4 standalone container) */
    if (memcmp(magic, "TFC1", 4) == 0) {
        uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
        if (!buf) { fclose(f); return TFC_ERR_OUT_OF_MEMORY; }
        if (fread(buf, 1, (size_t)fsize, f) != (size_t)fsize) {
            free(buf); fclose(f); return TFC_ERR_IO;
        }
        fclose(f);

        tfc_status_t status = tfc_read_header(buf, (size_t)fsize, out_header);
        if (status != TFC_OK) { free(buf); return status; }

        size_t total_bytes = (size_t)out_header->width * out_header->height * out_header->num_frames * 3;
        uint8_t *rgb = (uint8_t *)malloc(total_bytes);
        if (!rgb) { free(buf); return TFC_ERR_OUT_OF_MEMORY; }

        status = tfc_decode_v3_v4_memory(buf, (size_t)fsize, rgb, total_bytes, out_header);
        free(buf);
        if (status != TFC_OK) { free(rgb); return status; }

        *out_rgb24 = rgb;
        *out_size = total_bytes;
        return TFC_OK;
    }

    /* Case 2: TFC5 (multi-chunk streaming container) */
    if (memcmp(magic, "TFC5", 4) == 0) {
        uint32_t meta_len = 0;
        fseek(f, 4, SEEK_SET);
        if (fread(&meta_len, 4, 1, f) != 1) { fclose(f); return TFC_ERR_IO; }

        char *json = (char *)malloc(meta_len + 1);
        if (!json) { fclose(f); return TFC_ERR_OUT_OF_MEMORY; }
        if (fread(json, 1, meta_len, f) != meta_len) {
            free(json); fclose(f); return TFC_ERR_IO;
        }
        json[meta_len] = '\0';

        tfc_header_t hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.version = 5;
        parse_json_uint(json, "width", &hdr.width);
        parse_json_uint(json, "height", &hdr.height);
        parse_json_uint(json, "num_frames", &hdr.num_frames);
        hdr.fps_num = 30; hdr.fps_den = 1;
        parse_json_uint(json, "fps_num", &hdr.fps_num);
        parse_json_uint(json, "fps_den", &hdr.fps_den);
        hdr.color_format = TFC_COLOR_RGB;
        hdr.tiling = 1;
        hdr.segment_count = 1;
        free(json);

        *out_header = hdr;
        size_t total_bytes = (size_t)hdr.width * hdr.height * hdr.num_frames * 3;
        uint8_t *rgb = (uint8_t *)malloc(total_bytes);
        if (!rgb) { fclose(f); return TFC_ERR_OUT_OF_MEMORY; }

        uint32_t completed_frames = 0;

        /* Read records */
        while (completed_frames < hdr.num_frames) {
            uint8_t rec_hdr[16];
            if (fread(rec_hdr, 1, 16, f) != 16) {
                free(rgb); fclose(f); return TFC_ERR_CORRUPT_STREAM;
            }
            char tag[5] = {0};
            memcpy(tag, rec_hdr, 4);
            uint64_t rec_size = read_u64_le(rec_hdr + 4);
            uint32_t rec_crc = read_u32_le(rec_hdr + 12);

            if (strcmp(tag, "DONE") == 0) {
                break;
            }
            if (strcmp(tag, "AUDI") == 0) {
                fseek(f, (long)rec_size, SEEK_CUR);
                continue;
            }
            if (strcmp(tag, "VIDS") == 0) {
                uint8_t *chunk_buf = (uint8_t *)malloc((size_t)rec_size);
                if (!chunk_buf) { free(rgb); fclose(f); return TFC_ERR_OUT_OF_MEMORY; }
                if (fread(chunk_buf, 1, (size_t)rec_size, f) != (size_t)rec_size) {
                    free(chunk_buf); free(rgb); fclose(f); return TFC_ERR_IO;
                }
#if TFC_HAS_ZLIB
                if (rec_crc != 0) {
                    uLong computed_crc = crc32(0L, Z_NULL, 0);
                    computed_crc = crc32(computed_crc, chunk_buf, (uInt)rec_size);
                    if ((uint32_t)computed_crc != rec_crc) {
                        free(chunk_buf); free(rgb); fclose(f); return TFC_ERR_CHECKSUM_MISMATCH;
                    }
                }
#endif
                tfc_header_t chunk_hdr;
                size_t offset_bytes = (size_t)completed_frames * hdr.height * hdr.width * 3;
                size_t avail_bytes = total_bytes - offset_bytes;

                tfc_status_t status = tfc_decode_v3_v4_memory(
                    chunk_buf, (size_t)rec_size,
                    rgb + offset_bytes, avail_bytes,
                    &chunk_hdr
                );
                free(chunk_buf);
                if (status != TFC_OK) {
                    free(rgb); fclose(f); return status;
                }
                completed_frames += chunk_hdr.num_frames;
            } else {
                /* Unknown tag */
                free(rgb); fclose(f); return TFC_ERR_CORRUPT_STREAM;
            }
        }

        fclose(f);
        *out_rgb24 = rgb;
        *out_size = total_bytes;
        return TFC_OK;
    }

    fclose(f);
    return TFC_ERR_INVALID_MAGIC;
}
