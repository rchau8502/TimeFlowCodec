#ifndef TFC_H
#define TFC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Color format constants */
#define TFC_COLOR_RGB    1
#define TFC_COLOR_YUV444 2
#define TFC_COLOR_YUV420 3

/* Mode constants */
#define TFC_MODE_LINEAR 0
#define TFC_MODE_CONST  1
#define TFC_MODE_RAW    2
#define TFC_MODE_MATRIX 3

/* Compression constants */
#define TFC_COMP_NONE 0
#define TFC_COMP_ZLIB 1
#define TFC_COMP_LZMA 2
#define TFC_COMP_ZSTD 3

/* Status / Return codes */
typedef enum {
    TFC_OK = 0,
    TFC_ERR_INVALID_MAGIC = -1,
    TFC_ERR_UNSUPPORTED_VERSION = -2,
    TFC_ERR_CORRUPT_HEADER = -3,
    TFC_ERR_UNSUPPORTED_COMPRESSION = -4,
    TFC_ERR_CORRUPT_STREAM = -5,
    TFC_ERR_OUT_OF_MEMORY = -6,
    TFC_ERR_IO = -7,
    TFC_ERR_BUFFER_TOO_SMALL = -8,
    TFC_ERR_CHECKSUM_MISMATCH = -9
} tfc_status_t;

/* Header information */
typedef struct {
    uint16_t version;
    uint32_t width;
    uint32_t height;
    uint32_t num_frames;
    uint8_t  color_format;
    uint8_t  payload_comp_type;
    uint16_t tiling;
    uint16_t segment_count;
    uint32_t fps_num;
    uint32_t fps_den;
} tfc_header_t;

/**
 * @brief Parse container header information from a memory buffer.
 *
 * Supports both TFC1 (v3/v4) and TFC5 container formats.
 */
tfc_status_t tfc_read_header(const uint8_t *buffer, size_t size, tfc_header_t *header);

/**
 * @brief Decode a single v3/v4 container from a memory buffer to contiguous RGB24 frames.
 *
 * @param in_buffer Pointer to raw v3/v4 container bytes.
 * @param in_size Size of in_buffer in bytes.
 * @param out_rgb24 Destination buffer of at least (width * height * num_frames * 3) bytes.
 * @param out_capacity Size of out_rgb24 in bytes.
 * @param out_header Optional pointer to receive header metadata.
 * @return TFC_OK on success, or negative error code.
 */
tfc_status_t tfc_decode_v3_v4_memory(
    const uint8_t *in_buffer,
    size_t in_size,
    uint8_t *out_rgb24,
    size_t out_capacity,
    tfc_header_t *out_header
);

/**
 * @brief Decode an entire .tfc file (supports TFC1 v3/v4 and TFC5 multi-chunk containers).
 *
 * Allocates the RGB24 output buffer, which must be freed using tfc_free().
 *
 * @param filepath Path to the .tfc file.
 * @param out_rgb24 Output pointer that will receive the allocated RGB24 frames.
 * @param out_size Output pointer that will receive total byte size of out_rgb24.
 * @param out_header Output pointer to receive header metadata.
 * @return TFC_OK on success, or negative error code.
 */
tfc_status_t tfc_decode_file(
    const char *filepath,
    uint8_t **out_rgb24,
    size_t *out_size,
    tfc_header_t *out_header
);

/**
 * @brief Free memory allocated by tfc_decode_file.
 */
void tfc_free(void *ptr);

/**
 * @brief Get human-readable description of a tfc_status_t error code.
 */
const char *tfc_status_to_string(tfc_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* TFC_H */
