#include "tfc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

static double get_time_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
#endif
}

static void print_usage(const char *prog) {
    fprintf(stderr, "TimeFlowCodec ANSI C Reference Decoder (libtfc)\n");
    fprintf(stderr, "Usage: %s <input.tfc> [output.raw | output.ppm | -] [--bench]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  output.raw : Dumps contiguous raw RGB24 frames to file\n");
    fprintf(stderr, "  output.ppm : Writes first frame as PPM image\n");
    fprintf(stderr, "  -          : Pipes raw RGB24 frames to stdout (e.g. into ffplay/mpv)\n");
    fprintf(stderr, "  --bench    : Measures wall-clock decoding throughput and FPS\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = NULL;
    int bench_mode = 0;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = 1;
        } else if (!output_path) {
            output_path = argv[i];
        }
    }

    uint8_t *rgb_data = NULL;
    size_t rgb_size = 0;
    tfc_header_t hdr;

    int iterations = bench_mode ? 10 : 1;
    double total_elapsed = 0.0;
    tfc_status_t status = TFC_OK;

    for (int it = 0; it < iterations; ++it) {
        if (rgb_data) {
            tfc_free(rgb_data);
            rgb_data = NULL;
        }
        double t0 = get_time_sec();
        status = tfc_decode_file(input_path, &rgb_data, &rgb_size, &hdr);
        double t1 = get_time_sec();
        if (status != TFC_OK) break;
        total_elapsed += (t1 - t0);
    }

    if (status != TFC_OK) {
        fprintf(stderr, "Error decoding '%s': %s (%d)\n", input_path, tfc_status_to_string(status), status);
        return 1;
    }

    double avg_elapsed = total_elapsed / iterations;
    double fps = avg_elapsed > 0.0 ? (double)hdr.num_frames / avg_elapsed : 0.0;
    double mb_per_sec = avg_elapsed > 0.0 ? ((double)rgb_size / (1024.0 * 1024.0)) / avg_elapsed : 0.0;

    fprintf(stderr, "Decoded %s (v%u): %ux%u, %u frames (%u/%u fps)\n",
            input_path, hdr.version, hdr.width, hdr.height, hdr.num_frames, hdr.fps_num, hdr.fps_den);
    if (bench_mode) {
        fprintf(stderr, "[BENCHMARK 10-runs] Avg Time: %.4f s | Throughput: %.2f FPS (%.2f MB/s RGB24)\n",
                avg_elapsed, fps, mb_per_sec);
    } else {
        fprintf(stderr, "Elapsed: %.4f s | Throughput: %.2f FPS (%.2f MB/s RGB24)\n",
                avg_elapsed, fps, mb_per_sec);
    }

    if (output_path) {
        if (strcmp(output_path, "-") == 0) {
            fwrite(rgb_data, 1, rgb_size, stdout);
        } else if (strstr(output_path, ".ppm")) {
            /* Write first frame as PPM */
            FILE *f = fopen(output_path, "wb");
            if (f) {
                fprintf(f, "P6\n%u %u\n255\n", hdr.width, hdr.height);
                fwrite(rgb_data, 1, (size_t)hdr.width * hdr.height * 3, f);
                fclose(f);
                fprintf(stderr, "Saved first frame PPM to %s\n", output_path);
            }
        } else {
            FILE *f = fopen(output_path, "wb");
            if (f) {
                fwrite(rgb_data, 1, rgb_size, f);
                fclose(f);
                fprintf(stderr, "Saved %zu raw RGB bytes to %s\n", rgb_size, output_path);
            }
        }
    }

    tfc_free(rgb_data);
    return 0;
}
