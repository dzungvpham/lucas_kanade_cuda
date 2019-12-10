/**
 * (c) 2019 Dzung Pham
 *
 * This program contains the CUDA C code for computing optical flow using
 * the single-pass Lucas-Kanade method
 *
 * To run, type:
 * lucas_kanade <WINDOW_SIZE> <BLOCK_SIZE> <PATH_TO_FRAME_1> <PATH_TO_FRAME_2> <PATH_TO_FLOW_VISUALIZATION>
 *
 * Example:
 * lucas_kanade 25 32 input/sphere_big_odd/frame_0.jpg input/sphere_big_odd/frame_1.jpg out.jpg
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <math.h>
#include <math_constants.h> // CUDA Math constants
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "jpeglib.h"

// ----------------------- JPEG Helper code ------------------------

#define JPEG_OUTPUT_QUALITY 75

/*
 * IMAGE DATA FORMATS:
 *
 * The standard input image format is a rectangular array of pixels, with
 * each pixel having the same number of "component" values (color channels).
 * Each pixel row is an array of JSAMPLEs (which typically are unsigned chars).
 * If you are working with color data, then the color values for each pixel
 * must be adjacent in the row; for example, R,G,B,R,G,B,R,G,B,... for 24-bit
 * RGB color.
 */

/* The "frame structure" structure contains an image frame (in RGB or grayscale
 * formats) for passing around the CS338 projects.
 */
typedef struct frame_struct {
    JSAMPLE *image_buffer;	/* Points to large array of R,G,B-order/grayscale data
                             * Access directly with:
                             *   image_buffer[num_channels*pixel + component]
                             */
    JSAMPLE **row_pointers;	/* Points to an array of pointers to the beginning
                             * of each row in the image buffer.  Use to access
                             * the image buffer in a row-wise fashion, with:
                             *   row_pointers[row][num_channels*pixel + component]
                             */
    int image_height;		/* Number of rows in image */
    int image_width;		/* Number of columns in image */
    int num_channels;	/* Number of components (usually RGB=3 or gray=1) */
} frame_struct_t;
typedef frame_struct_t *frame_ptr;

/* Read/write JPEGs, for program startup & shutdown */
void write_JPEG_file (const char * filename, frame_ptr p_info, int quality);
frame_ptr read_JPEG_file (char * filename);

/* Allocate/deallocate frame buffers */
frame_ptr allocate_frame(int height, int width, int num_channels);
void destroy_frame(frame_ptr kill_me);

/*
 * write_JPEG_file writes out the contents of an image buffer to a JPEG.
 * A quality level of 2-100 can be provided (default = 75, high quality = ~95,
 * low quality = ~25, utter pixellation = 2).  Note that unlike read_JPEG_file,
 * it does not do any memory allocation on the buffer passed to it.
 */
void write_JPEG_file (const char * filename, frame_ptr p_info, int quality) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE * outfile;		/* target file */

    /* Step 1: allocate and initialize JPEG compression object */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    /* Step 2: specify data destination (eg, a file) */
    /* Note: steps 2 and 3 can be done in either order. */

    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "ERROR: Can't open output file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    jpeg_stdio_dest(&cinfo, outfile);

    /* Step 3: set parameters for compression */

    /* Set basic picture parameters (not optional) */
    cinfo.image_width = p_info->image_width; 	/* image width and height, in pixels */
    cinfo.image_height = p_info->image_height;
    cinfo.input_components = p_info->num_channels; /* # of color components per pixel */
    if (p_info->num_channels == 3)
        cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */
    else if (p_info->num_channels == 1)
        cinfo.in_color_space = JCS_GRAYSCALE;
    else {
        fprintf(stderr, "ERROR: Non-standard colorspace for compressing!\n");
        exit(EXIT_FAILURE);
    }
    /* Fill in the defaults for everything else, then override quality */
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);

    /* Step 4: Start compressor */
    jpeg_start_compress(&cinfo, TRUE);

    /* Step 5: while (scan lines remain to be written) */
    /*           jpeg_write_scanlines(...); */
    while (cinfo.next_scanline < cinfo.image_height) {
    (void) jpeg_write_scanlines(&cinfo, &(p_info->row_pointers[cinfo.next_scanline]), 1);
    }

    /* Step 6: Finish compression & close output */

    jpeg_finish_compress(&cinfo);
    fclose(outfile);

    /* Step 7: release JPEG compression object */
    jpeg_destroy_compress(&cinfo);
}

/*
 * read_JPEG_file reads the contents of a JPEG into an image buffer, which
 * is automatically allocated after the size of the image is determined.
 * We want to return a frame struct on success, NULL on error.
 */

frame_ptr read_JPEG_file (char * filename) {
    /* This struct contains the JPEG decompression parameters and pointers to
    * working space (which is allocated as needed by the JPEG library).
    */
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE * infile;		/* source file */
    frame_ptr p_info;		/* Output frame information */

    //  JSAMPLE *realBuffer;
    //  JSAMPLE **buffer;		/* Output row buffer */
    //  int row_stride;		/* physical row width in output buffer */

    /* Step 1: allocate and initialize JPEG decompression object */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    /* Step 2: open & specify data source (eg, a file) */
    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "ERROR: Can't open input file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    jpeg_stdio_src(&cinfo, infile);

    /* Step 3: read file parameters with jpeg_read_header() */
    (void) jpeg_read_header(&cinfo, TRUE);

    /* Step 4: use default parameters for decompression */

    /* Step 5: Start decompressor */
    (void) jpeg_start_decompress(&cinfo);

    /* Step X: Create a frame struct & buffers and fill in the blanks */
    fprintf(stderr, "  Opened %s: height = %d, width = %d, c = %d\n",
        filename, cinfo.output_height, cinfo.output_width, cinfo.output_components);
    p_info = allocate_frame(cinfo.output_height, cinfo.output_width, cinfo.output_components);

    /* Step 6: while (scan lines remain to be read) */
    /*           jpeg_read_scanlines(...); */
    while (cinfo.output_scanline < cinfo.output_height) {
        (void) jpeg_read_scanlines(&cinfo, &(p_info->row_pointers[cinfo.output_scanline]), 1);
    }

    /* Step 7: Finish decompression */
    (void) jpeg_finish_decompress(&cinfo);

    /* Step 8: Release JPEG decompression object & file */
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    /* At this point you may want to check to see whether any corrupt-data
    * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
    */

    /* And we're done! */
    return p_info;
}

/*
 * allocate/destroy_frame allocate a frame_struct_t and fill in the
 *  blanks appropriately (including allocating the actual frames), and
 *  then destroy them afterwards.
 */
frame_ptr allocate_frame(int height, int width, int num_channels) {
    int row_stride;		/* physical row width in output buffer */
    int i;
    frame_ptr p_info;		/* Output frame information */

    /* JSAMPLEs per row in output buffer */
    row_stride = width * num_channels;

    /* Basic struct and information */
    if ((p_info = (frame_struct_t*) malloc(sizeof(frame_struct_t))) == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failure\n");
        exit(EXIT_FAILURE);
    }
    p_info->image_height = height;
    p_info->image_width = width;
    p_info->num_channels = num_channels;

    /* Image array and pointers to rows */
    if ((p_info->row_pointers = (JSAMPLE**) malloc(sizeof(JSAMPLE *) * height)) == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failure\n");
        exit(EXIT_FAILURE);
    }
    if ((p_info->image_buffer = (JSAMPLE*) calloc(1, sizeof(JSAMPLE) * row_stride * height)) == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failure\n");
        exit(EXIT_FAILURE);
    }
    for (i=0; i < height; i++)
    	p_info->row_pointers[i] = & (p_info->image_buffer[i * row_stride]);

    /* And send it back! */
    return p_info;
}

void destroy_frame(frame_ptr kill_me) {
    free(kill_me->image_buffer);
    free(kill_me->row_pointers);
    free(kill_me);
}

// --------------------- Project Code -----------------------

// Force function inlining
#ifdef _MSC_VER
    #define forceinline __forceinline
#elif defined(__GNUC__)
    #define forceinline inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
    #if __has_attribute(__always_inline__)
        #define forceinline inline __attribute__((__always_inline__))
    #else
        #define forceinline inline
    #endif
#else
    #define forceinline inline
#endif

// Unit constants
#define BYTE_PER_KB 1024
#define BYTE_PER_MB 1048576

// Constants for converting RGB to grayscale
#define R_GRAYSCALE 0.2126
#define G_GRAYSCALE 0.7152
#define B_GRAYSCALE 0.0722

// Math constants
#define TWO_PI (CUDART_PI_F * 2)
#define EIGEN_THRESHOLD 0.02 // For checking the eigenvalue in the flow computation

// Run configuration constants
#define NUM_RUN 1 // Number of times to run the different GPU kernels for nvprof
#define RUN_TEST 0 // Whether or not to test against uniprocessor
#define MEASURE_CPU_TIME 0 // Whether or not to measure the cpu time
#define RUN_SIMPLE_GPU_KERNEL 0 // Whether or not to run the simple GPU kernel
#define RUN_HORIZ_TILED_GPU_KERNEL 1 // Whether or not to run the horizontal tiled GPU kernel
#define RUN_VERT_TILED_GPU_KERNEL 0 // Whether or not to run the vertical tiled GPU kernel
#define USE_PITCH 0 // Whether or not to run the kernel with pitch

/**
 * Makes sure the two input frames have the same dimensions
 */
void checkFrameDim(frame_ptr f1, frame_ptr f2) {
    if (
        f1->image_height != f2->image_height ||
        f1->image_width != f2->image_width ||
        f1->num_channels != f2->num_channels) {
        fprintf(stderr, "Dimensions do not match\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Makes sure values match in the two frames.
 * If there is a difference of 1, ignore because of rounding error.
 */
void checkResults(frame_ptr f1, frame_ptr f2) {
    checkFrameDim(f1, f2);
    int i, j, k;
    JSAMPLE j1, j2;

    for (i = 0; i < f1->image_height; i++){
        for (j = 0; j < f1->image_width; j++){
            for (k = 0; k < f1->num_channels; k++){
                j1 = f1->row_pointers[i][(f1->num_channels) * j + k];
                j2 = f2->row_pointers[i][(f2->num_channels) * j + k];
                if (abs(j1 - j2) > 1) { // Values between CPU and GPU can sometimes differ by 1
                    fprintf(stderr, "Values do not match at (%d, %d, %d) \n", i, j, k);
                    fprintf(stderr, "Frame 1: %d\n", j1);
                    fprintf(stderr, "Frame 2: %d\n", j2);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
}

/**
 * Queries the properties of the GPU device with the given device_id,
 * fills it in the given device_prop, and prints interesting info
 *
 * @param device_prop The cuda device properties struct
 * @param device_id The id of the gpu device
 */
void query_device(cudaDeviceProp *device_prop, int device_id) {
    cudaGetDeviceProperties(device_prop, device_id);
    printf(
        "******* GPU Device Properties *******\n"
        "Name: %s\n"
        "Compute capabability: %d.%d\n"
        "Clock rate: %dMHz | Memory clock rate: %dMHz\n"
        "Total constant memory: %zu KB | Total global memory: %zu MB\n"
        "Streaming processor (SM) count: %d\n"
        "Shared memory per SM: %zu B | per block: %zu B\n"
        "Registers per SM: %d | per block: %d\n"
        "Max threads per SM: %d | per block: %d\n"
        "Max grid size: %d %d %d\n"
        "Max block dimension: %d %d %d\n"
        "Warp size: %d\n"
        "*************************************\n",
        device_prop->name,
        device_prop->major, device_prop->minor,
        device_prop->clockRate / 1000, device_prop->memoryClockRate / 1000,
        device_prop->totalConstMem / BYTE_PER_KB, device_prop->totalGlobalMem / BYTE_PER_MB,
        device_prop->multiProcessorCount,
        device_prop->sharedMemPerMultiprocessor, device_prop->sharedMemPerBlock,
        device_prop->regsPerMultiprocessor, device_prop->regsPerBlock,
        device_prop->maxThreadsPerMultiProcessor, device_prop->maxThreadsPerBlock,
        device_prop->maxGridSize[0], device_prop->maxGridSize[1], device_prop->maxGridSize[2],
        device_prop->maxThreadsDim[0], device_prop->maxThreadsDim[1], device_prop->maxThreadsDim[2],
        device_prop->warpSize
    );
}

/**
 * Convert HSV to RGB. Saturation is assumed to be 1
 * @param h Hue [0, 360]
 * @param v Value [0, 1].
 * @return A float3 with R, G, B in [0, 1]
 */
inline __host__ __device__ float3 hsv2rgb(float h, float v) {
    h = clamp(h / 360.0f, 0.0f, 1.0f); // Make sure hue is in [0, 1]
    float R = abs(h * 6 - 3) - 1;
    float G = 2 - abs(h * 6 - 2);
    float B = 2 - abs(h * 6 - 4);
    return clamp(make_float3(R, G, B), 0.0f, 1.0f) * v;
}

/**
 * Convert RGB to grayscale (betwene 0 and 1)
 */
inline __host__ __device__ float rgb2gray(unsigned char r, unsigned char g, unsigned char b) {
    return round(R_GRAYSCALE * r + G_GRAYSCALE * g + B_GRAYSCALE * b) / 255.0f;
}

/**
 * Get angle in degrees [0, 360) from flow
 */
inline __host__ __device__ float get_angle(float2 flow) {
    return fmodf((atan2f(flow.y, flow.x) + TWO_PI), TWO_PI) * 180.0f / CUDART_PI_F;
}

/**
 * Get magnitude from flow
 */
inline __host__ __device__ float get_magnitude(float2 flow) {
    return sqrt(flow.x * flow.x + flow.y * flow.y);
}

/**
 * Divide then take ceiling
 */
inline __host__ __device__ int divide_up(int a, int b) {
    return (int) ceil(a / (float) b);
}

/**
 * Calculate flow using (A^T A)^{-1} A^T b.
 * @param AtA_00 The upper left entry of A^T A, corresponding to sum of (fx)^2
 * @param AtA_01 The upper right/bottom left entry of A^T A, corresponding to sum of fx * fy
 * @param AtA_11 The bottom right entry of A^T A, corresponding to sum of (fy)^2
 * @param Atb_0 The top entry of A^T b, corresponding to sum of fx * ft
 * @param Atb_1 The bottom entry of A^T b, corresponding to sum of fy * ft
 * @return A float2 containing the flow
 */
inline __host__ __device__ float2 calc_flow_from_matrix(
    float AtA_00, float AtA_01, float AtA_11, float Atb_0, float Atb_1
) {
     // Calculate determinant and make sure it's not too small in order for the matrix to be invertible
    float det = AtA_00 * AtA_11 - AtA_01 * AtA_01;
    if (abs(det) <= 1.5e-5) { // 1.5e-5 is based on 1/(255*255)
        return make_float2(0.0f, 0.0f);
    }

    // Calculate the smaller eigenvalue of A^T A and make sure it is > threshold
    float trace = AtA_00 + AtA_11; // Trace of A^T A
    float twice_delta = sqrtf(trace * trace - 4.0f * det); // Delta times 2
    if (isnan(twice_delta) || // Must check if delta is NA or not due to numerical issue with sqrt
        trace - twice_delta <= EIGEN_THRESHOLD) {  // comparing the smaller eigen value (multiplied by 2)
        return make_float2(0.0f, 0.0f);
    }

    // Calculate flow components
    return make_float2(AtA_11*Atb_0 - AtA_01*Atb_1, -AtA_01*Atb_0 + AtA_00*Atb_1) / det;
}

/**
 * Allocate a 2d array of dimension height x width, prefilled with 0
 */
float** alloc_2d_float_array(int height, int width) {
    float **ptr;

    if ((ptr = (float **) malloc(height * sizeof(float *))) == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failure\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < height; i++) {
        if ((ptr[i] = (float *) calloc(1, width * sizeof(float *))) == NULL) {
            fprintf(stderr, "ERROR: Memory allocation failure\n");
            exit(EXIT_FAILURE);
        }
    }
    return ptr;
}

/**
 * Allocate a 1d array of length size, prefilled with 0
 */
float* alloc_1d_float_array(int size) {
    float *ptr;
    if ((ptr = (float *) calloc(1, size * sizeof(float))) == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failure\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/**
 * Free a 2d array
 */
void free_2d_float_array(float **arr, int height) {
    for (int row = 0; row < height; row++) {
        free(arr[row]);
    }
    free(arr);
}

/**
 * Normalize RGB frame to be between 0 and 1, and grayscale if necessary
 */
float** get_normalized_2d_float_array(frame_ptr in) {
    int height = in->image_height, width = in->image_width;
    float **out = alloc_2d_float_array(height, width);
    float *cur_out_row;
    JSAMPLE *cur_in_row;
    int row, col;

    if (in->num_channels == 3) { // Grayscale
        for (row = 0; row < height; row++) {
            cur_in_row = in->row_pointers[row];
            cur_out_row = out[row];
            for (col = 0; col < width; col++) {
                cur_out_row[col] = rgb2gray(
                    cur_in_row[col * 3],
                    cur_in_row[col * 3 + 1],
                    cur_in_row[col * 3 + 2]
                );
            }
        }
    } else { // Already grayscaled
        for (row = 0; row < height; row++) {
            cur_in_row = in->row_pointers[row];
            cur_out_row = out[row];
            for (col = 0; col < width; col++) {
                cur_out_row[col] = cur_in_row[col] / 255.0f;
            }
        }
    }

    return out;
}

/**
 * Flatten a heigh x width 2d array into a 1d array
 */
float* flatten_2d_float_array(float **src, int height, int width) {
    float *arr;
    if ((arr = (float *) malloc(height * width * sizeof(float))) == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failure\n");
        exit(EXIT_FAILURE);
    }

    for (int row = 0; row < height; row++) {
        memcpy((void *) (arr + row * width), (void *) src[row], width * sizeof(float));
    }

    return arr;
}

/**
 * Pretty-print a 2d array for debugging purpose
 */
void print_2d_float_array(float **arr, int height, int width) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            printf("%.2f ", arr[row][col]);
        }
        printf("\n");
    }
}

/**
 * Pretty-print a 1d array as a 2d array for debugging purpose
 */
void print_1d_float_array_as_2d(float *arr, int height, int width) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            printf("%.2f ", arr[row * width + col]);
        }
        printf("\n");
    }
}

/**
 * Visualize the flow matrix using HSV and convert to RGB.
 *
 * @param out The output frame
 * @param angle The angle of the flows
 * @param mag The magnitude of the flows
 * @param height Height of frame
 * @param width Width of frame
 * @param s Floor of window size / 2
 */
void create_flow_visualization(
    frame_ptr out, float *angle, float *mag,
    int height, int width, int s
) {
    // Find min and max magnitude
    float cur_mag, max_mag = -INFINITY, min_mag = INFINITY;
    int row, col;
    for (row = s; row < height - s; row++) {
        for (col = s; col < width - s; col++) {
            cur_mag = mag[row * width + col];
            if (cur_mag > max_mag) {
                max_mag = cur_mag;
            }
            if (cur_mag < min_mag) {
                min_mag = cur_mag;
            }
        }
    }
    if (max_mag <= min_mag) {
        return;
    }

    // Convert angle & magnitude to RGB
    JSAMPLE *cur_row;
    max_mag -= min_mag;
    for (row = s; row < height - s; row++) {
        cur_row = out->row_pointers[row];

        for (col = s; col < width - s; col++) {
            float3 rgb = hsv2rgb(
                angle[row * width + col], // Angle corresponds to Hue
                clamp((mag[row * width + col] - min_mag) / max_mag, 0.0f, 1.0f) // Magnitude (scaled) corresponds to Value
            );
            cur_row[col * 3] = round(rgb.x * 255);
            cur_row[col * 3 + 1] = round(rgb.y * 255);
            cur_row[col * 3 + 2] = round(rgb.z * 255);
        }
    }
}

/**
 * Calculates the spatial derivative fx and fy and the temporal derivative ft
 * from two frames using the Prewitt operator (without dividng by 3)
 *
 * @param in1 Frame 1
 * @param in2 Frame 2
 * @param fx Horizontal spatial derivative
 * @param fy Vertical spatial derivative
 * @param ft Temporal derivative
 * @param height, width Height and width of frame
 */
void calculate_derivative(
    float **in1, float **in2,
    float *fx, float *fy, float *ft,
    int height, int width
) {
    int offset;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            offset = row * width + col;
            if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
                fx[offset] =
                    in1[row - 1][col + 1] + in1[row][col + 1] + in1[row + 1][col + 1] -
                    in1[row - 1][col - 1] - in1[row][col - 1] - in1[row + 1][col - 1];
                fy[offset] =
                    in1[row - 1][col - 1] + in1[row - 1][col] + in1[row - 1][col + 1] -
                    in1[row + 1][col - 1] - in1[row + 1][col] - in1[row + 1][col + 1];
            } else {
                fx[offset] = fy[offset] = 0.0f;
            }
            ft[offset] = in2[row][col] - in1[row][col];
        }
    }
}

/**
 * Calculate optical flow with Lucas-Kanade using CPU from 2 normalized frames
 *
 * @param fx, fy, ft The derivatives
 * @param out Output frame for visualization
 * @param height, width Height and width of input
 * @param window_size An odd positive integer >= 3 for the window size
 * @return Time needed to calculate flow only
 */
float uniprocessor_lucas_kanade(
    float *fx, float *fy, float *ft, frame_ptr out,
    int height, int width, int window_size
) {
    // Allocate the derivative matrices and the flow matrices (represented by angle and magnitude)
    float *angle = alloc_1d_float_array(height * width);
    float *mag = alloc_1d_float_array(height * width);
    float AtA_00, AtA_01, AtA_11, Atb_0, Atb_1;
    float2 flow;
    int s = window_size / 2;
    int row, col, i, j, offset;

    // Calculate flow
    clock_t start_clock = clock();
    for (row = s; row < height - s; row++) {
        for (col = s; col < width - s; col++) {
            AtA_00 = AtA_01 = AtA_11 = Atb_0 = Atb_1 = 0.0f;
            for (i = row - s; i <= row + s; i++) {
                for (j = col - s; j <= col + s; j++) {
                    offset = i * width + j;
                    AtA_00 += fx[offset] * fx[offset];
                    AtA_11 += fy[offset] * fy[offset];
                    AtA_01 += fx[offset] * fy[offset];
                    Atb_0 -= fx[offset] * ft[offset];
                    Atb_1 -= fy[offset] * ft[offset];
                }
            }

            // Calculate flow and convert to polar coordinates
            flow = calc_flow_from_matrix(AtA_00, AtA_01, AtA_11, Atb_0, Atb_1);
            angle[row * width + col] = get_angle(flow);
            mag[row * width + col] = get_magnitude(flow);
        }
    }
    clock_t end_clock = clock();
    float ms_elapsed = (end_clock - start_clock) * 1000 / CLOCKS_PER_SEC;

    // Create visualization and write to output frame
    create_flow_visualization(out, angle, mag, height, width, s);

    // Clean up
    free(angle);
    free(mag);

    return ms_elapsed;
}

// --------- CUDA Kernel code ------------

/**
 * A simple GPU kernel with no tiling
 */
__global__ void simple_lucas_kanade_kernel(
    float *fx, float *fy, float *ft, float *angle, float *mag,
    int height, int width, int pitch, int s
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y + s;
    int col = blockDim.x * blockIdx.x + threadIdx.x + s;
    if (row < height - s && col < width - s) {
        float AtA_00, AtA_01, AtA_11, Atb_0, Atb_1; // Entries of (A^T A)^-1 and A^T b
        float cur_fx, cur_fy, cur_ft;
        AtA_00 = AtA_01 = AtA_11 = Atb_0 = Atb_1 = 0.0f;

        for (int i = row - s; i <= row + s; i++) {
            for (int j = col - s; j <= col + s; j++) {
                cur_fx = fx[i * pitch + j];
                cur_fy = fy[i * pitch + j];
                cur_ft = ft[i * pitch + j];

                AtA_00 += cur_fx * cur_fx;
                AtA_11 += cur_fy * cur_fy;
                AtA_01 += cur_fx * cur_fy;
                Atb_0 -= cur_fx * cur_ft;
                Atb_1 -= cur_fy * cur_ft;
            }
        }

        // Calculate flow and convert to polar coordinates
        float2 flow = calc_flow_from_matrix(AtA_00, AtA_01, AtA_11, Atb_0, Atb_1);
        angle[row * pitch + col] = get_angle(flow);
        mag[row * pitch + col] = get_magnitude(flow);
    }
}

/**
 * The normal (horizontal) tiled GPU kernel
 */
__global__ void tiled_lucas_kanade_kernel(
    float *fx, float *fy, float *ft, float *angle, float *mag,
    int height, int width, int pitch, int s, int out_block_size, int in_block_size,
    int tile_height, int tile_width, int num_tile
) {
    // Get column for loading data
    int cur_col = out_block_size * blockIdx.x + threadIdx.x;
    if (cur_col >= width) {
        return;
    }

    // Load data into shared mem tile by tile
    extern __shared__ float shared_mem[]; // Contains fx, fy, ft
    int offset;
    int cur_block_row = threadIdx.y - tile_height;
    int cur_row = out_block_size * blockIdx.y - tile_height + threadIdx.y;
    for (int k = 0; k < num_tile; k++) {
        cur_row += tile_height;
        cur_block_row += tile_height;
        if (cur_row < height && cur_block_row < in_block_size) {
            offset = 3 * (cur_block_row * in_block_size + threadIdx.x);
            shared_mem[offset] = fx[cur_row * pitch + cur_col];
            shared_mem[offset + 1] = fy[cur_row * pitch + cur_col];
            shared_mem[offset + 2] = ft[cur_row * pitch + cur_col];
        }
    }

    __syncthreads(); // Wait for memory loading

    // Get rid of unnecessary threads based on column
    if (threadIdx.x < s || threadIdx.x >= in_block_size - s || cur_col >= width - s) {
        return;
    }

    // Start calculating flow tile by tile
    float AtA_00, AtA_01, AtA_11, Atb_0, Atb_1; // Entries of (A^T A)^-1 and A^T b
    float cur_fx, cur_fy, cur_ft;
    cur_block_row = threadIdx.y - tile_height; // reset row
    cur_row = out_block_size * blockIdx.y - tile_height + threadIdx.y; // reset row

    for (int k = 0; k < num_tile; k++) {
        cur_row += tile_height;
        cur_block_row += tile_height;

        if (cur_block_row >= s && cur_block_row < in_block_size - s && cur_row < height - s) {
            AtA_00 = AtA_01 = AtA_11 = Atb_0 = Atb_1 = 0.0f;
            for (int row = cur_block_row - s; row <= cur_block_row + s; row++) {
                for (int col = threadIdx.x - s; col <= threadIdx.x + s; col++) {
                    offset = 3 * (row * in_block_size + col);
                    cur_fx = shared_mem[offset];
                    cur_fy = shared_mem[offset + 1];
                    cur_ft = shared_mem[offset + 2];

                    AtA_00 += cur_fx * cur_fx;
                    AtA_11 += cur_fy * cur_fy;
                    AtA_01 += cur_fx * cur_fy;
                    Atb_0 -= cur_fx * cur_ft;
                    Atb_1 -= cur_fy * cur_ft;
                }
            }

            // Calculate flow and convert to polar coordinates
            float2 flow = calc_flow_from_matrix(AtA_00, AtA_01, AtA_11, Atb_0, Atb_1);
            angle[cur_row * pitch + cur_col] = get_angle(flow);
            mag[cur_row * pitch + cur_col] = get_magnitude(flow);
        }
    }
}

/**
 * The vertical tiled GPU kernel
 */
__global__ void vertical_tiled_lucas_kanade_kernel(
    float *fx, float *fy, float *ft, float *angle, float *mag,
    int height, int width, int pitch, int s, int out_block_size, int in_block_size,
    int tile_height, int tile_width, int num_tile
) {
    // Get row for loading data
    int cur_row = out_block_size * blockIdx.y + threadIdx.y;
    if (cur_row >= height) {
        return;
    }

    // Load data into shared mem tile by tile
    extern __shared__ float shared_mem[]; // Contains fx, fy, ft
    int offset;
    int cur_block_col = threadIdx.x - tile_width;
    int cur_col = out_block_size * blockIdx.x - tile_width + threadIdx.x;
    for (int k = 0; k < num_tile; k++) {
        cur_col += tile_width;
        cur_block_col += tile_width;
        if (cur_col < width && cur_block_col < in_block_size) {
            offset = 3 * (threadIdx.y * in_block_size + cur_block_col);
            shared_mem[offset] = fx[cur_row * pitch + cur_col];
            shared_mem[offset + 1] = fy[cur_row * pitch + cur_col];
            shared_mem[offset + 2] = ft[cur_row * pitch + cur_col];
        }
    }

    __syncthreads(); // Wait for memory loading

    // Get rid of unnecessary threads based on row
    if (threadIdx.y < s || threadIdx.y >= in_block_size - s || cur_row >= height - s) {
        return;
    }

    // Start calculating flow tile by tile
    float AtA_00, AtA_01, AtA_11, Atb_0, Atb_1; // Entries of (A^T A)^-1 and A^T b
    float cur_fx, cur_fy, cur_ft;
    cur_block_col = threadIdx.x - tile_width; // reset col
    cur_col = out_block_size * blockIdx.x - tile_width + threadIdx.x; // reset col

    for (int k = 0; k < num_tile; k++) {
        cur_col += tile_width;
        cur_block_col += tile_width;

        if (cur_block_col >= s && cur_block_col < in_block_size - s && cur_col < width - s) {
            AtA_00 = AtA_01 = AtA_11 = Atb_0 = Atb_1 = 0.0f;
            for (int row = threadIdx.y - s; row <= threadIdx.y + s; row++) {
                for (int col = cur_block_col - s; col <= cur_block_col + s; col++) {
                    offset = 3 * (row * in_block_size + col);
                    cur_fx = shared_mem[offset];
                    cur_fy = shared_mem[offset + 1];
                    cur_ft = shared_mem[offset + 2];

                    AtA_00 += cur_fx * cur_fx;
                    AtA_11 += cur_fy * cur_fy;
                    AtA_01 += cur_fx * cur_fy;
                    Atb_0 -= cur_fx * cur_ft;
                    Atb_1 -= cur_fy * cur_ft;
                }
            }

            // Calculate flow and convert to polar coordinates
            float2 flow = calc_flow_from_matrix(AtA_00, AtA_01, AtA_11, Atb_0, Atb_1);
            angle[cur_row * pitch + cur_col] = get_angle(flow);
            mag[cur_row * pitch + cur_col] = get_magnitude(flow);
        }
    }
}

// ----- Run kernel code -----

// Function type for the simple flow kernel
typedef void (*simple_flow_kernel_t)(
    float *, float *, float *, float *, float *,
    int, int, int, int
);

// Function type for the tiled flow kernel
typedef void (*tiled_flow_kernel_t)(
    float *, float *, float *, float *, float *,
    int, int, int, int, int, int, int, int, int
);

/**
 * This function manages the memory and grid/block configuration
 * for both simple and tiled GPU kernels, pitch or no pitch.
 * Then, it executes the appropriate kernel and creates visualization.
 *
 * @param fx, fy, ft The derivatives
 * @param out The output frame for visualization
 * @param grid_dim, block_dim The grid and block dimension for the kernel
 * @param simple_flow_kernel, tiled_flow_kernel The function pointer to the kernel to run
 * @param is_tiled Whether to run the simple kernel or the tiled kernel
 * @param use_pitch Whether or not to use pitch memory allocation
 * @param out_block_size, in_block_size Size of output and input block
 * @param shared_mem_size Total shared memory to allocate
 * @param tile_height, tile_width Dimensions of the tile
 * @param num_tile Number of tiles per block
 */
void run_generic_flow_kernel(
    float *fx, float *fy, float *ft, frame_ptr out,
    int height, int width, int window_size,
    dim3 grid_dim, dim3 block_dim,
    simple_flow_kernel_t simple_flow_kernel, tiled_flow_kernel_t tiled_flow_kernel,
    bool is_tiled, bool use_pitch,
    int out_block_size, int in_block_size, size_t shared_mem_size,
    int tile_height, int tile_width, int num_tile
) {
    // Allocate and set GPU memory for fx, fy, ft, angle and mag
    int s = window_size / 2;
    size_t pitch_bytes;
    int pitch_floats;
    size_t width_in_bytes = width * sizeof(float);
    size_t size = height * width_in_bytes;
    float *angle, *mag, *d_fx, *d_fy, *d_ft, *d_angle, *d_mag;
    angle = alloc_1d_float_array(size);
    mag = alloc_1d_float_array(size);

    if (use_pitch) {
        checkCudaErrors(cudaMallocPitch((void **) &d_fx, &pitch_bytes, width_in_bytes, height));
        checkCudaErrors(cudaMallocPitch((void **) &d_fy, &pitch_bytes, width_in_bytes, height));
        checkCudaErrors(cudaMallocPitch((void **) &d_ft, &pitch_bytes, width_in_bytes, height));
        checkCudaErrors(cudaMallocPitch((void **) &d_angle, &pitch_bytes, width_in_bytes, height));
        checkCudaErrors(cudaMallocPitch((void **) &d_mag, &pitch_bytes, width_in_bytes, height));

        checkCudaErrors(cudaMemcpy2D(
            d_fx, pitch_bytes, fx, width_in_bytes,
            width_in_bytes, height, cudaMemcpyHostToDevice
        ));
        checkCudaErrors(cudaMemcpy2D(
            d_fy, pitch_bytes, fy, width_in_bytes,
            width_in_bytes, height, cudaMemcpyHostToDevice
        ));
        checkCudaErrors(cudaMemcpy2D(
            d_ft, pitch_bytes, ft, width_in_bytes,
            width_in_bytes, height, cudaMemcpyHostToDevice
        ));
        checkCudaErrors(cudaMemset((void *) d_angle, 0, height * pitch_bytes));
        checkCudaErrors(cudaMemset((void *) d_mag, 0, height * pitch_bytes));

        pitch_floats = divide_up((int) pitch_bytes, (int) sizeof(float));

    } else {
        checkCudaErrors(cudaMalloc((void **) &d_fx, size));
        checkCudaErrors(cudaMalloc((void **) &d_fy, size));
        checkCudaErrors(cudaMalloc((void **) &d_ft, size));
        checkCudaErrors(cudaMalloc((void **) &d_angle, size));
        checkCudaErrors(cudaMalloc((void **) &d_mag, size));

        checkCudaErrors(cudaMemcpy(d_fx, fx, size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_fy, fy, size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_ft, ft, size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset((void *) d_angle, 0, size));
        checkCudaErrors(cudaMemset((void *) d_mag, 0, size));

        pitch_floats = width;
    }

    // Execute kernel
    if (is_tiled) {
        tiled_flow_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            d_fx, d_fy, d_ft, d_angle, d_mag, height, width, pitch_floats, s,
            out_block_size, in_block_size, tile_height, tile_width, num_tile
        );
    } else {
        simple_flow_kernel<<<grid_dim, block_dim>>>(
            d_fx, d_fy, d_ft, d_angle, d_mag, height, width, pitch_floats, s
        );
    }

    // Copy result to host's memory and make visualization
    if (use_pitch) {
        checkCudaErrors(cudaMemcpy2D(
            angle, width_in_bytes, d_angle, pitch_bytes,
            width_in_bytes, height, cudaMemcpyDeviceToHost
        ));
        checkCudaErrors(cudaMemcpy2D(
            mag, width_in_bytes, d_mag, pitch_bytes,
            width_in_bytes, height, cudaMemcpyDeviceToHost
        ));
    } else {
        checkCudaErrors(cudaMemcpy(angle, d_angle, size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(mag, d_mag, size, cudaMemcpyDeviceToHost));
    }

    create_flow_visualization(out, angle, mag, height, width, s);

    // Clean up
    free(angle);
    free(mag);
    checkCudaErrors(cudaFree(d_fx));
    checkCudaErrors(cudaFree(d_fy));
    checkCudaErrors(cudaFree(d_ft));
    checkCudaErrors(cudaFree(d_angle));
    checkCudaErrors(cudaFree(d_mag));
}

/**
 * This function sets up the grid/block dimensions for the simple flow kernel
 * then calls the generic kernel with the appropriate flow kernel
 *
 * @param fx, fy, ft The derivatives
 * @param out Output frame for visualization
 * @param height, width Dimension of the inputs
 * @param window_size Window size
 * @param block_size Block size for the kernel
 * @param use_pitch Whether or not to run the pitched kernel
 */
void run_simple_kernel(
    float *fx, float *fy, float *ft, frame_ptr out,
    int height, int width, int window_size, int block_size, bool use_pitch
) {
    // Configures grid and block dimensions
    int out_width = width - window_size + 1;
    int out_height = height - window_size + 1;
    dim3 block_dim(block_size, block_size, 1);
    dim3 grid_dim(divide_up(out_width, block_size), divide_up(out_height, block_size), 1);

    // Choose kernel to execute
    run_generic_flow_kernel(
        fx, fy, ft, out, height, width, window_size,
        grid_dim, block_dim,
        simple_lucas_kanade_kernel, NULL, false, use_pitch,
        0, 0, 0, 0, 0, 0
    );
}

/**
 * This function sets up the grid/block dimensions for the tiled flow kernel
 * then calls the generic tiled kernel with the appropriate flow kernel
 *
 * @param fx, fy, ft The derivatives
 * @param out Output frame for visualization
 * @param height, width Dimension of the inputs
 * @param window_size Window size
 * @param block_size Block size for the kernel
 * @param out_block_size Size of output block
 * @param max_threads_per_block Maxinum # of threads per block from the device properties
 * @param use_pitch Whether or not to run the pitched kernel
 * @param vertical Whether or not to run the vertical tile version
 */
void run_tiled_kernel(
    float *fx, float *fy, float *ft, frame_ptr out,
    int height, int width, int window_size,
    int out_block_size, int max_threads_per_block, bool use_pitch, bool vertical
) {
    // Configures tile, grid and block dimensions
    int in_block_size = out_block_size + window_size - 1;
    int smaller_tile_dim = min(in_block_size, max_threads_per_block / in_block_size); // The smaller dimension
    tiled_flow_kernel_t tiled_flow_kernel;
    int tile_height, tile_width;
    if (vertical) {
        tiled_flow_kernel = vertical_tiled_lucas_kanade_kernel;
        tile_height = in_block_size;
        tile_width = smaller_tile_dim;
    } else {
        tiled_flow_kernel = tiled_lucas_kanade_kernel;
        tile_height = smaller_tile_dim;
        tile_width = in_block_size;
    }
    dim3 block_dim(tile_width, tile_height, 1);
    int num_tile = divide_up(in_block_size, smaller_tile_dim);
    int out_width = width - window_size + 1;
    int out_height = height - window_size + 1;
    dim3 grid_dim(divide_up(out_width, out_block_size), divide_up(out_height, out_block_size), 1);
    size_t shared_mem_size = 3 * in_block_size * in_block_size * sizeof(float); // reserve shared mem for fx, fy, ft

    // Choose kernel to execute
    run_generic_flow_kernel(
        fx, fy, ft, out, height, width, window_size,
        grid_dim, block_dim,
        NULL, tiled_flow_kernel, true, use_pitch,
        out_block_size, in_block_size, shared_mem_size,
        tile_height, tile_width, num_tile
    );
}

/**
 * Calculate max window size for tiling
 * @param device_prop GPU device properties
 * @param out_block_size Size of the output block
 */
size_t calc_max_window_size(cudaDeviceProp *device_prop, int out_block_size) {
    // Maximum number of floats per shared mem array (fx, fy, ft)
    int max_float_num = device_prop->sharedMemPerBlock / sizeof(float) / 3;
    int max_window_size = ((int) floor(sqrt(max_float_num))) - out_block_size + 1;
    return max_window_size % 2 == 1 ? max_window_size : max_window_size - 1; // Make sure it's odd
}

// ----------- Host main --------------

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr,
            "Usage: lucas_kanade <WINDOW_SIZE> <BLOCK_SIZE>"
            " <PATH_TO_FRAME_1> <PATH_TO_FRAME_2> <PATH_TO_FLOW_VISUALIZATION>\n");
        exit(EXIT_FAILURE);
    }

    // Get max block size from GPU's props
    int device_id = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(device_id));
    cudaDeviceProp device_prop;
    query_device(&device_prop, device_id);
    int max_block_size = (int) floor(sqrt(device_prop.maxThreadsPerBlock));

    // Get inputs
    int window_size = atoi(argv[1]);
    if (window_size < 3 || window_size % 2 != 1) {
        fprintf(stderr, "Window size must be an odd integer >= 3\n");
        exit(EXIT_FAILURE);
    }

    // Check window size for tiling
    int max_window_size = calc_max_window_size(&device_prop, max_block_size);
    if (max_window_size < window_size) {
        fprintf(stderr, "Window size must be <= %d\n", max_window_size);
        exit(EXIT_FAILURE);
    }

    // Check block size
    int block_size = atoi(argv[2]);
    if (block_size > max_block_size) {
        fprintf(stderr, "Block size must be <= %d\n", max_block_size);
        exit(EXIT_FAILURE);
    }

    printf("Block size %d | Window size %d\n", block_size, window_size);

    // Check input frames
    frame_ptr raw_in1 = read_JPEG_file(argv[3]);
    frame_ptr raw_in2 = read_JPEG_file(argv[4]);
    checkFrameDim(raw_in1, raw_in2);
    if (raw_in1->num_channels != 1 && raw_in1->num_channels != 3) { // in1 and in2 has same # of channels
        fprintf(stderr, "Input frame must have either 1 or 3 channels\n");
        exit(EXIT_FAILURE);
    }

    // Create 2d array of normalized image pixel between [0, 1]
    int height = raw_in1->image_height;
    int width = raw_in2->image_width;
    float **in1 = get_normalized_2d_float_array(raw_in1);
    float **in2 = get_normalized_2d_float_array(raw_in2);

    // Calculate derivative
    float *fx = alloc_1d_float_array(height * width);
    float *fy = alloc_1d_float_array(height * width);
    float *ft = alloc_1d_float_array(height * width);
    calculate_derivative(in1, in2, fx, fy, ft, height, width);

    // Allocate output frames
    frame_ptr out_gpu = allocate_frame(height, width, 3);
    frame_ptr out_cpu = allocate_frame(height, width, 3);

    // Run CPU version
    if (MEASURE_CPU_TIME) {
        printf("Running CPU version... ");
        float total_cpu_ms = 0.0f;
        for (int i = 0; i < NUM_RUN; i++) {
            total_cpu_ms += uniprocessor_lucas_kanade(fx, fy, ft, out_cpu, height, width, window_size);
        }
        printf("Average time for CPU: %f ms\n", total_cpu_ms / (float) NUM_RUN);
        printf("Finished!\n");
    } else if (RUN_TEST) {
        printf("Running CPU version... ");
        uniprocessor_lucas_kanade(fx, fy, ft, out_cpu, height, width, window_size);
        printf("Finished!\n");
    }

    // Run GPU version several times for profiler while also testing against CPU version
    printf("Running GPU version... ");
    for (int i = 0; i < NUM_RUN; i++) {
        if (RUN_SIMPLE_GPU_KERNEL) {
            run_simple_kernel(
                fx, fy, ft, out_gpu,
                height, width, window_size, block_size, USE_PITCH
            );
            if (RUN_TEST) checkResults(out_gpu, out_cpu);
        }

        if (RUN_VERT_TILED_GPU_KERNEL) {
            run_tiled_kernel(
                fx, fy, ft, out_gpu,
                height, width, window_size, block_size,
                device_prop.maxThreadsPerBlock, USE_PITCH, true // Vertical tile
            );
            if (RUN_TEST) checkResults(out_gpu, out_cpu);
        }

        if (RUN_HORIZ_TILED_GPU_KERNEL) {
            run_tiled_kernel(
                fx, fy, ft, out_gpu,
                height, width, window_size, block_size,
                device_prop.maxThreadsPerBlock, USE_PITCH, false // Horizontal tile
            );
            if (RUN_TEST) checkResults(out_gpu, out_cpu);
        }

    }
    printf("Finished!\n");

    // Write out the visualization and clean up
    write_JPEG_file(argv[5], out_cpu, JPEG_OUTPUT_QUALITY);
    destroy_frame(raw_in1);
    destroy_frame(raw_in2);
    free_2d_float_array(in1, height);
    free_2d_float_array(in2, height);
    free(fx);
    free(fy);
    free(ft);
    destroy_frame(out_gpu);
    destroy_frame(out_cpu);
    return 0;
}
