/**
 * (c) 2019 Dzung Pham
 */

#include <math.h>
#include <math_constants.h> // CUDA Math constants
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "jpeglib.h"

// ----------------------- Helper code ------------------------

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

#define BYTE_PER_KB 1024
#define BYTE_PER_MB 1048576
#define RUN_TEST 0 // Whether or not to test against uniprocessor
#define JPEG_OUTPUT_QUALITY 75
#define R_GRAYSCALE 0.2126
#define G_GRAYSCALE 0.7152
#define B_GRAYSCALE 0.0722
#define TWO_PI (CUDART_PI_F * 2)
#define EIGEN_THRESHOLD 0.01

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

    for (i = 0; i < f1->image_height; i++){
        for (j = 0; j < f1->image_width; j++){
            for (k = 0; k < f1->num_channels; k++){
                JSAMPLE j1 = f1->row_pointers[i][(f1->num_channels) * j + k];
                JSAMPLE j2 = f2->row_pointers[i][(f2->num_channels) * j + k];
                if (abs(j1 - j2) > 1) {
                    fprintf(stderr, "Values do not match at (%d, %d, %d) \n", i, j, k);
                    fprintf(stderr, "in %d\n", j1);
                    fprintf(stderr, "to %d\n", j2);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    printf("Results are the same\n");
}

/**
 * Queries the properties of the GPU device with the given device_id,
 * fills it in the given device_prop, and print interesting info
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
     // Calculate determinant and make sure it is not 0 in order for the matrix to be invertible
    float det = AtA_00 * AtA_11 - AtA_01 * AtA_01;
    if (det == 0.0f) {
        return make_float2(0.0f, 0.0f);
    }

    // Calculate the eigenvalues of A^T A and make sure they are > threshold
    float trace_half = (AtA_00 + AtA_11) / 2.0f; // Half of the trace of A^T A
    float delta = sqrt(trace_half * trace_half - det);
    float eigen1 = trace_half + delta;
    float eigen2 = trace_half - delta;
    if (eigen1 <= EIGEN_THRESHOLD || eigen2 <= EIGEN_THRESHOLD) {
        return make_float2(0.0f, 0.0f);
    }

    // Calculate flow components
    return make_float2(
        AtA_11 * Atb_0 - AtA_01 * Atb_1,
        -AtA_01 * Atb_0 + AtA_00 * Atb_1
    ) / det;
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
 * Normalize RGB frame and grayscale if necessary
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
 */
frame_ptr create_flow_visualization(
    float *angle, float *mag,
    int height, int width, int s
) {
    float cur_mag, max_mag = -INFINITY, min_mag = INFINITY;
    int row, col;

    // Find min and max magnitude
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

    frame_ptr out = allocate_frame(height, width, 3);
    if (max_mag <= min_mag) {
        return out;
    }

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

    return out;
}

/**
 * Calculate optical flow with Lucas-Kanade using CPU from 2 normalized frames
 *
 * @param in1 Frame at time t
 * @param in2 Frame at time t + 1
 * @param window_size An odd positive integer >= 3 for the window size
 * @return A frame for visualizing the optical flow.
 */
frame_ptr uniprocessor_lucas_kanade(
    float **in1, float **in2,
    int height, int width, int window_size
) {
    // Calculate derivatives
    float **fx, **fy, **ft;
    fx = alloc_2d_float_array(height, width);
    fy = alloc_2d_float_array(height, width);
    ft = alloc_2d_float_array(height, width);

    int row, col;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
                fx[row][col] =
                    in1[row - 1][col + 1] + in1[row][col + 1] + in1[row + 1][col + 1] -
                    in1[row - 1][col - 1] - in1[row][col - 1] - in1[row + 1][col - 1];
                fy[row][col] =
                    in1[row - 1][col - 1] + in1[row - 1][col] + in1[row - 1][col + 1] -
                    in1[row + 1][col - 1] - in1[row + 1][col] - in1[row + 1][col + 1];
            }
            ft[row][col] = in2[row][col] - in1[row][col];
        }
    }

    // Calculate flows
    int i, j;
    float *angle = alloc_1d_float_array(height * width);
    float *mag = alloc_1d_float_array(height * width);
    float AtA_00, AtA_01, AtA_11, Atb_0, Atb_1;
    float2 flow;
    int s = window_size / 2;

    for (row = s; row < height - s; row++) {
        for (col = s; col < width - s; col++) {
            AtA_00 = AtA_01 = AtA_11 = Atb_0 = Atb_1 = 0.0f;
            for (i = row - s; i <= row + s; i++) {
                for (j = col - s; j <= col + s; j++) {
                    AtA_00 += fx[i][j] * fx[i][j];
                    AtA_11 += fy[i][j] * fy[i][j];
                    AtA_01 += fx[i][j] * fy[i][j];
                    Atb_0 -= fx[i][j] * ft[i][j];
                    Atb_1 -= fy[i][j] * ft[i][j];
                }
            }

            // Calculate flow and convert to polar coordinates
            flow = calc_flow_from_matrix(AtA_00, AtA_01, AtA_11, Atb_0, Atb_1);
            angle[row * width + col] = get_angle(flow);
            mag[row * width + col] = get_magnitude(flow);
        }
    }

    // Create and write to output frame
    frame_ptr out = create_flow_visualization(angle, mag, height, width, s);

    // Clean up
    free_2d_float_array(fx, height);
    free_2d_float_array(fy, height);
    free_2d_float_array(ft, height);
    free(angle);
    free(mag);

    return out;
}

__global__ void simple_derivative_kernel(
    float *in1, float *in2,
    float *fx, float *fy, float *ft,
    int height, int width
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= 0 && row < height && col >= 0 && col < width) {
        int offset = row * width + col;
        float top_left, top_mid, top_right, mid_left, mid_right, bottom_left, bottom_mid, bottom_right;

        if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
            top_left = in1[offset - width - 1];
            top_mid = in1[offset - width];
            top_right = in1[offset - width + 1];
            bottom_left = in1[offset + width - 1];
            bottom_mid = in1[offset + width];
            bottom_right = in1[offset + width + 1];
            mid_left = in1[offset - 1];
            mid_right = in1[offset + 1];

            fx[offset] = top_right - top_left + mid_right - mid_left + bottom_right - bottom_left;
            fy[offset] = top_left - bottom_left + top_mid - bottom_mid + top_right - bottom_right;
            ft[offset] = in2[offset] - in1[offset];
        } else {
            fx[offset] = fy[offset] = ft[offset] = 0.0f;
        }
    }
}

__global__ void simple_lucas_kanade_kernel(
    float *fx, float *fy, float *ft,
    float *angle, float *mag,
    int height, int width, int s
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= 0 && row < height && col >= 0 && col < width) {
        if (row < s || row >= height - s || col < s || col >= width - s) {
            angle[row * width + col] = mag[row * width + col] = 0.0f;
        } else {
            float AtA_00, AtA_01, AtA_11, Atb_0, Atb_1; // Entries of (A^T A)^-1 and A^T b
            float cur_fx, cur_fy, cur_ft;
            AtA_00 = AtA_01 = AtA_11 = Atb_0 = Atb_1 = 0.0f;

            for (int i = row - s; i <= row + s; i++) {
                for (int j = col - s; j <= col + s; j++) {
                    cur_fx = fx[i * width + j];
                    cur_fy = fy[i * width + j];
                    cur_ft = ft[i * width + j];

                    AtA_00 += cur_fx * cur_fx;
                    AtA_11 += cur_fy * cur_fy;
                    AtA_01 += cur_fx * cur_fy;
                    Atb_0 -= cur_fx * cur_ft;
                    Atb_1 -= cur_fy * cur_ft;
                }
            }

            // Calculate flow and convert to polar coordinates
            float2 flow = calc_flow_from_matrix(AtA_00, AtA_01, AtA_11, Atb_0, Atb_1);
            angle[row * width + col] = get_angle(flow);
            mag[row * width + col] = get_magnitude(flow);
        }
    }
}

frame_ptr run_kernel(
    float **in1, float **in2,
    int height, int width,
    int window_size, int block_size
) {
    dim3 block_dim(block_size, block_size, 1);
    dim3 grid_dim(ceil((float) width / block_size), ceil((float) height / block_size), 1);

    int s = window_size / 2;
    size_t size = height * width * sizeof(float);
    float *flattened_in1, *flattened_in2, *angle, *mag;
    float *d_in1, *d_in2, *d_fx, *d_fy, *d_ft, *d_angle, *d_mag;
    flattened_in1 = flatten_2d_float_array(in1, height, width);
    flattened_in2 = flatten_2d_float_array(in2, height, width);
    angle = alloc_1d_float_array(size);
    mag = alloc_1d_float_array(size);
    checkCudaErrors(cudaMalloc((void **) &d_fx, size));
    checkCudaErrors(cudaMalloc((void **) &d_fy, size));
    checkCudaErrors(cudaMalloc((void **) &d_ft, size));
    checkCudaErrors(cudaMalloc((void **) &d_in1, size));
    checkCudaErrors(cudaMalloc((void **) &d_in2, size));
    checkCudaErrors(cudaMalloc((void **) &d_angle, size));
    checkCudaErrors(cudaMalloc((void **) &d_mag, size));
    checkCudaErrors(cudaMemcpy(d_in1, flattened_in1, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in2, flattened_in2, size, cudaMemcpyHostToDevice));

    simple_derivative_kernel<<<grid_dim, block_dim>>>(
        d_in1, d_in2, d_fx, d_fy, d_ft, height, width
    );

    simple_lucas_kanade_kernel<<<grid_dim, block_dim>>>(
        d_fx, d_fy, d_ft, d_angle, d_mag, height, width, s
    );

    checkCudaErrors(cudaMemcpy(angle, d_angle, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mag, d_mag, size, cudaMemcpyDeviceToHost));
    frame_ptr out = create_flow_visualization(angle, mag, height, width, s);

    free(flattened_in1);
    free(flattened_in2);
    free(angle);
    free(mag);
    checkCudaErrors(cudaFree(d_fx));
    checkCudaErrors(cudaFree(d_fy));
    checkCudaErrors(cudaFree(d_ft));
    checkCudaErrors(cudaFree(d_in1));
    checkCudaErrors(cudaFree(d_in2));
    checkCudaErrors(cudaFree(d_angle));
    checkCudaErrors(cudaFree(d_mag));
    return out;
}

/**
 * Host main routine
 */
int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr,
            "Usage: lucas_kanade <WINDOW_SIZE>"
            " <PATH_TO_FRAME_1> <PATH_TO_FRAME_2> <PATH_TO_FLOW_OUTPUT>\n");
        exit(EXIT_FAILURE);
    }

    // Get inputs
    frame_ptr raw_in1 = read_JPEG_file(argv[2]);
    frame_ptr raw_in2 = read_JPEG_file(argv[3]);
    checkFrameDim(raw_in1, raw_in2);

    int window_size = atoi(argv[1]);
    if (window_size < 3 || window_size % 2 != 1) {
        fprintf(stderr, "Window size must be an odd integer >= 3");
        exit(EXIT_FAILURE);
    }

    // Create 2d array of normalized image pixel between [0, 1]
    int height = raw_in1->image_height;
    int width = raw_in2->image_width;
    float **in1 = get_normalized_2d_float_array(raw_in1);
    float **in2 = get_normalized_2d_float_array(raw_in2);

    // Get max block size from GPU's props
    int device_id = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(device_id));
    cudaDeviceProp device_prop;
    query_device(&device_prop, device_id);
    int max_block_size = (int) floor(sqrt(device_prop.maxThreadsPerBlock));

    // Run and test
    frame_ptr out_gpu = run_kernel(in1, in2, height, width, window_size, max_block_size);
    frame_ptr out_cpu = uniprocessor_lucas_kanade(in1, in2, height, width, window_size);
    checkResults(out_gpu, out_cpu);

    // Write out the flows and clean up
    write_JPEG_file(argv[4], out_gpu, JPEG_OUTPUT_QUALITY);
    destroy_frame(raw_in1);
    destroy_frame(raw_in2);
    free_2d_float_array(in1, height);
    free_2d_float_array(in2, height);
    destroy_frame(out_gpu);
    destroy_frame(out_cpu);
    return 0;
}
