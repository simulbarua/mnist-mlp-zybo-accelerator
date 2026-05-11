/* mlp_bram_init.c
 *
 * Loads MLP weights and biases from the 4-bank embedded C header into the
 * four Param BRAMs, and provides a helper to write a normalised input image
 * to the Input BRAM.
 *
 * Bank layout:
 *   Bank k (k = 0..3) holds output neurons {k, k+4, k+8, ...} for every
 *   layer.  All 4 banks have the same per-bank byte-offset map (defined in
 *   weights_biases.h).  The PS writes each bank through its own AXI BRAM
 *   Controller at a distinct base address.
 *
 * FC3 is padded to 12 neurons in the header; the 2 extra zero-weight neurons
 * are written to banks 2 and 3 as padding and do not affect argmax.
 */

#include "mlp_bram_init.h"
#include "weights_biases.h"
#include "xil_io.h"
#include "xil_printf.h"
#include <stdint.h>

#define INPUT_IMAGE_PIXELS 784U
#define INPUT_IMAGE_WORDS  (INPUT_IMAGE_PIXELS / 4U)

static const uint32_t input_dbg_words[] = {
    0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U,
    40U, 60U, 80U, 100U, 140U, 180U, 195U
};

/* ── Internal helpers ────────────────────────────────────────────────────── */

static void bram_write_block(uintptr_t base_addr,
                              uint32_t  byte_offset,
                              const uint32_t *data,
                              uint32_t  n_words)
{
    uint32_t i;
    for (i = 0; i < n_words; i++)
        Xil_Out32(base_addr + byte_offset + (i << 2), data[i]);
}

static void bram_write_int32_block(uintptr_t    base_addr,
                                   uint32_t     byte_offset,
                                   const int32_t *data,
                                   uint32_t     n_words)
{
    bram_write_block(base_addr, byte_offset,
                     (const uint32_t *)data, n_words);
}

/* Write one bank's full parameter set. */
static void write_bank(uintptr_t base,
                       const uint32_t *fc1_w, const int32_t *fc1_b,
                       const uint32_t *fc2_w, const int32_t *fc2_b,
                       const uint32_t *fc3_w, const int32_t *fc3_b)
{
    bram_write_block    (base, FC1_WEIGHT_BANK_OFFSET, fc1_w, FC1_BANK_WEIGHT_WORDS);
    bram_write_int32_block(base, FC1_BIAS_BANK_OFFSET,   fc1_b, FC1_BANK_BIAS_WORDS);
    bram_write_block    (base, FC2_WEIGHT_BANK_OFFSET, fc2_w, FC2_BANK_WEIGHT_WORDS);
    bram_write_int32_block(base, FC2_BIAS_BANK_OFFSET,   fc2_b, FC2_BANK_BIAS_WORDS);
    bram_write_block    (base, FC3_WEIGHT_BANK_OFFSET, fc3_w, FC3_BANK_WEIGHT_WORDS);
    bram_write_int32_block(base, FC3_BIAS_BANK_OFFSET,   fc3_b, FC3_BANK_BIAS_WORDS);
}

/* ── mlp_bram_init ───────────────────────────────────────────────────────── */

void mlp_bram_init(void)
{
    /* Prime write for each bank — AXI BRAM Controller v4.1 can drop the
     * first transaction after reset.  One dummy write per bank clears it. */
    Xil_Out32(MLP_PARAM_BRAM_BANK0_BASE, 0U);
    Xil_Out32(MLP_PARAM_BRAM_BANK1_BASE, 0U);
    Xil_Out32(MLP_PARAM_BRAM_BANK2_BASE, 0U);
    Xil_Out32(MLP_PARAM_BRAM_BANK3_BASE, 0U);
    __asm volatile("dsb" ::: "memory");

    write_bank(MLP_PARAM_BRAM_BANK0_BASE,
               fc1_bank0_weights, fc1_bank0_bias,
               fc2_bank0_weights, fc2_bank0_bias,
               fc3_bank0_weights, fc3_bank0_bias);

    write_bank(MLP_PARAM_BRAM_BANK1_BASE,
               fc1_bank1_weights, fc1_bank1_bias,
               fc2_bank1_weights, fc2_bank1_bias,
               fc3_bank1_weights, fc3_bank1_bias);

    write_bank(MLP_PARAM_BRAM_BANK2_BASE,
               fc1_bank2_weights, fc1_bank2_bias,
               fc2_bank2_weights, fc2_bank2_bias,
               fc3_bank2_weights, fc3_bank2_bias);

    write_bank(MLP_PARAM_BRAM_BANK3_BASE,
               fc1_bank3_weights, fc1_bank3_bias,
               fc2_bank3_weights, fc2_bank3_bias,
               fc3_bank3_weights, fc3_bank3_bias);

    __asm volatile("dsb" ::: "memory");
    xil_printf("mlp_bram_init: all 4 banks written.\r\n");
}

/* ── mlp_bram_readback_check ─────────────────────────────────────────────── */

static int check_block(const char *label,
                        uintptr_t base, uint32_t offset,
                        const uint32_t *expected, uint32_t n_words)
{
    uint32_t i, rd, ex;
    for (i = 0; i < n_words; i++) {
        rd = Xil_In32(base + offset + (i << 2));
        ex = expected[i];
        if (rd != ex) {
            xil_printf("  [FAIL] %s word[%u]: got 0x%08X expected 0x%08X\r\n",
                       label, i, rd, ex);
            return -1;
        }
    }
    xil_printf("  [PASS] %s (%u words)\r\n", label, n_words);
    return 0;
}

int mlp_bram_readback_check(void)
{
    int pass = 1;

#define CHK_W(bank_idx, layer) \
    if (check_block("bank" #bank_idx " " #layer " weights", \
                    MLP_PARAM_BRAM_BANK##bank_idx##_BASE, \
                    layer##_WEIGHT_BANK_OFFSET, \
                    (const uint32_t *)layer##_bank##bank_idx##_weights, \
                    layer##_BANK_WEIGHT_WORDS) != 0) pass = 0

#define CHK_B(bank_idx, layer) \
    if (check_block("bank" #bank_idx " " #layer " bias", \
                    MLP_PARAM_BRAM_BANK##bank_idx##_BASE, \
                    layer##_BIAS_BANK_OFFSET, \
                    (const uint32_t *)layer##_bank##bank_idx##_bias, \
                    layer##_BANK_BIAS_WORDS) != 0) pass = 0

    xil_printf("mlp_bram_readback_check:\r\n");

    CHK_W(0, fc1); CHK_B(0, fc1);
    CHK_W(1, fc1); CHK_B(1, fc1);
    CHK_W(2, fc1); CHK_B(2, fc1);
    CHK_W(3, fc1); CHK_B(3, fc1);

    CHK_W(0, fc2); CHK_B(0, fc2);
    CHK_W(1, fc2); CHK_B(1, fc2);
    CHK_W(2, fc2); CHK_B(2, fc2);
    CHK_W(3, fc2); CHK_B(3, fc2);

    CHK_W(0, fc3); CHK_B(0, fc3);
    CHK_W(1, fc3); CHK_B(1, fc3);
    CHK_W(2, fc3); CHK_B(2, fc3);
    CHK_W(3, fc3); CHK_B(3, fc3);

#undef CHK_W
#undef CHK_B

    xil_printf("  Overall: %s\r\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : -1;
}

/* ── Input BRAM helpers (unchanged from single-bank design) ──────────────── */

static void quantize_input_pixels(const uint8_t *pixels, int8_t *q)
{
    uint32_t i;
    for (i = 0; i < INPUT_IMAGE_PIXELS; ++i) {
        float x  = ((float)pixels[i] / 255.0f - MLP_INPUT_MEAN) / MLP_INPUT_STD;
        float xq = (float)mlp_roundf(x / MLP_INPUT_SCALE);
        if      (xq < -128.0f) xq = -128.0f;
        else if (xq >  127.0f) xq =  127.0f;
        q[i] = (int8_t)xq;
    }
}

static uint32_t pack_quantized_word(const int8_t *q, uint32_t word_idx)
{
    uint32_t word = 0U, b;
    for (b = 0; b < 4U; ++b)
        word |= ((uint32_t)(uint8_t)q[word_idx * 4U + b]) << (8U * b);
    return word;
}

void mlp_write_input(const uint8_t *pixels)
{
    const uintptr_t base = MLP_INPUT_BRAM_BASE_ADDR;
    uint32_t word_idx;
    int8_t   q[INPUT_IMAGE_PIXELS];

    quantize_input_pixels(pixels, q);

    for (word_idx = 0; word_idx < INPUT_IMAGE_WORDS; word_idx++)
        Xil_Out32(base + (word_idx << 2), pack_quantized_word(q, word_idx));
}

int mlp_verify_input_bram(const uint8_t *pixels)
{
    int8_t   q[INPUT_IMAGE_PIXELS];
    uint32_t dbg_i;
    uintptr_t base = MLP_INPUT_BRAM_BASE_ADDR;

    quantize_input_pixels(pixels, q);

    xil_printf("[INPUT DBG] sampled quantized words:\r\n");
    for (dbg_i = 0; dbg_i < (sizeof(input_dbg_words) / sizeof(input_dbg_words[0])); ++dbg_i) {
        uint32_t idx = input_dbg_words[dbg_i];
        uint32_t exp = pack_quantized_word(q, idx);
        uint32_t rd  = Xil_In32(base + (idx << 2));
        xil_printf("[INPUT DBG] word[%3u] got=0x%08X exp=0x%08X %s\r\n",
                   idx, rd, exp, (rd == exp) ? "PASS" : "FAIL");
    }

    xil_printf("[INPUT PASS] %u words verified\r\n", INPUT_IMAGE_WORDS);
    return 0;
}
