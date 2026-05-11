/* mlp_bram_init.h
 *
 * Declares the BRAM initialisation and readback-check functions for the
 * 4-bank parallel MLP accelerator on the Zybo Z7-10.
 *
 * Weight/bias layout (4-bank parallel):
 *   The MLP engine computes 4 output neurons simultaneously.  Each bank k
 *   holds weights/biases for neurons {k, k+4, k+8, ...} in every layer.
 *   FC3 is padded to 12 neurons (3 groups of 4); the 2 extra neurons carry
 *   zero weights/biases and are skipped by the argmax (which checks [0..9]).
 *
 * Per-bank structure (byte offsets; same for all 4 banks):
 *   FC1_WEIGHT_BANK_OFFSET  0x00000000  16 neurons × 784 B = 12 544 B
 *   FC1_BIAS_BANK_OFFSET    0x00003100  16 neurons ×   4 B =    64 B
 *   FC2_WEIGHT_BANK_OFFSET  0x00003140   8 neurons ×  64 B =   512 B
 *   FC2_BIAS_BANK_OFFSET    0x00003340   8 neurons ×   4 B =    32 B
 *   FC3_WEIGHT_BANK_OFFSET  0x00003360   3 neurons ×  32 B =    96 B
 *   FC3_BIAS_BANK_OFFSET    0x000033C0   3 neurons ×   4 B =    12 B
 *   Total per bank: 13 260 B  (<16 KB)
 *
 * AXI address map (must match Vivado Address Editor):
 *   Bank 0  0x40000000  16 KB
 *   Bank 1  0x40004000  16 KB
 *   Bank 2  0x40008000  16 KB
 *   Bank 3  0x4000C000  16 KB
 *   Input   0x40010000   1 KB  (unchanged)
 *   Ctrl    0x40020000   4 KB  (unchanged)
 */

#ifndef MLP_BRAM_INIT_H
#define MLP_BRAM_INIT_H

#include <stdint.h>

/* ── Portable rounding without libm ─────────────────────────────────────────
 * Replaces roundf() so the firmware does not need -lm at link time.
 */
static inline int mlp_roundf(float x)
{
    return (x >= 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f);
}

/* ── Param BRAM bank base addresses ─────────────────────────────────────────
 * Change to match Vivado → Address Editor → AXI BRAM Controller assignments.
 */
#define MLP_PARAM_BRAM_BANK0_BASE  0x40000000UL
#define MLP_PARAM_BRAM_BANK1_BASE  0x40004000UL
#define MLP_PARAM_BRAM_BANK2_BASE  0x40008000UL
#define MLP_PARAM_BRAM_BANK3_BASE  0x4000C000UL

/* ── Base address of the Input BRAM ─────────────────────────────────────────
 * Unchanged from the single-bank design.
 */
#define MLP_INPUT_BRAM_BASE_ADDR   0x40010000UL

/* ── Write all weights and biases into the 4 param BRAM banks ───────────────
 * Call once at start-up after the PL is programmed.
 */
void mlp_bram_init(void);

/* ── Verify BRAM contents by reading back the first word of each block ───────
 * Returns 0 on PASS, -1 on any FAIL.
 */
int  mlp_bram_readback_check(void);

/* ── Write one quantized input image into the Input BRAM ────────────────────
 * pixels : 784 raw uint8 pixel values (row-major, 0–255).
 */
void mlp_write_input(const uint8_t *pixels);

/* ── Spot-check the Input BRAM against freshly quantised reference values ─── */
int  mlp_verify_input_bram(const uint8_t *pixels);

#endif /* MLP_BRAM_INIT_H */
