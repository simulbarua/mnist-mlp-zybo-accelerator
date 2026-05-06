#include <stdint.h>
#include "xil_io.h"
#include "xil_printf.h"
#include "xiltimer.h"
#include "xparameters.h"
#include "mlp_bram_init.h"

/* AXI-Lite register addresses: must match Vivado Address Editor */
#define MLP_AXILITE_BASE  0x40020000UL
#define MLP_CTRL_ADDR    (MLP_AXILITE_BASE + 0x00UL)
#define MLP_STATUS_ADDR  (MLP_AXILITE_BASE + 0x04UL)
#define MLP_RESULT_ADDR  (MLP_AXILITE_BASE + 0x08UL)

/* BRAM sizes: must match the Address Editor ranges in Vivado. */
#define PARAM_BRAM_WORDS  (0x10000U >> 2)
#define INPUT_BRAM_WORDS  (0x0400U  >> 2)

#define IMAGE_PIXELS      784U

#define UART_MAGIC0       'I'
#define UART_MAGIC1       'M'
#define UART_MAGIC2       'G'
#define UART_MAGIC3       '1'

/*
 * zero_bram()
 * Write 0x00000000 to every 32-bit word of a BRAM over the AXI bus.
 *
 * A priming write is issued first because the Xilinx AXI BRAM Controller
 * v4.1 can drop the very first write transaction after reset.
 */
static void zero_bram(uintptr_t base, uint32_t n_words)
{
    uint32_t i;

    Xil_Out32(base, 0U);
    __asm volatile("dsb" ::: "memory");

    for (i = 0; i < n_words; i++) {
        Xil_Out32(base + (i << 2), 0U);
    }

    __asm volatile("dsb" ::: "memory");
}

static void initialize_device(void)
{
    xil_printf("initialize_device: zeroing Param BRAM...\r\n");
    zero_bram(MLP_PARAM_BRAM_BASE_ADDR, PARAM_BRAM_WORDS);

    xil_printf("initialize_device: zeroing Input BRAM...\r\n");
    zero_bram(MLP_INPUT_BRAM_BASE_ADDR, INPUT_BRAM_WORDS);

    xil_printf("initialize_device: done.\r\n");
}

static void cleanup_device(void)
{
    xil_printf("cleanup_device: clearing Param BRAM...\r\n");
    zero_bram(MLP_PARAM_BRAM_BASE_ADDR, PARAM_BRAM_WORDS);

    xil_printf("cleanup_device: clearing Input BRAM...\r\n");
    zero_bram(MLP_INPUT_BRAM_BASE_ADDR, INPUT_BRAM_WORDS);

    xil_printf("cleanup_device: done.\r\n");
}

static void write_image_to_input_bram(const uint8_t *pixels)
{
    mlp_write_input(pixels);
}

static void uart_recv_exact(uint8_t *dst, uint32_t n_bytes)
{
    uint32_t i;

    for (i = 0; i < n_bytes; ++i) {
        dst[i] = (uint8_t)inbyte();
    }
}

static void uart_wait_for_image_header(void)
{
    uint8_t state = 0U;

    for (;;) {
        uint8_t ch = (uint8_t)inbyte();

        switch (state) {
        case 0:
            state = (ch == (uint8_t)UART_MAGIC0) ? 1U : 0U;
            break;
        case 1:
            state = (ch == (uint8_t)UART_MAGIC1) ? 2U :
                    (ch == (uint8_t)UART_MAGIC0) ? 1U : 0U;
            break;
        case 2:
            state = (ch == (uint8_t)UART_MAGIC2) ? 3U :
                    (ch == (uint8_t)UART_MAGIC0) ? 1U : 0U;
            break;
        default:
            if (ch == (uint8_t)UART_MAGIC3) {
                return;
            }
            state = (ch == (uint8_t)UART_MAGIC0) ? 1U : 0U;
            break;
        }
    }
}

static uint32_t run_inference_and_get_class(uint64_t *elapsed_us)
{
    XTime t0;
    XTime t1;
    uint32_t status;
    uint32_t status_after_clear_1;
    uint32_t status_after_clear_2;
    uint32_t status_after_start;
    uint32_t final_status;
    uint32_t raw_result;
    uint32_t cls;
    uint64_t delta_counts;
    uint32_t timeout;
    uint32_t poll_count;
    int busy_seen;
    int done_seen;

    /* Clear any stale done_latch from a previous inference. STATUS[0] is
     * read-to-clear in axilite_ctrl.v, so discard one or two reads before
     * asserting a new start pulse.
     */
    status_after_clear_1 = Xil_In32(MLP_STATUS_ADDR);
    status_after_clear_2 = Xil_In32(MLP_STATUS_ADDR);
    // xil_printf("[CTRL DBG] status_clear1=0x%08X status_clear2=0x%08X\r\n",
    //            status_after_clear_1,
    //            status_after_clear_2);

    XTime_GetTime(&t0);
    Xil_Out32(MLP_CTRL_ADDR, 0x00000001UL);
    status_after_start = Xil_In32(MLP_STATUS_ADDR);
    // xil_printf("[CTRL DBG] status_after_start=0x%08X\r\n", status_after_start);

    busy_seen = ((status_after_start & 0x2U) != 0U);
    done_seen = ((status_after_start & 0x1U) != 0U);

    /* Wait until we observe either busy=1 or done=1. STATUS.done is
     * read-to-clear, so if the accelerator finishes before the next poll we
     * must not discard that completion just because busy already dropped.
     */
    if (!busy_seen && !done_seen) {
        timeout = 10000000U;
        poll_count = 0U;
        do {
            status = Xil_In32(MLP_STATUS_ADDR);
            // if (poll_count < 8U) {
            //     xil_printf("[CTRL DBG] start_poll[%u]=0x%08X\r\n", poll_count, status);
            // }
            ++poll_count;
            if ((status & 0x2U) != 0U) {
                busy_seen = 1;
                break;
            }
            if ((status & 0x1U) != 0U) {
                done_seen = 1;
                break;
            }
        } while (--timeout != 0U);

        if (timeout == 0U) {
            *elapsed_us = 0ULL;
            // xil_printf("[CTRL DBG] start_timeout last_status=0x%08X polls=%u\r\n",
            //            status,
            //            poll_count);
            return 0xFFFFFFFFU;
        }
    }

    // xil_printf("[CTRL DBG] start_seen busy=%u done=%u last_status=0x%08X\r\n",
    //            busy_seen,
    //            done_seen,
    //            status);

    if (!done_seen) {
        timeout = 100000000U;
        poll_count = 0U;
        do {
            status = Xil_In32(MLP_STATUS_ADDR);
            // if (poll_count < 8U) {
            //     xil_printf("[CTRL DBG] done_poll[%u]=0x%08X\r\n", poll_count, status);
            // }
            ++poll_count;
            if ((status & 0x1U) != 0U) {
                done_seen = 1;
                break;
            }
        } while (--timeout != 0U);

        if (timeout == 0U) {
            *elapsed_us = 0ULL;
            // xil_printf("[CTRL DBG] done_timeout last_status=0x%08X polls=%u\r\n",
            //            status,
            //            poll_count);
            return 0xFFFFFFFEU;
        }
    }

    XTime_GetTime(&t1);
    raw_result = Xil_In32(MLP_RESULT_ADDR);
    final_status = Xil_In32(MLP_STATUS_ADDR);
    cls = raw_result & 0xFU;

    delta_counts = (uint64_t)(t1 - t0);
    *elapsed_us = (delta_counts * 1000000ULL) / (uint64_t)(XPAR_CPU_CORE_CLOCK_FREQ_HZ / 2UL);
    // xil_printf("[CTRL DBG] done_seen status=0x%08X final_status=0x%08X raw_result=0x%08X counts=%llu polls=%u\r\n",
    //            status,
    //            final_status,
    //            raw_result,
    //            delta_counts,
    //            poll_count);
    return cls;
}

int main(void)
{
    static uint8_t image_pixels[IMAGE_PIXELS];
    uint32_t cls;
    uint64_t elapsed_us;

    /* Enable the ARM Cortex-A9 Global Timer. The standalone BSP does not
     * guarantee this is running after reset, so XTime_GetTime() returns 0
     * until bit 0 of the control register is set. */
    Xil_Out32(0xF8F00208U, 0x00000001U);
    initialize_device();

    xil_printf("Loading MLP parameters into BRAM...\r\n");
    mlp_bram_init();
    if (mlp_bram_readback_check() != 0) {
        xil_printf("ERROR: BRAM readback failed. Check MLP_PARAM_BRAM_BASE_ADDR.\r\n");
        cleanup_device();
        return -1;
    }

    xil_printf("BRAM loaded OK.\r\n");
    xil_printf("UART inference mode ready.\r\n");
    xil_printf("Send frame: 4-byte header \"IMG1\" + 784 grayscale bytes.\r\n");

    for (;;) {
        xil_printf("READY\r\n");
        uart_wait_for_image_header();
        uart_recv_exact(image_pixels, IMAGE_PIXELS);

        write_image_to_input_bram(image_pixels);
        (void)mlp_verify_input_bram(image_pixels);
        cls = run_inference_and_get_class(&elapsed_us);

        if (cls == 0xFFFFFFFFU) {
            xil_printf("RESULT error=busy_timeout time_us=0\r\n");
        } else if (cls == 0xFFFFFFFEU) {
            xil_printf("RESULT error=done_timeout time_us=0\r\n");
        } else {
            xil_printf("RESULT class=%u time_us=%llu\r\n",
                       (unsigned)cls,
                       elapsed_us);
        }
    }
}
