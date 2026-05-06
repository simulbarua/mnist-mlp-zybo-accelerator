# MNIST MLP Zybo Accelerator

Hardware MLP inference accelerator for MNIST digit classification on the Digilent Zybo Z7-10 FPGA.

**Network:** 784 → FC1(64, ReLU) → FC2(32, ReLU) → FC3(10, argmax)  
**Weights:** INT8 post-training quantization  
**Throughput:** ~0.40 ms per inference at 100 MHz  
**Interface:** UART — send a 28×28 image from a PC, receive the predicted class

---

## Repository structure

```
mnist-mlp-zybo-accelerator/
├── training/                        # Python training and weight export
│   ├── train_and_export.py
│   ├── collect_sample_images.py
│   ├── infer_image.py
│   ├── requirements.txt
│   └── outputs/
│       └── coe/                     # Exported quantized weights (.coe + scales.txt)
├── hardware/
│   ├── vivado/                      # Vivado project sources
│   │   ├── create_vivado_project.tcl
│   │   └── src/
│   │       ├── rtl/                 # mlp_top.v, mlp_engine.v, axilite_ctrl.v, mlp_params.vh
│   │       └── constraints/         # zybo_z710_mlp.xdc
│   └── vitis/                       # Vitis workspace
│       └── mlp_accelerator_app/
│           ├── src/                 # main.c, mlp_bram_init.c, CMakeLists.txt, UserConfig.cmake, lscript.ld
│           └── include/             # mlp_bram_init.h, weights_biases.h
├── host/                            # PC-side scripts
│   ├── send_image.py
│   ├── int_infer_check.py
│   └── requirements.txt
└── data/
    └── test_images/                 # 100 sample MNIST PNGs
```

---

## Prerequisites

| Tool | Version |
|---|---|
| Vivado | 2025.2.1 |
| Vitis | 2025.2.1 |
| Python | 3.10+ |
| Board | Digilent Zybo Z7-10 |

---

## Step 1 — Train and export weights

```bash
cd training
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
python train_and_export.py
```

This trains the MLP on MNIST and writes:
- `training/outputs/coe/` — quantized weight `.coe` files and `scales.txt`
- `hardware/vivado/src/rtl/mlp_params.vh` — requantization shift parameters for the RTL
- `hardware/vitis/mlp_accelerator_app/include/weights_biases.h` — C header for BRAM loading

> Re-run this step any time you retrain. After updating `mlp_params.vh` you must re-synthesise in Vivado.

---

## Step 2 — Recreate the Vivado project

1. Open **Vivado 2025.2.1**.
2. In the Tcl console, `cd` to the vivado directory and source the script:

```tcl
cd {C:/path/to/mnist-mlp-zybo-accelerator/hardware/vivado}
source create_vivado_project.tcl
```

This recreates the full project — RTL sources, constraints, and block design — in `hardware/vivado/`.

---

## Step 3 — Synthesise, implement, and generate bitstream

In Vivado:

1. **Run Synthesis** — Flow Navigator → Run Synthesis
2. **Run Implementation** — Flow Navigator → Run Implementation
3. **Generate Bitstream** — Flow Navigator → Generate Bitstream
4. **Export Hardware (XSA)** — File → Export → Export Hardware → include bitstream → save as `hardware/vivado/mnist_mlp_accelerator_wrapper.xsa`

---

## Step 4 — Create the Vitis platform

1. Open **Vitis 2025.2.1**, set workspace to `hardware/vitis/`.
2. Create a new platform component:
   - **Name:** `zybo_z710_platform`
   - **XSA:** select `hardware/vivado/mnist_mlp_accelerator_wrapper.xsa`
   - **OS:** standalone, processor `ps7_cortexa9_0`
3. Build the platform.

---

## Step 5 — Build the firmware

1. In the same Vitis workspace, add the existing app component:
   - Point Vitis at `hardware/vitis/mlp_accelerator_app/`
   - Platform: `zybo_z710_platform`, domain: `standalone_ps7_cortexa9_0`
2. Build the app — this produces `mlp_accelerator_app.elf`.

---

## Step 6 — Program the board

1. Connect the Zybo Z7-10 over USB (JTAG + UART).
2. In Vitis, run the application on the board (or use the Vivado Hardware Manager to program the bitstream, then load the ELF via Vitis debugger).
3. Open a serial terminal at **115200 baud** on the board's UART port.
4. You should see:
```
BRAM loaded OK.
UART inference mode ready.
Send frame: 4-byte header "IMG1" + 784 grayscale bytes.
READY
```

---

## Step 7 — Run inference from the PC

```bash
cd host
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

Send an image:

```bash
python send_image.py ../data/test_images/mnist_0_label_7.png --port COM4
```

Expected output:

```
sent 784 image bytes
[board] RESULT class=7 time_us=412
[host] mnist_0_label_7.png: RESULT class=7 time_us=412
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--port` | required | Serial port (e.g. `COM4`, `/dev/ttyUSB1`) |
| `--baud` | 115200 | UART baud rate |
| `--invert` | off | Invert image (for white digit on black background) |
| `--no-wait-ready` | off | Send immediately without waiting for `READY` |
| `--timeout` | 10.0 s | Serial read timeout |

---

## Software-only inference check

To verify the integer arithmetic matches the FPGA before flashing:

```bash
cd host
python int_infer_check.py ../data/test_images/mnist_0_label_7.png
```

This mirrors `mlp_engine.v` exactly in Python — same INT8 MACs, same requantization shifts, same argmax.

---

## AXI address map

| Peripheral | Base address | Size |
|---|---|---|
| Param BRAM (weights/biases) | `0x40000000` | 64 KB |
| Input BRAM (image pixels) | `0x40010000` | 1 KB |
| MLP AXI-Lite control | `0x40020000` | 4 KB |

### AXI-Lite registers

| Offset | Direction | Description |
|---|---|---|
| `0x00` | Write | Control — write `0x1` to start inference |
| `0x04` | Read | Status — bit 0: done (read-to-clear), bit 1: busy |
| `0x08` | Read | Result — bits [3:0] predicted class (0–9) |
