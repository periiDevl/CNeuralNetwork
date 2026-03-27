"""
paint_digit.py  —  Draw a digit, export MNIST-ready .bin for your C network.

Requirements:
    pip install pillow numpy

Run:
    python paint_digit.py                  # saves to input_image.bin
    python paint_digit.py my_digit.bin     # saves to custom path
"""

import sys
import struct
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageFilter, ImageOps

OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "input_image.bin"

# ── Canvas size (we draw big, then shrink to 28x28) ──────────────────────────
CANVAS_PX   = 420          # display canvas in pixels
GRID        = 28           # MNIST grid
CELL        = CANVAS_PX // GRID   # pixels per logical "MNIST cell"
BRUSH_CELLS = 1.5          # brush radius in MNIST cells

class DigitPainter:
    def __init__(self, root):
        self.root = root
        root.title("Digit Painter → MNIST .bin")
        root.configure(bg="#0d0d0d")
        root.resizable(False, False)

        # ── grid: 28x28 float values (0=background, 1=ink) ──
        self.grid = np.zeros((GRID, GRID), dtype=np.float32)

        self._build_ui()
        self._bind_events()
        self._redraw()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title
        title = tk.Label(self.root, text="DRAW A DIGIT",
                         font=("Courier New", 13, "bold"),
                         fg="#39ff14", bg="#0d0d0d", pady=8)
        title.pack()

        # Canvas frame with glow border
        frame = tk.Frame(self.root, bg="#39ff14", padx=2, pady=2)
        frame.pack(padx=20)

        self.canvas = tk.Canvas(frame,
                                width=CANVAS_PX, height=CANVAS_PX,
                                bg="#000000", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()

        # Brush size slider
        ctrl = tk.Frame(self.root, bg="#0d0d0d", pady=10)
        ctrl.pack(fill="x", padx=20)

        tk.Label(ctrl, text="BRUSH", font=("Courier New", 9),
                 fg="#888", bg="#0d0d0d").pack(side="left")

        self.brush_var = tk.DoubleVar(value=BRUSH_CELLS)
        slider = tk.Scale(ctrl, from_=0.5, to=3.0, resolution=0.25,
                          orient="horizontal", variable=self.brush_var,
                          bg="#0d0d0d", fg="#39ff14", troughcolor="#1a1a1a",
                          highlightthickness=0, bd=0, length=180)
        slider.pack(side="left", padx=8)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#0d0d0d", pady=4)
        btn_frame.pack()

        self._btn(btn_frame, "CLEAR",  self._clear,  "#ff3c3c").pack(side="left", padx=6)
        self._btn(btn_frame, "SAVE → .BIN", self._save, "#39ff14").pack(side="left", padx=6)

        # Status bar
        self.status = tk.Label(self.root, text=f"Output: {OUTPUT_PATH}",
                               font=("Courier New", 8),
                               fg="#555", bg="#0d0d0d", pady=6)
        self.status.pack()

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Courier New", 10, "bold"),
                         fg="#0d0d0d", bg=color, activebackground=color,
                         relief="flat", padx=12, pady=6, cursor="hand2")

    # ─────────────────────────────────────────────────────────────────────────
    def _bind_events(self):
        self.canvas.bind("<B1-Motion>",    self._paint)
        self.canvas.bind("<ButtonPress-1>", self._paint)

    def _pixel_to_cell(self, px, py):
        """Convert canvas pixel → (col, row) in MNIST grid."""
        return px / CANVAS_PX * GRID, py / CANVAS_PX * GRID

    def _paint(self, event):
        cx, cy = self._pixel_to_cell(event.x, event.y)
        r = self.brush_var.get()

        # Paint a soft circular brush onto the grid
        ix0 = max(0, int(cx - r - 1))
        ix1 = min(GRID, int(cx + r + 2))
        iy0 = max(0, int(cy - r - 1))
        iy1 = min(GRID, int(cy + r + 2))

        for gy in range(iy0, iy1):
            for gx in range(ix0, ix1):
                dist = ((gx + 0.5 - cx)**2 + (gy + 0.5 - cy)**2) ** 0.5
                # soft falloff: 1.0 at centre → 0.0 at radius edge
                strength = max(0.0, 1.0 - dist / r)
                self.grid[gy, gx] = min(1.0, self.grid[gy, gx] + strength * 0.6)

        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        for gy in range(GRID):
            for gx in range(GRID):
                v = self.grid[gy, gx]
                if v < 0.01:
                    continue
                brightness = int(v * 255)
                color = f"#{brightness:02x}{min(255, brightness + 40):02x}{int(brightness * 0.2):02x}"
                x0 = gx * CELL
                y0 = gy * CELL
                self.canvas.create_rectangle(x0, y0, x0 + CELL, y0 + CELL,
                                             fill=color, outline="")

    def _clear(self):
        self.grid[:] = 0.0
        self._redraw()
        self.status.config(text="Cleared.", fg="#888")

    # ─────────────────────────────────────────────────────────────────────────
    def _save(self):
        """
        Apply the same preprocessing MNIST uses, then write 784 doubles.
        Steps:
          1. Tight-crop to ink bounding box
          2. Re-pad to square with ~20% margin  (MNIST centering)
          3. Gaussian blur  (matches MNIST stroke softness)
          4. Normalize to [0, 1]
        """
        # Build a PIL image from the grid
        img_data = (self.grid * 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='L')

        # 1. Tight crop
        bin_img = img.point(lambda p: 255 if p > 25 else 0)
        bbox = bin_img.getbbox()
        if bbox is None:
            self.status.config(text="⚠  Canvas is empty — draw something first!", fg="#ff3c3c")
            return
        img = img.crop(bbox)

        # 2. Pad to square with margin
        w, h = img.size
        margin = int(max(w, h) * 0.25)
        side   = max(w, h) + 2 * margin
        square = Image.new('L', (side, side), 0)
        ox = margin + (side - 2 * margin - w) // 2
        oy = margin + (side - 2 * margin - h) // 2
        square.paste(img, (ox, oy))

        # 3. Resize to 28×28 with high-quality downsampling
        img28 = square.resize((28, 28), Image.LANCZOS)

        # 4. Gaussian blur (softens strokes to match MNIST feel)
        img28 = img28.filter(ImageFilter.GaussianBlur(radius=0.85))

        # 5. Normalize
        arr = np.array(img28, dtype=np.float64) / 255.0

        # 6. Write binary
        flat = arr.flatten()
        with open(OUTPUT_PATH, 'wb') as f:
            f.write(struct.pack(f'{len(flat)}d', *flat))

        # ASCII preview in terminal
        print("\n── 28×28 preview ──")
        chars = " ·:;+=xX$&#"
        for row in range(28):
            print("".join(chars[min(int(arr[row, col] * 10), 10)] for col in range(28)))

        self.status.config(
            text=f"✓  Saved to {OUTPUT_PATH}  —  run: ./mnist predict {OUTPUT_PATH}",
            fg="#39ff14")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = DigitPainter(root)
    root.mainloop()
