"""
paint_digit.py  —  Draw a digit, export MNIST-ready .bin
Requires:  pip install pygame pillow numpy

Run:
    python3 paint_digit.py              # saves to input_image.bin
    python3 paint_digit.py my4.bin      # custom output path
"""

import sys, struct
import numpy as np
from PIL import Image, ImageFilter
import pygame

OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "input_image.bin"

GRID      = 28
CELL      = 20          # pixels per grid cell  (window = 560px)
WIN_SIZE  = GRID * CELL
PANEL_H   = 80

BG        = (13,  13,  13)
PANEL_BG  = (20,  20,  20)
BTN_SAVE  = (30, 210,  80)
BTN_CLEAR = (210,  50,  50)
BTN_TEXT  = (10,  10,  10)
LABEL_COL = (100, 100, 100)
GREEN     = (57, 255,  20)

def lerp_color(v):
    b = int(v * 255)
    return (b, min(255, b + 30), int(b * 0.7))

def save_bin(grid):
    img_data = (grid * 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode='L')

    thresh = img.point(lambda p: 255 if p > 25 else 0)
    bbox = thresh.getbbox()
    if bbox is None:
        return False, "Canvas is empty!"
    img = img.crop(bbox)

    w, h = img.size
    margin = int(max(w, h) * 0.25)
    side   = max(w, h) + 2 * margin
    square = Image.new('L', (side, side), 0)
    ox = margin + (side - 2*margin - w) // 2
    oy = margin + (side - 2*margin - h) // 2
    square.paste(img, (ox, oy))

    img28 = square.resize((28, 28), Image.LANCZOS)
    img28 = img28.filter(ImageFilter.GaussianBlur(radius=0.85))
    arr   = np.array(img28, dtype=np.float64) / 255.0

    with open(OUTPUT_PATH, 'wb') as f:
        f.write(struct.pack(f'{784}d', *arr.flatten()))

    chars = " ·:;+=xX$&#"
    print(f"\n── 28×28 preview ({OUTPUT_PATH}) ──")
    for row in range(28):
        print("".join(chars[min(int(arr[row,c]*10), 10)] for c in range(28)))

    return True, f"Saved → {OUTPUT_PATH}   |   run: ./mnist predict {OUTPUT_PATH}"

def draw_button(surf, rect, label, color):
    pygame.draw.rect(surf, color, rect, border_radius=6)
    font = pygame.font.SysFont("monospace", 14, bold=True)
    txt  = font.render(label, True, BTN_TEXT)
    surf.blit(txt, txt.get_rect(center=rect.center))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_SIZE, WIN_SIZE + PANEL_H))
    pygame.display.set_caption("Digit Painter → MNIST .bin")

    grid       = np.zeros((GRID, GRID), dtype=np.float32)
    brush_r    = 1.5
    drawing    = False
    status_msg = f"Output: {OUTPUT_PATH}"
    status_ok  = True

    grid_surf = pygame.Surface((WIN_SIZE, WIN_SIZE))

    btn_clear = pygame.Rect(16,             WIN_SIZE + 20, 110, 40)
    btn_save  = pygame.Rect(WIN_SIZE - 200, WIN_SIZE + 20, 184, 40)
    btn_minus = pygame.Rect(WIN_SIZE//2 - 70, WIN_SIZE + 22, 32, 36)
    btn_plus  = pygame.Rect(WIN_SIZE//2 + 38, WIN_SIZE + 22, 32, 36)

    def paint(px, py):
        cx, cy = px / CELL, py / CELL
        r = brush_r
        ix0 = max(0, int(cx - r - 1))
        ix1 = min(GRID, int(cx + r + 2))
        iy0 = max(0, int(cy - r - 1))
        iy1 = min(GRID, int(cy + r + 2))
        for gy in range(iy0, iy1):
            for gx in range(ix0, ix1):
                dist = ((gx + 0.5 - cx)**2 + (gy + 0.5 - cy)**2) ** 0.5
                strength = max(0.0, 1.0 - dist / r)
                grid[gy, gx] = min(1.0, grid[gy, gx] + strength * 0.55)

    def redraw_grid():
        grid_surf.fill(BG)
        for gy in range(GRID):
            for gx in range(GRID):
                v = grid[gy, gx]
                if v < 0.01:
                    continue
                pygame.draw.rect(grid_surf, lerp_color(v),
                                 (gx*CELL, gy*CELL, CELL, CELL))

    font_sm = pygame.font.SysFont("monospace", 12)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_clear.collidepoint(event.pos):
                    grid[:] = 0.0
                    status_msg, status_ok = "Cleared.", True
                elif btn_save.collidepoint(event.pos):
                    status_ok, status_msg = save_bin(grid)
                elif btn_minus.collidepoint(event.pos):
                    brush_r = max(0.5, round(brush_r - 0.5, 1))
                elif btn_plus.collidepoint(event.pos):
                    brush_r = min(4.0, round(brush_r + 0.5, 1))
                elif event.pos[1] < WIN_SIZE:
                    drawing = True
                    paint(*event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing and event.pos[1] < WIN_SIZE:
                    paint(*event.pos)

        redraw_grid()
        screen.blit(grid_surf, (0, 0))

        pygame.draw.rect(screen, PANEL_BG, (0, WIN_SIZE, WIN_SIZE, PANEL_H))
        pygame.draw.line(screen, (40,40,40), (0, WIN_SIZE), (WIN_SIZE, WIN_SIZE), 1)

        draw_button(screen, btn_clear, "CLEAR",      BTN_CLEAR)
        draw_button(screen, btn_save,  "SAVE → .BIN", BTN_SAVE)
        draw_button(screen, btn_minus, "−", (60,60,60))
        draw_button(screen, btn_plus,  "+", (60,60,60))

        blabel = font_sm.render(f"brush: {brush_r:.1f}", True, LABEL_COL)
        screen.blit(blabel, (WIN_SIZE//2 - blabel.get_width()//2, WIN_SIZE + 62))

        stxt = font_sm.render(status_msg, True, GREEN if status_ok else (255,80,80))
        screen.blit(stxt, (16, WIN_SIZE + PANEL_H - 18))

        pygame.display.flip()
        pygame.time.Clock().tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
