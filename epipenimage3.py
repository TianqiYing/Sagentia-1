"""
EpiPen injection stages — clean rotated final
Pipeline:
1) Draw the horizontal six-step figure WITHOUT any caption/legend (clean base).
2) Rotate the base 90° clockwise and horizontally mirror it so timesteps read T1→T6.
3) Compose a final canvas and add a NEW upright caption and legend.

Outputs:
- epipen_final_rotated_clean.png  (publication-ready)
- epipen_final_rotated_clean.svg  (bitmap-embedded SVG for convenience)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageOps


# ========= 1) DRAW CLEAN HORIZONTAL BASE (no caption/legend, no blue dots) =========
def draw_clean_horizontal_base(out_png="__epipen_base_clean.png", figsize=(7,12)):
    # Colors
    skin_c, fat_c, muscle_c = "#f9dfba", "#f7cc7a", "#e46b59"
    barrel_c, plunger_c, spring_c, needle_c = "#b1b6be", "#9aa1a9", "#1e1e1e", "k"

    # Canvas
    W, H = figsize
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Layout (vertical stacking of 6 rows)
    rows = 6
    top_margin = 0.035
    row_h = (1 - 2 * top_margin) / rows

    # Right-side continuous tissue
    tissue_x0, tissue_w = 0.72, 0.22
    skin_frac, fat_frac, muscle_frac = 0.06, 0.18, 0.76

    def draw_tissue(x0, y0, w, h):
        sw, fw, mw = w * skin_frac, w * fat_frac, w * muscle_frac
        ax.add_patch(Rectangle((x0, y0), sw, h, facecolor=skin_c, edgecolor='k', lw=1))
        ax.add_patch(Rectangle((x0 + sw, y0), fw, h, facecolor=fat_c, edgecolor='k', lw=1))
        ax.add_patch(Rectangle((x0 + sw + fw, y0), mw, h, facecolor=muscle_c, edgecolor='k', lw=1))
        ax.plot([x0 + sw, x0 + sw], [y0, y0 + h], color='k', lw=0.8)
        ax.plot([x0 + sw + fw, x0 + sw + fw], [y0, y0 + h], color='k', lw=0.8)

    draw_tissue(tissue_x0, top_margin, tissue_w, 1 - 2 * top_margin)
    skin_start = tissue_x0

    # Syringe geometry
    barrel_len, barrel_h = 0.25, 0.075
    needle_len = 0.065
    anchor_x = 0.10

    def draw_spring(x0, y, x1, amp=0.008, turns=8):
        xs = np.linspace(x0, x1, turns * 2 + 1)
        ys = y + amp * np.array([(-1) ** i for i in range(len(xs))])
        ax.plot(xs, ys, color=spring_c, lw=1.5)

    def draw_syringe(barrel_x, y, plunger_ratio):
        ax.add_patch(Rectangle((barrel_x, y - barrel_h / 2), barrel_len, barrel_h,
                               facecolor=barrel_c, edgecolor='k', lw=1))
        # flat-ended needle
        ax.plot([barrel_x + barrel_len, barrel_x + barrel_len + needle_len], [y, y],
                color=needle_c, lw=1.3)
        # plunger
        depth = barrel_len * plunger_ratio
        plunger_x = barrel_x + depth - barrel_h * 0.42
        ax.add_patch(Rectangle((plunger_x, y - barrel_h * 0.40),
                               barrel_h * 0.84, barrel_h * 0.80,
                               facecolor=plunger_c, edgecolor='k', lw=1))
        return plunger_x

    # Timesteps
    initial_gap, g3_gap = 0.03, 0.02
    barrel_x_s1 = skin_start - (barrel_len + needle_len) - initial_gap
    barrel_x_s2 = skin_start - (barrel_len + needle_len)
    barrel_x_s3 = (skin_start - g3_gap) - barrel_len
    barrel_x_s4 = skin_start - barrel_len
    plunger_ratios = [0.06, 0.08, 0.10, 0.12, 0.50, 0.85]
    positions = [
        (barrel_x_s1, plunger_ratios[0]),
        (barrel_x_s2, plunger_ratios[1]),
        (barrel_x_s3, plunger_ratios[2]),
        (barrel_x_s4, plunger_ratios[3]),
        (barrel_x_s4, plunger_ratios[4]),
        (barrel_x_s4, plunger_ratios[5]),
    ]

    for i, (bx, pr) in enumerate(positions):
        y0 = 1 - top_margin - (i + 1) * row_h
        y_mid = y0 + row_h * 0.55
        plunger_x = draw_syringe(bx, y_mid, pr)
        attach_x = bx if i < 4 else plunger_x
        draw_spring(anchor_x, y_mid, attach_x, amp=0.007 + 0.0015 * i, turns=8)
        # NOTE: blue droplets intentionally omitted

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png, (skin_c, fat_c, muscle_c)

base_png, (skin_c, fat_c, muscle_c) = draw_clean_horizontal_base()

# ========= 2) ROTATE 90° CW + MIRROR (so order reads T1→T6) =========
base = Image.open(base_png)
rot = base.rotate(-90, expand=True)      # clockwise 90°
rot_mir = ImageOps.mirror(rot)           # flip horizontally → T1…T6 left→right
rot_mir.save("__epipen_rot_mir.png", dpi=(300, 300))

# ========= 3) COMPOSE FINAL CANVAS + NEW CAPTION & LEGEND =========
fig, ax = plt.subplots(figsize=(5,2.8))  # vertical layout
ax.imshow(rot_mir)
ax.axis("off")

# Caption (upright)
caption = "Sequential stages of spring-driven injection process"
fig.text(0.5, 0.05 , caption, ha="center", va="center", fontsize=11)

# Legend (upright, bottom-center)
legend_patches = [
    mpatches.Patch(facecolor=skin_c, edgecolor="black", label="Skin"),
    mpatches.Patch(facecolor=fat_c,   edgecolor="black", label="Fat"),
    mpatches.Patch(facecolor=muscle_c,edgecolor="black", label="Muscle"),
]
ax.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    frameon=True,
    framealpha=0.95,
    edgecolor="black",
    fontsize=10,
)

# Save final outputs
plt.savefig("epipen_final_rotated_clean.png", dpi=300, bbox_inches="tight")
plt.savefig("epipen_final_rotated_clean.svg", dpi=300, bbox_inches="tight")


plt.show()
