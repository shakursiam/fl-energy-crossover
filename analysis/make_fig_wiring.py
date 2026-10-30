"""Wiring schematic of the two-board ESP32-S3 testbed (DUT + logger + INA238).

Output: figures/fig_wiring.pdf (vector) and figures/fig_wiring.png (preview).
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})

C_PWR, C_GND, C_I2C, C_MRK, C_USB = '#c0392b', '#000000', '#1f5fa8', '#1b7f3b', '#7b3fa0'
LW = 0.9

fig, ax = plt.subplots(figsize=(7.16, 4.7))
ax.set_xlim(-1, 163); ax.set_ylim(-1, 104); ax.set_aspect('equal'); ax.axis('off')


def wire(pts, c, ls='-', lw=LW):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=c, ls=ls, lw=lw, solid_capstyle='round', solid_joinstyle='round', zorder=2)


def dot(x, y, c):
    ax.plot(x, y, 'o', ms=2.6, color=c, zorder=4)


def pin(x, y, c='k'):
    ax.plot(x, y, 's', ms=2.4, mfc='white', mec=c, mew=0.7, zorder=5)


def ground(x, y):
    for i, w in enumerate((3.0, 2.0, 1.0)):
        ax.plot([x - w / 2, x + w / 2], [y - i * 0.9] * 2, color=C_GND, lw=LW, zorder=3)


def resistor_v(x, y_top, y_bot, c, n=6):
    """Vertical zig-zag resistor between y_top and y_bot (body in the middle 60 %)."""
    L = y_top - y_bot; b0, b1 = y_top - 0.2 * L, y_bot + 0.2 * L
    pts = [(x, y_top), (x, b0)]
    for i in range(n):
        pts.append((x + (0.9 if i % 2 == 0 else -0.9), b0 - (i + 0.5) * (b0 - b1) / n))
    pts += [(x, b1), (x, y_bot)]
    wire(pts, c)


def resistor_h(x0, x1, y, c, n=6):
    L = x1 - x0; b0, b1 = x0 + 0.2 * L, x1 - 0.2 * L
    pts = [(x0, y), (b0, y)]
    for i in range(n):
        pts.append((b0 + (i + 0.5) * (b1 - b0) / n, y + (0.9 if i % 2 == 0 else -0.9)))
    pts += [(b1, y), (x1, y)]
    wire(pts, c)


def board(x0, y0, w, h, title, sub):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                fc='#eef3ee', ec='k', lw=1.0, zorder=1))
    ax.text(x0 + w / 2, y0 + h - 4.2, title, ha='center', va='center', fontweight='bold', fontsize=6.6)
    ax.text(x0 + w / 2, y0 + h - 7.6, sub, ha='center', va='center', fontsize=6.3, style='italic')


# ------------------------------------------------------------------ boards
AX0, AX1, BX0, BX1, BY0, BY1 = 38, 74, 124, 160, 24, 82
board(AX0, BY0, AX1 - AX0, BY1 - BY0, 'Board A (DUT)', 'ESP32-S3-DevKitC-1')
board(BX0, BY0, BX1 - BX0, BY1 - BY0, 'Board B (logger)', 'ESP32-S3-DevKitC-1')
for mx in (44, 140):
    ax.add_patch(Rectangle((mx, 43), 14, 20, fc='#d5d8dc', ec='#555', lw=0.7, zorder=1.5))
    ax.text(mx + 7, 53, 'ESP32-S3\nWROOM-1', ha='center', va='center', fontsize=5.8, linespacing=1.1)

# USB connectors on the board tops
UA, UB = 56, 142
for ux in (UA, UB):
    ax.add_patch(Rectangle((ux - 3, BY1 - 0.5), 6, 2.6, fc='#bbbbbb', ec='k', lw=0.7, zorder=3))

# ------------------------------------------------------------------ phase-marker bus with pull-downs
mark_y = [68, 62.5, 57, 51.5, 46]
a_pins = ['GPIO10', 'GPIO11', 'GPIO12', 'GPIO13', 'GPIO14']
b_pins = ['GPIO6', 'GPIO7', 'GPIO8', 'GPIO9', 'GPIO18']
names = ['TRAIN', 'COMM', 'PHASE b0', 'PHASE b1', 'PHASE b2']
res_x = [82, 87.5, 93, 98.5, 104]
GND_Y = 31
for y, pa, pb, nm, rx in zip(mark_y, a_pins, b_pins, names, res_x):
    wire([(AX1, y), (BX0, y)], C_MRK)
    pin(AX1, y, C_MRK); pin(BX0, y, C_MRK)
    ax.text(AX1 - 1.3, y, pa, ha='right', va='center', fontsize=6)
    ax.text(BX0 + 1.3, y, pb, ha='left', va='center', fontsize=6)
    ax.text(114.5, y + 1.1, nm, ha='center', va='bottom', fontsize=5.8, color=C_MRK)
    dot(rx, y, C_MRK)
    wire([(rx, y), (rx, 42)], C_MRK)
    resistor_v(rx, 42, 34, C_MRK)
    wire([(rx, 34), (rx, GND_Y)], C_GND)
    dot(rx, GND_Y, C_GND)
ax.text(107, 38, r'$5\times100\,\mathrm{k\Omega}$' + '\npull-down', ha='left', va='center', fontsize=6, linespacing=1.15)

# shared ground between the boards
wire([(AX1, GND_Y), (BX0, GND_Y)], C_GND)
pin(AX1, GND_Y); pin(BX0, GND_Y)
ax.text(AX1 - 1.3, GND_Y, 'GND', ha='right', va='center', fontsize=6)
ax.text(BX0 + 1.3, GND_Y, 'GND', ha='left', va='center', fontsize=6)
wire([(116, GND_Y), (116, GND_Y - 2.5)], C_GND); ground(116, GND_Y - 2.5)

# ------------------------------------------------------------------ bench supply
ax.add_patch(FancyBboxPatch((3, 46), 26, 16, boxstyle='round,pad=0,rounding_size=1', fc='#fbeeee', ec='k', lw=1.0, zorder=1))
ax.text(16, 57.5, 'Regulated DC\nsupply, 5.0 V', ha='center', va='center', fontsize=6.5, fontweight='bold', linespacing=1.15)
ax.text(9, 48.6, '+', ha='center', va='center', fontsize=8, color=C_PWR, fontweight='bold')
ax.text(24.5, 50.5, '−', ha='center', va='center', fontsize=8)
pin(9, 46, C_PWR); pin(29, 50.5)
wire([(29, 50.5), (33, 50.5), (33, 47)], C_GND); ground(33, 47)

# ------------------------------------------------------------------ INA238
IX0, IX1, IY0, IY1 = 3, 37, 2, 22
ax.add_patch(FancyBboxPatch((IX0, IY0), IX1 - IX0, IY1 - IY0, boxstyle='round,pad=0,rounding_size=1', fc='#eaf0f8', ec='k', lw=1.0, zorder=1))
ax.text(13, 8.6, 'TI INA238', ha='center', va='center', fontweight='bold', fontsize=7)
ax.text(13, 5.3, 'I²C addr. 0x40', ha='center', va='center', fontsize=5.8)
# power path: supply + -> IN+ -> shunt -> IN- -> Board A 5V
INP, INM = 9, 21
wire([(9, 46), (INP, IY1)], C_PWR, lw=1.3)
pin(INP, IY1, C_PWR); pin(INM, IY1, C_PWR)
ax.text(INP, IY1 - 1.5, 'IN+', ha='center', va='top', fontsize=5.6)
ax.text(INM, IY1 - 1.5, 'IN−', ha='center', va='top', fontsize=5.6)
wire([(INP, IY1), (INP, 15.5)], C_PWR, lw=1.3)
resistor_h(INP, INM, 15.5, C_PWR)
wire([(INM, 15.5), (INM, IY1)], C_PWR, lw=1.3)
ax.text(15, 12.6, r'$R_{\mathrm{sh}}=15\,\mathrm{m\Omega}$', ha='center', va='center', fontsize=5.8)
wire([(INM, IY1), (INM, 38), (AX0, 38)], C_PWR, lw=1.3)
dot(INM, 34.7, C_PWR)
ax.text(INM + 1.0, 34.7, 'VBUS sense', ha='left', va='center', fontsize=5.4, color=C_PWR)
pin(AX0, 38, C_PWR)
ax.text(AX0 + 1.3, 38, '5V', ha='left', va='center', fontsize=6)
ax.text(28, 39.6, 'high-side', ha='center', va='bottom', fontsize=5.6, color=C_PWR, style='italic')
# Board A ground
pin(AX0, 31.5)
ax.text(AX0 + 1.3, 31.5, 'GND', ha='left', va='center', fontsize=6)
wire([(AX0, 31.5), (34.5, 31.5), (34.5, 29)], C_GND); ground(34.5, 29)

# I2C + supply of the INA238 from Board B (bottom pins)
i2c = [('VCC', '3V3', 19.0, 129, C_PWR), ('GND', 'GND', 14.5, 135, C_GND),
       ('SDA', 'GPIO4', 10.0, 141, C_I2C), ('SCL', 'GPIO5', 5.5, 147, C_I2C)]
for lab_i, lab_b, y, bx, c in i2c:
    c_line = C_I2C if c == C_I2C else c
    wire([(IX1, y), (bx, y), (bx, BY0)], c_line)
    pin(IX1, y, c_line); pin(bx, BY0, c_line)
    ax.text(IX1 - 1.3, y, lab_i, ha='right', va='center', fontsize=5.8)
    ax.text(bx, BY0 + 1.4, lab_b, ha='center', va='bottom', fontsize=5.4, rotation=90)
ax.text(80, 10.0 + 0.8, 'SDA', ha='center', va='bottom', fontsize=5.6, color=C_I2C)
ax.text(80, 5.5 + 0.8, 'SCL (I²C, 400 kHz)', ha='center', va='bottom', fontsize=5.6, color=C_I2C)
ax.text(80, 19.0 + 0.8, '3.3 V logic supply', ha='center', va='bottom', fontsize=5.6, color=C_PWR)
dot(110, 14.5, C_GND); wire([(110, 14.5), (110, 12.5)], C_GND); ground(110, 12.5)

# ------------------------------------------------------------------ host PC + USB
ax.add_patch(FancyBboxPatch((76, 91), 50, 12, boxstyle='round,pad=0,rounding_size=1', fc='#f4f4f4', ec='k', lw=1.0, zorder=1))
ax.text(101, 98.8, 'Host PC', ha='center', va='center', fontsize=6.6, fontweight='bold')
ax.text(101, 95.0, '(flashing, control, data capture)', ha='center', va='center', fontsize=6.0)
wire([(UA, BY1 + 2.1), (UA, 97), (76, 97)], C_USB, ls=(0, (3, 1.6)), lw=1.1)
ax.text(UA + 1.5, 89.0, 'data-only USB\n(VBUS cut)', ha='left', va='center', fontsize=5.8, color=C_USB, linespacing=1.15)
wire([(UB, BY1 + 2.1), (UB, 97), (126, 97)], C_USB, lw=1.1)
ax.text(UB + 1.5, 89.5, 'USB: 5 V power\n+ serial log stream', ha='left', va='center', fontsize=5.8, color=C_USB, linespacing=1.15)

# ------------------------------------------------------------------ legend
handles = [Line2D([], [], color=C_PWR, lw=1.3, label='DUT power path (5 V)'),
           Line2D([], [], color=C_MRK, lw=LW, label='GPIO phase markers'),
           Line2D([], [], color=C_I2C, lw=LW, label='I²C'),
           Line2D([], [], color=C_GND, lw=LW, label='common ground'),
           Line2D([], [], color=C_USB, lw=1.1, ls=(0, (3, 1.6)), label='USB, data only'),
           Line2D([], [], color=C_USB, lw=1.1, label='USB, power + data')]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=5.8,
          handlelength=2.2, labelspacing=0.3, borderaxespad=0.2)

fd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
plt.savefig(os.path.join(fd, 'fig_wiring.pdf'), bbox_inches='tight', pad_inches=0.02)
plt.savefig(os.path.join(fd, 'fig_wiring.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
print('saved', os.path.abspath(fd))
