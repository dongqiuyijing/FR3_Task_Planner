from pathlib import Path

import cv2

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ==============================
# 1. 标定板参数
# ==============================

COLS = 8
ROWS = 6

SQUARE_M = 0.020
MARKER_M = 0.015

DPI = 600

OUTPUT_DIR = Path(__file__).resolve().parent

STEM = "charuco_8x6_20mm_15mm"


# ==============================
# 2. 创建真正的 ChArUco 标定板
# ==============================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

board = cv2.aruco.CharucoBoard(
    (COLS, ROWS),
    SQUARE_M,
    MARKER_M,
    dictionary
)


# ==============================
# 3. 计算图片分辨率并生成 PNG
# ==============================

width_mm = COLS * SQUARE_M * 1000
height_mm = ROWS * SQUARE_M * 1000

width_px = round(width_mm / 25.4 * DPI)
height_px = round(height_mm / 25.4 * DPI)

image = board.generateImage(
    (width_px, height_px),
    marginSize=0,
    borderBits=1
)

assert image.shape == (height_px, width_px)

png_path = OUTPUT_DIR / f"{STEM}_600dpi.png"

if not cv2.imwrite(str(png_path), image):
    raise RuntimeError("Could not write PNG")


# ==============================
# 4. 自动验证生成的标定板
# ==============================

detector = cv2.aruco.CharucoDetector(board)

corners, ids, marker_corners, marker_ids = (
    detector.detectBoard(image)
)

expected_markers = (COLS * ROWS) // 2
expected_corners = (COLS - 1) * (ROWS - 1)

if marker_ids is None or len(marker_ids) != expected_markers:
    raise RuntimeError("ArUco 标记验证失败")

if ids is None or len(ids) != expected_corners:
    raise RuntimeError("ChArUco 角点验证失败")

if sorted(marker_ids.reshape(-1).tolist()) != sorted(
    board.getIds().reshape(-1).tolist()
):
    raise RuntimeError("ArUco ID 与生成的标定板不一致")


# ==============================
# 5. 生成实际物理尺寸的 A4 PDF
# ==============================

pdf_path = OUTPUT_DIR / f"{STEM}_A4.pdf"

pdf = canvas.Canvas(
    str(pdf_path),
    pagesize=A4
)

page_w, page_h = A4

board_w = width_mm * mm
board_h = height_mm * mm

x = (page_w - board_w) / 2
y = (page_h - board_h) / 2

pdf.drawImage(
    ImageReader(str(png_path)),
    x,
    y,
    width=board_w,
    height=board_h,
    preserveAspectRatio=False
)

pdf.showPage()
pdf.save()


# ==============================
# 6. 输出结果
# ==============================

print("OpenCV:", cv2.__version__)

print(
    f"棋盘尺寸：{width_mm:g} × {height_mm:g} mm"
)

print(
    f"图片分辨率：{width_px} × {height_px}"
)

print(
    f"验证通过：{len(marker_ids)} 个 ArUco 标记，"
    f"{len(ids)} 个 ChArUco 角点"
)

print("PNG：", png_path)
print("PDF：", pdf_path)