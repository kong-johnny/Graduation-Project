from paddleocr import PaddleOCR, draw_ocr
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='ch', ocr_version='PP-OCRv4')

def optic_ch_reg(img: np.ndarray):
    result = ocr.ocr(img, cls=True)
    return result