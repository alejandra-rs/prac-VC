from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAMES = {"angry": "enfado", "disgust": "asco", "fear": "miedo", "happy": "alegría", "sad": "tristeza",
         "surprise": "sorpresa", "neutral": "neutral"}
EMOTION_BGR = {"angry": (40, 39, 214), "disgust": (75, 86, 140), "fear": (189, 103, 148), "happy": (12, 180, 232),
               "sad": (180, 119, 31), "surprise": (207, 190, 23), "neutral": (220, 220, 220)}
WHITE = (255, 255, 255)


def load_font(name, size):
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size)


FONT, FONT_BIG = load_font("arial.ttf", 16), load_font("arialbd.ttf", 24)


def wrap(text, font, max_width):
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and font.getlength(candidate) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]


@lru_cache(maxsize=None)
def text_mask(text, font, max_width):
    lines = wrap(text, font, max_width)
    step = int(font.size * 1.25)
    image = Image.new("L", (max_width, step * len(lines)), 0)
    canvas = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        canvas.text((0, step * i), line, fill=255, font=font)
    return np.asarray(image, np.float32)[..., None] / 255


def paste_text(frame, mask, x, y, color):
    region = frame[y:y + mask.shape[0], x:x + mask.shape[1]]
    mask = mask[:region.shape[0], :region.shape[1]]
    region[:] = (region * (1 - mask) + np.array(color, np.float32) * mask).astype(np.uint8)


def draw_emotion(frame, emotion):
    width = frame.shape[1]
    cv2.rectangle(frame, (width - 150, 8), (width - 8, 44), (0, 0, 0), -1)
    paste_text(frame, text_mask(NAMES[emotion], FONT_BIG, 130), width - 140, 13, EMOTION_BGR[emotion])


def draw_sentence(frame, sentence):
    height, width = frame.shape[:2]
    mask = text_mask(sentence, FONT, width - 20)
    top = height - 8 - mask.shape[0]
    frame[top - 6:] = (frame[top - 6:] * 0.35).astype(np.uint8)
    paste_text(frame, mask, 10, top, WHITE)


def draw(frame, label, emotion, sentence):
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    if emotion is not None:
        draw_emotion(frame, emotion)
    if sentence:
        draw_sentence(frame, sentence)
    return frame
