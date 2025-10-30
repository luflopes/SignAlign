import torch
import torch.nn.functional as F
import random
import numpy as np
import os
from PIL import Image



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def cosine_similarity(a, b):
    return F.cosine_similarity(a, b)


def paste_center_on_canvas(img: Image.Image, canvas_size: int = 224, background=(255,255,255)):
    img = img.convert("RGBA")
    w, h = img.size
    scale = min(canvas_size / w, canvas_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), color=background + (255,))
    offset = ((canvas_size - new_w) // 2, (canvas_size - new_h) // 2)
    canvas.paste(img_resized, offset, img_resized)
    return canvas.convert("RGB")


