import clip
import torch
import selectivesearch
import numpy as np
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(["car"]).to(device)

mask2 = np.reshape(np.tile(np.arange(0, 224), (224,)), (224, 224))
mask1 = mask2.T


def crop(image, candidate):
    x, y, w, h = candidate
    mask = (mask1 > y) & (mask1 < y + h) & (mask2 > x) & (mask2 < x + w)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return np.where(mask, image, 0)


image = Image.open("dog.jpg")
image = image.resize((298, 224), Image.BICUBIC)
image = image.crop((37, 0, 37 + 224, 224))
image = np.array(image)

img_lbl, regions = selectivesearch.selective_search(image)
candidates = set()
for region in regions:
    if region["rect"] in candidates:
        continue
    x, y, w, h = region["rect"]
    if w < 20 or h < 20 or w > 200 or h > 200 or w / h > 2 or h / w > 2:
        continue
    candidates.add(region["rect"])

loss0, image0 = 0, None
for i, candidate in enumerate(candidates):
    image_t = crop(image, candidate)
    image_t = Image.fromarray(image_t)
    img_t = preprocess(image_t).unsqueeze(0).to(device)
    loss, _ = clip_model(img_t, text)
    loss = loss[0][0].detach().numpy()
    print(i, loss)
    if loss > loss0:
        loss0 = loss
        image0 = image_t
image0.save(f"test/{loss0}.jpg")
