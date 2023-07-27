import clip
import numpy as np
from PIL import Image

import torch

MEAN = np.float32((0.48145466, 0.4578275, 0.40821073))
STD = np.float32((0.26862954, 0.26130258, 0.27577711))
MEAN = torch.from_numpy(MEAN)
STD = torch.from_numpy(STD)

mask2 = torch.reshape(torch.tile(torch.arange(0, 224), (224,)), (224, 224))
mask1 = mask2.T


def convert1(cx, cy, cw, ch):
    return (cx - 112) / 224, (cy - 112) / 224, torch.log(cw / 224), torch.log(ch / 224)


def convert2(cx, cy, cw, ch):
    return cx * 224 + 112, cy * 224 + 112, torch.exp(cw) * 224, torch.exp(ch) * 224


def preprocess(img):
    img = img.to(torch.float32)
    img /= 255
    img = (img - MEAN) / STD
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    return img


def deprocess(img):
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = img * STD + MEAN
    img = img * 255
    img = img.to(torch.uint8)
    return img


def post_process(image, cx, cy, cw, ch):
    cx, cy, cw, ch = convert2(cx, cy, cw, ch)

    x1 = cx - cw / 2
    y1 = cy - ch / 2
    x2 = cx + cw / 2
    y2 = cy + ch / 2

    a1 = torch.sigmoid(mask1 - y1)
    a2 = torch.sigmoid(y2 - mask1)
    a3 = torch.sigmoid(mask2 - x1)
    a4 = torch.sigmoid(x2 - mask2)

    return image * a1 * a2 * a3 * a4


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

clip_model, _ = clip.load("ViT-B/32", device=device)
text = clip.tokenize(["dog"]).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text)


img = Image.open("dog.jpg")
image = img.resize((298, 224), Image.BICUBIC)
image = image.crop((37, 0, 37 + 224, 224))
image = np.array(image)
image = torch.from_numpy(image)
image = preprocess(image)

cx = torch.tensor([0], dtype=torch.float32)
cy = torch.tensor([0], dtype=torch.float32)
cw = torch.tensor([-0.1], dtype=torch.float32)
ch = torch.tensor([-0.1], dtype=torch.float32)
cx = torch.autograd.Variable(cx, requires_grad=True)
cy = torch.autograd.Variable(cy, requires_grad=True)
cw = torch.autograd.Variable(cw, requires_grad=True)
ch = torch.autograd.Variable(ch, requires_grad=True)

optimizer = torch.optim.SGD([cx, cy, cw, ch], lr=1e-3)

for i in range(1000):
    image0 = post_process(image, cx, cy, cw, ch)
    loss, _ = clip_model(image0, text)
    loss = -loss
    print(i, loss.detach().numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        image1 = deprocess(image0)
        image1 = image1.numpy()
        image1 = Image.fromarray(image1)
        image1.save(f"image/{i}.jpg")
