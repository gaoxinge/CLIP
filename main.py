import clip
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d

device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.float32((0.26862954, 0.26130258, 0.27577711))
STD = np.float32((0.48145466, 0.4578275, 0.40821073))

model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(["a beautiful girl play with a lovely cat in the playground"]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)


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


def blur_image(img, sigma=1.0):
    img_np = img.cpu().clone().numpy()
    img_np = gaussian_filter1d(img_np, sigma, axis=2)
    img_np = gaussian_filter1d(img_np, sigma, axis=3)
    img.copy_(torch.Tensor(img_np).type_as(img))
    return img


def jitter(img, ox, oy):
    if ox != 0:
        left = img[:, :, :, :-ox]
        right = img[:, :, :, -ox:]
        img = torch.cat([right, left], dim=3)
    if oy != 0:
        top = img[:, :, :-oy]
        bottom = img[:, :, -oy:]
        img = torch.cat([bottom, top], dim=2)
    return img


class SGD:

    def __init__(self, img, lr):
        self.img = img
        self.lr = lr

    def zero_grad(self):
        grad = self.img.grad
        if grad is not None:
            grad.zero_()

    def step(self):
        grad = self.img.grad
        self.img.data -= self.lr * grad


img = torch.rand(224, 224, 3) * 255
img = preprocess(img)
for i in range(1000):
    ox, oy = np.random.randint(0, 16, 2)
    img = jitter(img, ox, oy)

    img = Variable(img, requires_grad=True)
    optimizer = SGD(img, lr=250)
    optimizer.zero_grad()
    image_features = model.encode_image(img)
    loss = -torch.cosine_similarity(text_features, image_features, dim=1)
    print(i, loss.detach().numpy()[0])
    loss.backward()
    optimizer.step()
    img = img.data

    img = jitter(img, -ox, -oy)
    for c in range(3):
        lo = -MEAN[c] / STD[c]
        hi = (1.0 - MEAN[c]) / STD[c]
        img[:, c].clamp_(min=lo, max=hi)
    if i % 10 == 0:
        img = blur_image(img, sigma=0.5)

    if i % 30 == 0 or i == 1000 - 1:
        img0 = deprocess(img)
        img0 = img0.detach().numpy()
        img0 = Image.fromarray(img0, mode="RGB")
        img0.save(f"image/{i}.jpg", "JPEG")
