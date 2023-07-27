import clip
import numpy as np
from PIL import Image

import torch
from torchvision.models import vgg11

MEAN = np.float32((0.48145466, 0.4578275, 0.40821073))
STD = np.float32((0.26862954, 0.26130258, 0.27577711))
MEAN = torch.from_numpy(MEAN)
STD = torch.from_numpy(STD)

mask2 = torch.reshape(torch.tile(torch.arange(0, 224), (224,)), (224, 224))
mask1 = mask2.T


def convert1(coord):
    cx, cy, cw, ch = coord[0], coord[1], coord[2], coord[3]
    cx, cy, cw, ch = (cx - 112) / 224, (cy - 112) / 224, torch.log(cw / 224), torch.log(ch / 224)
    return cx, cy, cw, ch


def convert2(coord):
    cx, cy, cw, ch = coord[0], coord[1], coord[2], coord[3]
    cx, cy, cw, ch = cx * 224 + 112, cy * 224 + 112, torch.exp(cw) * 224, torch.exp(ch) * 224
    return cx, cy, cw, ch


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


def post_process0(image, coord):
    h, w, _ = image.shape
    x1, y1, x2, y2 = coord
    mask1 = np.reshape(np.repeat(np.arange(0, h), w), (h, w))
    mask2 = np.reshape(np.repeat(np.arange(0, w), h), (w, h)).T
    mask = (mask1 > y1) & (mask1 < y2) & (mask2 > x1) & (mask2 < x2)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    image = np.where(mask, image, 0)
    return image


def post_process1(image, coord):
    cx, cy, cw, ch = convert2(coord)

    x1 = torch.maximum(cx - cw / 2, torch.tensor(0.))
    y1 = torch.maximum(cy - ch / 2, torch.tensor(0.))
    x2 = torch.minimum(cx + cw / 2, torch.tensor(224.))
    y2 = torch.minimum(cy + ch / 2, torch.tensor(224.))

    mask = (mask1 > y1) & (mask1 < y2) & (mask2 > x1) & (mask2 < x2)
    mask = mask.unsqueeze(-1).repeat(1, 1, 3).permute(2, 0, 1).unsqueeze(0)

    return torch.where(mask, image, 0)


def post_process2(image, coord):
    cx, cy, cw, ch = convert2(coord)

    x1 = cx - cw / 2
    y1 = cy - ch / 2
    x2 = cx + cw / 2
    y2 = cy + ch / 2

    # x1 = torch.minimum(torch.maximum(x1, torch.tensor(0.)), torch.tensor(224.))
    # y1 = torch.minimum(torch.maximum(y1, torch.tensor(0.)), torch.tensor(224.))
    # x2 = torch.maximum(torch.minimum(x2, torch.tensor(224.)), torch.tensor(0.))
    # y2 = torch.maximum(torch.minimum(y2, torch.tensor(224.)), torch.tensor(0.))

    # a1 = torch.sign(mask1 - y1) / 2 + 0.5
    # a2 = torch.sign(y2 - mask1) / 2 + 0.5
    # a3 = torch.sign(mask2 - x1) / 2 + 0.5
    # a4 = torch.sign(x2 - mask2) / 2 + 0.5

    # a1 = torch.tanh(mask1 - y1) / 2 + 0.5
    # a2 = torch.tanh(y2 - mask1) / 2 + 0.5
    # a3 = torch.tanh(mask2 - x1) / 2 + 0.5
    # a4 = torch.tanh(x2 - mask2) / 2 + 0.5

    a1 = torch.sigmoid(mask1 - y1)
    a2 = torch.sigmoid(y2 - mask1)
    a3 = torch.sigmoid(mask2 - x1)
    a4 = torch.sigmoid(x2 - mask2)

    return image * a1 * a2 * a3 * a4


class ODModel(torch.nn.Module):

    def __init__(self):
        super(ODModel, self).__init__()
        vgg_model = vgg11()
        self.vgg_model_features = vgg_model.features
        self.layer = torch.nn.Linear(in_features=25088, out_features=4)

    def __call__(self, x):
        x = self.vgg_model_features(x)
        x = torch.flatten(x)
        x = self.layer(x)
        return x


def draw_numpy():
    image = Image.open("dog.jpg")
    image = np.array(image)
    image = post_process0(image, [122, 222, 319, 542])
    image = Image.fromarray(image)
    image.save("image/test_0.jpg")


def draw_torch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    clip_model, _ = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(["dog"]).to(device)

    image = Image.open("dog.jpg")
    image = image.resize((298, 224), Image.BICUBIC)
    image = image.crop((37, 0, 37 + 224, 224))
    image = np.array(image)
    image = torch.from_numpy(image)
    image = preprocess(image)

    coords = [
        None,
        [
            (122 + 319) / 2 / 768 * 298 - 37,
            (222 + 542) / 2 / 576 * 224,
            (319 - 122) / 768 * 298,
            (542 - 222) / 576 * 224
        ],
        [
            (122 + 768) / 2 / 768 * 298 - 37,
            (222 + 542) / 2 / 576 * 224,
            (768 - 122) / 768 * 298,
            (542 - 222) / 576 * 224
        ],
        [
            (122 + 319) / 2 / 768 * 298 - 37,
            (222 + 576) / 2 / 576 * 224,
            (319 - 122) / 768 * 298,
            (576 - 222) / 576 * 224
        ],
        [
            (0 + 319) / 2 / 768 * 298 - 37,
            (222 + 542) / 2 / 576 * 224,
            (319 - 0) / 768 * 298,
            (542 - 222) / 576 * 224
        ],
        [
            (122 + 319) / 2 / 768 * 298 - 37,
            (0 + 542) / 2 / 576 * 224,
            (319 - 122) / 768 * 298,
            (542 - 0) / 576 * 224
        ],
        [
            (122 + 319) / 2 / 768 * 298 - 37,
            (222 + 542) / 2 / 576 * 224,
            (319 - 122 + 10) / 768 * 298,
            (542 - 222 + 10) / 576 * 224
        ],
        [
            (122 + 319) / 2 / 768 * 298 - 37,
            (222 + 542) / 2 / 576 * 224,
            (319 - 122 + 20) / 768 * 298,
            (542 - 222 + 20) / 576 * 224
        ],
        [
            (122 + 319) / 2 / 768 * 298 - 37,
            (222 + 542) / 2 / 576 * 224,
            (319 - 122 + 30) / 768 * 298,
            (542 - 222 + 30) / 576 * 224
        ],
        [
            (122 + 319) / 2 / 768 * 298 - 37,
            (222 + 542) / 2 / 576 * 224,
            (319 - 122 + 40) / 768 * 298,
            (542 - 222 + 40) / 576 * 224
        ],
        [
            (122 + 319) / 2 / 768 * 298 - 37,
            (222 + 542) / 2 / 576 * 224,
            (319 - 122 + 50) / 768 * 298,
            (542 - 222 + 50) / 576 * 224
        ],
        [
            (122 + 319) / 2 / 768 * 298 - 37,
            (222 + 542) / 2 / 576 * 224,
            (319 - 122 + 60) / 768 * 298,
            (542 - 222 + 60) / 576 * 224
        ],
    ]

    for i, coord in enumerate(coords):
        if coord is None:
            image_t = image
        else:
            coord = torch.tensor(coord)
            coord = convert1(coord)
            image_t = post_process2(image, coord)
        loss, _ = clip_model(image_t, text)
        loss = loss[0][0].detach().numpy()
        image_t = deprocess(image_t)
        image_t = image_t.numpy()
        image_t = Image.fromarray(image_t)
        image_t.save(f"image/test_{i}_{loss:.2f}.jpg")


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    clip_model, _ = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(["dog"]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)

    od_model = ODModel()
    od_model = od_model.to(device)

    img = Image.open("dog.jpg")
    image = img.resize((298, 224), Image.BICUBIC)
    image = image.crop((37, 0, 37 + 224, 224))
    image = np.array(image)
    image = torch.from_numpy(image)
    image = preprocess(image)

    image1 = deprocess(image)
    image1 = image1.numpy()
    image1 = Image.fromarray(image1)
    image1.save(f"image/test.jpg")

    optimizer = torch.optim.SGD(od_model.parameters(), lr=1e-2)

    for i in range(50):
        coord = od_model(image)
        loss = (coord[0]) ** 2 + (coord[1]) ** 2 + (coord[2] + 0.1) ** 2 + (coord[3] + 0.1) ** 2
        print(i, loss.detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            image0 = post_process2(image, coord)
            image1 = deprocess(image0)
            image1 = image1.numpy()
            image1 = Image.fromarray(image1)
            image1.save(f"image/1_{i}.jpg")

    optimizer = torch.optim.SGD(od_model.parameters(), lr=1e-4)

    for i in range(1000):
        coord = od_model(image)
        image0 = post_process2(image, coord)

        # image_features = clip_model.encode_image(image0)
        # loss = -torch.cosine_similarity(image_features, text_features, dim=1)
        loss, _ = clip_model(image0, text)
        loss = -loss
        print(i, loss.detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            image1 = deprocess(image0)
            image1 = image1.numpy()
            image1 = Image.fromarray(image1)
            image1.save(f"image/2_{i}.jpg")


if __name__ == "__main__":
    # draw_numpy()
    # draw_torch()
    train()
