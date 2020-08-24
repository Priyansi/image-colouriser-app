import io
import torch
import torchvision.transforms as T
import torch.nn as nn
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from PIL import Image
import numpy as np


def generate_l_ab(images):
    lab = rgb2lab(images.permute(0, 2, 3, 1).cpu().numpy())
    X = lab[:, :, :, 0]
    X = X.reshape(X.shape+(1,))
    Y = lab[:, :, :, 1:] / 128
    return to_device(torch.tensor(X, dtype=torch.float).permute(0, 3, 1, 2), device), to_device(torch.tensor(Y, dtype=torch.float).permute(0, 3, 1, 2), device)


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BaseModel(nn.Module):
    def training_batch(self, batch):
        images, _ = batch
        X, Y = generate_l_ab(images)
        outputs = self.forward(X)
        loss = F.mse_loss(outputs, Y)
        return loss

    def validation_batch(self, batch):
        images, _ = batch
        X, Y = generate_l_ab(images)
        outputs = self.forward(X)
        loss = F.mse_loss(outputs, Y)
        return {'val_loss': loss.item()}

    def validation_end_epoch(self, outputs):
        epoch_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        return {'epoch_loss': epoch_loss}


class Encoder_Decoder(BaseModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2,
                      padding=get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2,
                      padding=get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2,
                      padding=get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, kernel_size=3, padding=get_padding(3)),
            nn.Upsample(size=(64, 64)),
            nn.Conv2d(128, 64, kernel_size=3, padding=get_padding(3)),
            nn.Upsample(size=(128, 128)),
            nn.Conv2d(64, 32, kernel_size=3, padding=get_padding(3)),
            nn.Conv2d(32, 16, kernel_size=3, padding=get_padding(3)),
            nn.Conv2d(16, 2, kernel_size=3, padding=get_padding(3)),
            nn.Tanh(),
            nn.Upsample(size=(256, 256))
        )

    def forward(self, images):
        return self.network(images)


def transform_image(images_bytes):
    test_transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    image = Image.open(io.BytesIO(images_bytes))
    return test_transforms(image)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model


def to_rgb(grayscale_input, ab_output, save_path=None, save_name=None):
    color_image = torch.cat((grayscale_input, ab_output),
                            0).numpy()  # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1]
    color_image[:, :, 1:3] = (color_image[:, :, 1:3]) * 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    return color_image


def get_prediction(image):
    l_img = rgb2lab(image.permute(1, 2, 0))[:, :, 0]
    l_img = torch.tensor(l_img).type(
        torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    ab_img = model(l_img)
    l_img = l_img.squeeze(0)
    ab_img = ab_img.squeeze(0)
    return to_rgb(l_img.detach(), ab_img.detach())


PATH = 'app/model.pth'
model = Encoder_Decoder()
model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.eval()
