import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tqdm
import collections
import os
from scipy.ndimage.filters import maximum_filter

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import timm

from findpeaks import findpeaks

DATA_PATH = './data'
X_train = np.load(f'{DATA_PATH}/X_train.npy').reshape(-1, 20, 20)
y_train = np.load(f'{DATA_PATH}/y_train.npy')
X_test = np.load(f'{DATA_PATH}/X_test.npy').reshape(-1, 20, 20)


class DiceNumCounter():
    def __init__(self, high_intensity=200, dice_area=50):
        self.dice_area = dice_area
        self.high_intensity = high_intensity

    def count(self, image):
        dice_area_pix_count = np.sum(image > self.high_intensity)
        if dice_area_pix_count <= self.dice_area:
            return 1
        elif dice_area_pix_count <= self.dice_area * 2:
            return 2
        else:
            return 3


class DiceAreaDetector():
    def __init__(self, image_size=(20, 20), extract_image_size=10, conv_kernel_size=9, conv_inv=False):
        # 入力画像サイズ
        self.image_size = image_size
        # 出力画像サイズ(サイコロ領域抽出後の画像サイズ)
        self._extract_image_size = extract_image_size
        # 中心検出のために畳み込み演算をする円形カーネルの作成→畳み込み関数の定義
        x, y = np.meshgrid(np.arange(-0.5, 0.6, 1 / (conv_kernel_size - 1)),
                           np.arange(-0.5, 0.6, 1 / (conv_kernel_size - 1)))
        self._conv_kernel_array = np.sqrt(x ** 2 + y ** 2) if conv_inv is False else np.sqrt(x ** 2 + y ** 2)
        self._conv_kernel_array[0, 0] = 0;
        self._conv_kernel_array[1, 0] = 0;
        self._conv_kernel_array[0, 1] = 0
        self._conv_kernel_array[0, -1] = 0;
        self._conv_kernel_array[0, -2] = 0;
        self._conv_kernel_array[1, -1] = 0
        self._conv_kernel_array[-1, 0] = 0;
        self._conv_kernel_array[-2, 0] = 0;
        self._conv_kernel_array[-1, 1] = 0
        self._conv_kernel_array[-1, -1] = 0;
        self._conv_kernel_array[-1, -2] = 0;
        self._conv_kernel_array[-2, -1] = 0
        self._conv_kernel_tensor = torch.tensor(self._conv_kernel_array).unsqueeze(0).unsqueeze(0)
        self._conv = nn.Conv2d(20, 1, (conv_kernel_size, conv_kernel_size), bias=False, padding=conv_kernel_size // 2)
        self._conv.weight.data = self._conv_kernel_tensor
        # ピーク検出器の定義(OSS)
        self._peak_finder = findpeaks(method='topology', whitelist=['peak'], scale=True, denoise='fastnl',
                                      params={"window": 2},
                                      togray=True, imsize=image_size, verbose=0)
        # 検出結果保存リスト
        self._processed_images = []

    def detect(self, image, dice_num):
        # 検出結果画像のリセット
        self._processed_images = []
        # 入力画像の保存
        self._processed_images.append(image.copy())
        # 枠画像の抽出
        self._contour_image = self._find_contour(image)
        self._processed_images.append(self._contour_image.copy())
        # 円形画像カーネルの畳み込み演算の実施
        contour_image_tensor = torch.tensor(self._contour_image.astype(float)).unsqueeze(0).unsqueeze(0)
        self._conved_image = self._conv(contour_image_tensor).squeeze().detach().numpy()
        self._processed_images.append(self._conv_kernel_array.copy())
        self._processed_images.append(self._conved_image.copy())
        # ピークの抽出(畳こみ結果からピークを検出する.検出方法はOSSを利用)
        self._all_peak_info = self._peak_finder.fit(self._conved_image)
        self._processed_images.append(self._all_peak_info['Xranked'].copy())
        # 検出結果の抽出
        # サイコロの数に応じて(サイコロの中心座標リスト, サイコロ画像リスト)を抽出
        peaks, detected_images = self._extract_peaks(image, self._all_peak_info, dice_num)
        for i in range(dice_num):
            self._processed_images.append(detected_images[i].copy())

        return peaks, detected_images

    def _find_contour(self, image):
        im_clone = image.astype(np.uint8)
        im_clone = cv2.GaussianBlur(im_clone, (3, 3), 0)
        _, im_thresh = cv2.threshold(im_clone, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        im_clone1 = np.zeros(im_clone.shape)
        for i in range(len(contours)):
            contours_area = cv2.contourArea(contours[i])
            x, y, w, h = cv2.boundingRect(contours[i])
            dice_roi = im_clone1[y: (y + h), x: (x + w)]
        result_img = cv2.drawContours(im_clone1, contours, -1, (255, 255, 255), 1)
        return result_img

    def _extract_peaks(self, image, all_peak_info, peak_num):
        # 中心画像の取得
        dice_images = []
        peaks = []
        for i in range(peak_num):
            peak = [np.where(all_peak_info['Xranked'] == i + 1)[0][0],
                    np.where(all_peak_info['Xranked'] == i + 1)[1][0]]
            peaks.append(peak)
            x_slice = self._decice_crop_area(self.image_size[0], self._extract_image_size // 2, peak[0])
            y_slice = self._decice_crop_area(self.image_size[1], self._extract_image_size // 2, peak[1])
            dice_images.append(image[x_slice, y_slice])
        return peaks, dice_images

    def _decice_crop_area(self, img_size, extract_size, peak):
        if (peak - extract_size) < 0:
            sl = slice(0, extract_size * 2)
        elif img_size < (peak + extract_size):
            sl = slice(img_size - extract_size * 2, img_size)
        else:
            sl = slice(int(peak - extract_size), int(peak + extract_size))
        return sl

    def _show_result(self):
        title_list = ['input', 'contour_image', 'kernel_image', 'conved_image', 'peak_image', 'output1', 'output2',
                      'output3']
        fig = plt.figure(figsize=[20, 160])
        for i in range(0, 8):
            ax = fig.add_subplot(1, 8, i + 1);
            plt.title(title_list[i])
            if i < len(self._processed_images):
                ax.imshow(self._processed_images[i])
            else:
                ax.imshow(np.zeros([self._extract_image_size, self._extract_image_size]))
        plt.show()

# dice_num_counter = DiceNumCounter()
# dice_area_detector = DiceAreaDetector()
#
# # for image in X_train[:10]:
# #     dice_num = dice_num_counter.count(image)
# #     result_img = dice_area_detector.detect(image, dice_num)
# #     dice_area_detector._show_result()
#
# dice_num_counter = DiceNumCounter(high_intensity=200, dice_area=50)
# dice_area_detector = DiceAreaDetector()
#
# # for image in X_test[:10]:
# #     dice_num = dice_num_counter.count(image)
# #     result_img = dice_area_detector.detect(image, dice_num)
# #     dice_area_detector._show_result()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの定義
class DiceViT(nn.Module):
    def __init__(self, pretrained_model):
        super(DiceViT, self).__init__()
        self.pretrained_model = pretrained_model

    def forward(self, x):
        x = self.pretrained_model(x)
        return x

X_TRAIN_SINGLE_DICE_PATH = 'X_train_single_dice.npy'
Y_TRAIN_SINGLE_DICE_PATH = 'y_train_single_dice.npy'
try:
    X_train_single_dice = np.load(X_TRAIN_SINGLE_DICE_PATH)
    y_train_single_dice = np.load(Y_TRAIN_SINGLE_DICE_PATH)
except:
    ## 後で動作確認 ##
    X_train_single_dice = []
    y_train_single_dice = []
    dice_num_counter = DiceNumCounter()
    dice_area_detector = DiceAreaDetector()
    for idx, (X, y) in tqdm.tqdm(enumerate(zip(X_train, y_train))):
        dice_num = dice_num_counter.count(X)
        if dice_num==1:
            result = dice_area_detector.detect(X, dice_num=dice_num)
            X_train_single_dice.append(result[1][0])
            y_train_single_dice.append(y)
    X_train_single_dice = np.array(X_train_single_dice)
    y_train_single_dice = np.array(y_train_single_dice)
    np.save(X_TRAIN_SINGLE_DICE_PATH, X_train_single_dice)
    np.save(Y_TRAIN_SINGLE_DICE_PATH, y_train_single_dice)


# ノイズ関数の定義
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, device=torch.device('cpu')):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        tensor = tensor.to(self.device)
        return tensor + torch.randn(tensor.size()[1:]).to(self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# 0 - 255 を -1 - 1で正規化
class MinMaxNorm(object):
    def __init__(self, min=0, max=255):
        self.min = min
        self.max = max

    def __call__(self, tensor):
        return (tensor - self.min) / (self.max - self.min) * 2

    def __repr__(self):
        return self.__class__.__name__ + f'(min={self.min}, max={self.max})'


# 画像をカラー化
class GRAY2BGR(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return torch.stack([tensor, tensor, tensor], dim=1).squeeze()

    def __repr__(self):
        return self.__class__.__name__


class SingleDiceDataset(Dataset):
    def __init__(self, X_data, y_data, transform=None) -> None:
        super().__init__()
        self.X_data = X_data
        self.y_data = y_data
        self.transform = transform

    def __getitem__(self, index) -> tuple([torch.Tensor, torch.Tensor]):
        data = torch.Tensor(self.X_data[index]).unsqueeze(0)
        label = torch.Tensor([self.y_data[index]])
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self) -> int:
        return len(self.y_data)


# # 学習設定
# num_epochs = 10
# n_split = 2
# lr = 3e-5
# gamma = 0.7
# seed = 42
#
# # モデルのインスタンス化
# # (今回はOSSでアップロードされている学習済みモデルをロードする)
# pretrained_vit_model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=6).to(device)
# model = DiceViT(pretrained_vit_model)
#
# # 損失関数と最適化関数の設定
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
#
# # 交差検証
# kf = KFold(n_splits=n_split, shuffle=True)
# scores = []
#
# # 学習ループ
# for epoch in range(num_epochs):
#     print(f' --------- epoch {epoch}/{num_epochs} ---------- ')
#
#     for fold_num, (train_idx, valid_idx) in enumerate(kf.split(X_train_single_dice)):
#         print(f'*** Fold {fold_num} ***')
#         # データローダの準備
#         transform = transforms.Compose([
#             transforms.RandomRotation(degrees=(0, 360)),
#             AddGaussianNoise(0.0, 30.0, device=device),
#             transforms.Resize((224, 224), interpolation=0),
#             GRAY2BGR(),
#             MinMaxNorm(),
#         ])
#         train_dataset = SingleDiceDataset(X_train_single_dice[train_idx], y_train_single_dice[train_idx],
#                                           transform=transform)
#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#         valid_dataset = SingleDiceDataset(X_train_single_dice[valid_idx], y_train_single_dice[valid_idx],
#                                           transform=transform)
#         valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
#
#         # 訓練
#         print('Train')
#         model.train()
#         for images, labels in tqdm.tqdm(train_loader):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             optimizer.zero_grad()
#             loss = criterion(outputs, (labels).squeeze().long() - 1)
#             loss.backward()
#             optimizer.step()
#
#         # 評価
#         print('Valid')
#         model.eval()
#         with torch.no_grad():
#             all_preds = []
#             all_labels = []
#             for images, labels in tqdm.tqdm(valid_loader):
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, preds = torch.max(outputs, 1)
#                 all_preds.extend(preds.cpu().numpy() + 1)
#                 all_labels.extend(labels.cpu().numpy())
#             accuracy = accuracy_score(all_labels, all_preds)
#         scores.append(accuracy)
#         # 認識結果の表示
#         print(collections.Counter(all_preds))
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}] - Test Accuracy: {np.mean(scores):.5f}')
#
#     # モデルの保存
#     torch.save(model, f'model_weight_epoch{epoch}.pth')