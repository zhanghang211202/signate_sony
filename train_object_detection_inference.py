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
from train_object_detection import DiceNumCounter, DiceAreaDetector, DiceViT, GRAY2BGR, MinMaxNorm


class DiceNumSumEstimator():
    def __init__(self, counter, detector, recognizer, transform, device, result_max_num=100):
        self._device = device
        self._counter = counter
        self._detector = detector
        self._recognizer = recognizer
        self._transform = transform
        self._result_max_num = result_max_num
        self._result_num = [0, 0, 0, 0, 0, 0]
        self._result_images = None

    def estimate(self, image):
        # サイコロの数をカウント
        dice_num = self._counter.count(image)
        # 検出の実施
        _, single_dice_image_list = self._detector.detect(image, dice_num)
        # 認識の実施
        sum_number = 0
        for single_dice_image in single_dice_image_list:
            single_dice_image_tensor = torch.Tensor(single_dice_image).unsqueeze(0).to(self._device)
            single_dice_image_tensor = self._transform(single_dice_image_tensor).unsqueeze(0)
            output = torch.argmax(self._recognizer(single_dice_image_tensor))

            if self._result_images is None:
                self._result_images = [
                    np.zeros([self._result_max_num, single_dice_image.shape[0], single_dice_image.shape[1]]) for i in
                    range(6)]
            self._result_images[output][self._result_num[output] % self._result_max_num] = single_dice_image
            self._result_num[output] += 1

            sum_number += output + 1
        return sum_number

    def show_result_sample(self, n_sample=10):
        if n_sample > self._result_max_num:
            raise Exception('n_sample > self._result_max_num')
        _ = plt.figure(figsize=[10 * n_sample, 60])
        for number in range(6):
            for i_img in range(n_sample):
                ax = plt.subplot(6, n_sample, number * n_sample + i_img + 1)
                ax.imshow(self._result_images[number][i_img])
        plt.show()

X_test = np.load(f'data/X_test.npy').reshape(-1, 20, 20)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dice_num_counter = DiceNumCounter()
dice_area_detector = DiceAreaDetector()
dice_recognizer = DiceViT(torch.load('model_weight_epoch9.pth').to(device))
dice_recognizer.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=0),
    GRAY2BGR(),
    MinMaxNorm(),
])
dice_num_sum_estimator = DiceNumSumEstimator(
    counter=dice_num_counter,
    detector=dice_area_detector,
    recognizer=dice_recognizer,
    transform=transform,
    device=device,
)

output_list = []
with torch.no_grad():
    for image in tqdm.tqdm(X_test):
        output = int(dice_num_sum_estimator.estimate(image).cpu())
        output_list.append(output)
pd.DataFrame(output_list).to_csv('18_single_model.csv', header=False)
dice_num_sum_estimator.show_result_sample(n_sample=20)