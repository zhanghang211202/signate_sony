from PIL import Image
import numpy as np
import random
import os

###  バウンディングボックスを返す関数  ###
def get_non_transparent_area(image):
    bbox = image.split()[-1].getbbox()
    return bbox if bbox else (0, 0, 0, 0)

###  2つのバウンディングが重ならないか確認する関数  ###
def check_overlap(bbox1, bbox2):
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

###  画像を重ならない位置に配置する関数  ###
def place_image(background, image, non_transparent_area):
    rec_times = 0
    while True:
        rotated_size = image.size
        background_size = background.size

        max_x_rotated = background_size[0] - rotated_size[0]
        max_y_rotated = background_size[1] - rotated_size[1]

        max_x_rotated = max(max_x_rotated, 0)
        max_y_rotated = max(max_y_rotated, 0)

        random_x_rotated = random.randint(0, max_x_rotated)
        random_y_rotated = random.randint(0, max_y_rotated)


        new_bbox = (random_x_rotated, random_y_rotated, random_x_rotated + rotated_size[0], random_y_rotated + rotated_size[1])

        rec_times += 1
        if not check_overlap(non_transparent_area, new_bbox):
            background.paste(image, (random_x_rotated, random_y_rotated), image)
            return new_bbox, True
        if rec_times > 100:  # 100回試してもダメなら諦める
            return (0, 0, 0, 0), False

###  サイコロ1個の画像を生成する関数  ###
def create_dice1_img(num_img=1000, save_path="", dice_sample_path=""):
    # 背景の1辺:サイコロの1辺≒20:7ぐらい？
    background_size = (2000, 2000)
    downscaled_size = (700, 700)
    for idx_dice in range(1, 7):
        image_path = f'{dice_sample_path}/sample_dice1_{idx_dice}.png'
        original_image = Image.open(image_path).convert('RGBA')
        downscaled_image = original_image.resize(downscaled_size)
        for idx_img in range(num_img):
            non_transparent_area = (0, 0, 0, 0)
            background = Image.new("RGB", background_size, (0, 0, 0))
            random_angle = random.randint(0, 360)
            rotated_image = downscaled_image.rotate(random_angle, expand=True)
            non_transparent_area, has_1dice = place_image(background, rotated_image, non_transparent_area)

            dice1_img = background.convert("L")
            print(dice1_img)
            dice1_img_path = f"{save_path}/{0}_{idx_dice}_{idx_img}.png"
            dice1_img.save(dice1_img_path)


###  サイコロ2個の画像を生成する関数  ###
def create_dice2_img(num_img=1000, save_path="", dice_sample_path=""):
    # 背景の1辺:サイコロの1辺≒20:7ぐらい？
    background_size = (2000, 2000)
    downscaled_size = (700, 700)
    for idx_dice1 in range(1, 7):
        image1_path = f'{dice_sample_path}/sample_dice1_{idx_dice1}.png'
        original_image1 = Image.open(image1_path).convert('RGBA')
        downscaled_image1 = original_image1.resize(downscaled_size)
        for idx_dice2 in range(idx_dice1, 7):
            image2_path = f'{dice_sample_path}/sample_dice1_{idx_dice2}.png'
            original_image2 = Image.open(image2_path).convert('RGBA')
            downscaled_image2 = original_image2.resize(downscaled_size)
            for idx_img in range(num_img):
                non_transparent_area = (0, 0, 0, 0)
                background = Image.new("RGB", background_size, (0, 0, 0))
                random_angle1 = random.randint(0, 360)
                rotated_image1 = downscaled_image1.rotate(random_angle1, expand=True)
                non_transparent_area, has_1dice = place_image(background, rotated_image1, non_transparent_area)

                random_angle2 = random.randint(0, 360)
                rotated_image2 = downscaled_image2.rotate(random_angle2, expand=True)
                non_transparent_area, has_2dices = place_image(background, rotated_image2, non_transparent_area)
                if not has_2dices:
                    continue

                dice2_img = background.convert("L")
                dice2_img_path = f"{save_path}/{idx_dice1}_{idx_dice2}_{idx_img}.png"
                dice2_img.save(dice2_img_path)

save_path = "./data/generate_dataset/labels/train"
if not os.path.exists(save_path):
    os.makedirs(save_path)
