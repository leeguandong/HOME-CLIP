'''
@Time    : 2022/8/7 10:10
@Author  : leeguandon@gmail.com
'''
import csv
import os
import json
import argparse

from PIL import Image
from io import BytesIO
import base64
from tqdm import tqdm
from pathlib import Path


def img2base64(file_name):
    img = Image.open(file_name)  # 访问图片路径
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    base64_str = base64_str.decode("utf-8")  # str
    return base64_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_texts', default=r'E:\common_tools\Chinese-CLIP-master\dataset\furnitures\train_texts.txt')
    parser.add_argument('--train_imgs', default=r'E:\common_tools\Chinese-CLIP-master\dataset\furnitures\train_imgs.tsv')
    parser.add_argument('--url', default=r'E:\comprehensive_library\TpImgspider-main\saved\vcg_furnitures\txt_download\url.txt')
    parser.add_argument('--img', default=r"E:\comprehensive_library\TpImgspider-main\saved\vcg_furnitures\img_download")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.train_texts, 'w', encoding='utf-8') as f_txt:
        with open(args.train_imgs, 'w', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            with open(args.url, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for index, line in tqdm(enumerate(lines)):
                    try:
                        id = Path(line.split('\t')[0]).stem
                        # id = os.path.basename(line.split('\t')[0]).split('.jpg')[0]
                        title = line.split('\t')[1][3:]
                        title_clean = title.replace("素材", '').replace('图片', '')
                        id_map = {"text_id": index, "text": title_clean, "image_ids": [id]}
                        img_path = os.path.join(args.img, title + ".jpg")
                        tsv_writer.writerow([id, img2base64(img_path)])
                        f_txt.write(str(id_map).strip().replace("'", "\""))
                        f_txt.write('\n')
                    except:
                        continue
