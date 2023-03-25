'''
@Time    : 2022/8/2 15:16
@Author  : leeguandon@gmail.com
'''
import torch
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
image = preprocess(Image.open("../dataset/pokemon.jpeg")).unsqueeze(0).to(device)  # 1,3,224,224
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)  # 4,64

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
