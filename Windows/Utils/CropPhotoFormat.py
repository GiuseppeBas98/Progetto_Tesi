import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import math
#from align_trans import norm_crop
from facenet_pytorch import MTCNN
mtcnn = MTCNN(select_largest=True, post_process=False, min_face_size=64)


def align_images(in_folder, out_folder):
    print(in_folder)
    os.makedirs(out_folder, exist_ok=True)
    skipped_imgs = []

    img_names = os.listdir(in_folder)
    for img_name in tqdm(img_names):
        #if not (".png" in img_name or ".jpeg" in img_name or ".jpg" in img_name):
        #    continue
        filepath = os.path.join(in_folder, img_name)
        img = cv2.imread(filepath)
        if img is None:
            continue
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        if landmarks is None:
            skipped_imgs.append(img_name)
            continue

        facial5points = landmarks[0]

        #print(boxes[0])
        n = 224
        x1, y1, x2, y2 = boxes[0]
        w = x2-x1
        h = y2-y1
        ew = (n - w)/2
        eh = (n - h)/2

        x1 = round(x1 - ew)
        y1 = round(y1 - eh)
        x2 = x1 + n
        y2 = y1 + n
        crop_img = img[y1:y2, x1:x2]

        #warped_face = norm_crop(img, landmark=facial5points, createEvalDB=True)
        #warped_face = cv2.cvtColor(warped_face, cv2.COLOR_RGB2BGR)
        try:
            cv2.imwrite(os.path.join(out_folder, img_name), crop_img)
        except:
            skipped_imgs.append(img_name)
            pass

    print(skipped_imgs)
    print(f"Images with no Face: {len(skipped_imgs)}")


def main():
    parser = argparse.ArgumentParser(description='MTCNN alignment')
    parser.add_argument('--in_folder', type=str, default=r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\FRLL_dataset\FRLL_bonafide\neutral_front", help='folder with images')
    parser.add_argument('--out_folder', type=str, default=r"C:\Users\Giuseppe Basile\Desktop\FRLL_dataset_CROPPED\FRLL_bonafide\neutral_front", help="folder to save aligned images")

    args = parser.parse_args()
    align_images(args.in_folder, args.out_folder)


if __name__ == "__main__":
    main()