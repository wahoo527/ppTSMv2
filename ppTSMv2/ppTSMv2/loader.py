import cv2
import numpy as np
from PIL import Image
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def normalize_image(image, mean, std):
    """图像归一化"""
    image = (image - mean) / std
    return image


# def video2imgs(video_path):
#     """图像解码"""
#     cap = cv2.VideoCapture(video_path)
#     videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     num = 3
#
#     frames_idx = [i*num for i in range(videolen//num)]
#     imgs = []
#     for i in range(videolen):
#         ret = cap.grab()
#         if ret == False:
#             continue
#         if frames_idx and i == frames_idx[0]:
#             frames_idx.pop(0)
#             ret, frame = cap.retrieve()
#             if frame is None:
#                 break
#             imgbuf = frame[:, :, ::-1]
#             img = np.array(Image.fromarray(imgbuf, mode='RGB')) / 255.
#             img = img[:, img.shape[1]//3:, :]
#
#             img = cv2.resize(img, (256, 256))
#             img = img[16:256 - 16, 16:256 - 16]
#             img = normalize_image(img, mean, std)
#             imgs.append(img)
#         if frames_idx == None:
#             break
#     cap.release()
#     imgs =imgs[:len(imgs)-len(imgs)%16]
#     return imgs
def video2imgs(video_path):
    """图像解码"""
    cap = cv2.VideoCapture(video_path)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num = 3

    d = int(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS))
    f_pers = 2
    frames_idx = [i * (videolen // (d * f_pers)) + 1 for i in range(d * f_pers)]
    imgs = []
    idx = 0
    for i in range(videolen):

        ret = cap.grab()
        if ret == False:
            continue
        if frames_idx and i == frames_idx[0]:
            frames_idx.pop(0)
            ret, frame = cap.retrieve()
            if frame is None:
                break
            imgbuf = frame[:, :, ::-1]
            img = np.array(Image.fromarray(imgbuf, mode='RGB')) / 255.
            img = img[:, img.shape[1]//3:, :]

            img = cv2.resize(img, (256, 256))
            img = img[16:256 - 16, 16:256 - 16]
            img = normalize_image(img, mean, std)
            imgs.append(img)
            idx += 1
        if frames_idx == None:
            break
    img =  np.zeros_like(img)
    img = normalize_image(img, mean, std)
    for i in range(16*f_pers-idx):
        imgs.append(img)
        idx += 1
    cap.release()
    imgs =imgs[:len(imgs)-len(imgs)%16]
    return imgs

# 定义函数：使用decord读取并处理视频帧
# def video2imgs(video_path):
#     # 创建解码器并打开视频文件
#     vr = decord.VideoReader(video_path)
#     # 获取视频帧数
#     videolen = len(vr)
#     num = 3
#     # 获取按固定间隔取样的帧索引
#     frames_idx = [i * num for i in range(videolen // num)]
#     # 用于存储处理后的图像帧
#     imgs = []
#     # 遍历所有视频帧
#     for i in range(videolen):
#         # 读取视频帧数据
#         frame = vr[i]
#         # 如果帧无法读取，则继续下一个帧
#         if frame is None:
#             continue
#         # 如果当前帧是需要取样的帧，则进行处理
#         if frames_idx and i == frames_idx[0]:
#             frames_idx.pop(0)
#             # 将帧转换为PIL图像
#             imgbuf = frame.asnumpy()[:, :, ::-1]
#             img = np.array(Image.fromarray(imgbuf, mode='RGB')) / 255.
#             # 将处理后的图像帧添加到列表中
#             img = img[:, img.shape[1] // 3:, :]
#
#             img = cv2.resize(img, (256, 256))
#             img = img[16:256 - 16, 16:256 - 16]
#             img = normalize_image(img, mean, std)
#             imgs.append(img)
#         # 如果已经取样完所有需要取样的帧，则结束遍历
#         if not frames_idx:
#             break
#     imgs = imgs[:len(imgs) - len(imgs) % 16]
#     return imgs

def img_sample(imgs):
    """分段抽取图像"""
    imgs = np.array(imgs)
    imgs = np.reshape(imgs, (16, -1, 224, 224, 3))
    imgs = np.transpose(imgs, [1, 0, 4, 2, 3])
    return imgs


def get_data(video_path):
    imgs = video2imgs(video_path)
    imgs = img_sample(imgs).astype('float32')
    return imgs



