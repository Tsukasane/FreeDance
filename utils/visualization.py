import os
import cv2
import re

def img2video(file_dir='/home/xingqunqi/AI_dance/MMM/vq_image', key_word="vqvae_recons_iter300000"):
    """
    将连续帧存图片 --> 视频因为坐标系偏移抖动会很严重
    视频的帧率 --> 原本是30fps但是videowriter用30会motion很快
    TODO(yiwen) check MMM & EDGE & Bailando visualization method
    """
    ls = []

    imgs = os.listdir(file_dir)
    recons_imgs = []
    for i_str in imgs:
        if key_word in i_str:
            recons_imgs.append(i_str)

    sorted_list = sorted(recons_imgs, key=lambda x: int(re.search(r't(\d+)_', x).group(1))) # key frame <1000

    video = cv2.VideoWriter('outV.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),5,(1200,1200))

    for file_name in recons_imgs:
        img = cv2.imread(os.path.join(file_dir, file_name)) 
        img = cv2.resize(img,(1200,1200)) 
        video.write(img) 
        
    video.release()


if __name__=='__main__':
    img2video('/home/xingqunqi/AI_dance/MMM/vq_image', "vqvae_gt_iter300000")