import os
import sys
import shutil
import random
import cv2
import torch
import numpy as np
from ultralytics import YOLO


def xywhr2xyxyxyxy(center):
    # reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)



def main(to_pred_dir, result_save_path):
    run_py = os.path.abspath(__file__) # run.py
    dirname = os.path.dirname(run_py) #! model_dir (model)

    images_dir = os.path.abspath(os.path.join(to_pred_dir, "images"))
    imagenames = os.listdir(images_dir)
    basenames = [imagename.split('.')[0] for imagename in imagenames] # 图片name
    
    #! 加载模型进行预测, 需选手自行补充
    # 输入 name -> model -> txt
    
    model = YOLO(model=dirname+"/"+'best.pt')
    # print(model)
    # for img in basenames:
    # print("basename",basenames)
    for img_name in basenames:    
        img = cv2.imread(filename=(images_dir+"/"+img_name+".png"))
        results = model(img)[0]
        # print(len(results))
        # names   = results.names
        boxes   = results.obb.data.cpu()
        confs   = boxes[..., 5].tolist()
        classes = list(map(int, boxes[..., 6].tolist()))
        boxes   = xywhr2xyxyxyxy(boxes[..., :5])
        
        for i, box in enumerate(boxes):
            confidence = confs[i]
    
            # print(box)
                
            label_path = os.path.join(dirname, "submit", "%s.txt"%(img_name))
            mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            c = mapping[classes[i]]
            # print(c) 
            
            # box
            x1, y1 = box[1]
            x2, y2 = box[0]
            x3, y3 = box[3]
            x4, y4 = box[2]
            
            
            with open(label_path, "a") as fw:
                fw.write("%s %f %f %f %f %f %f %f %f %f"%(c, x1, y1, x2, y2, x3, y3, x4, y4, confidence))
                fw.write("\n")
        
    
    
    
    #!~ submit文件夹必须有
    # labels_dir = os.path.join(dirname, "submit")
    # if not os.path.exists(labels_dir): os.makedirs(labels_dir)
    
    # #!~ 示例中随意写一个任意数据, 请选手根据自己模型根据替换修改
    # #!~ 该示例仅能保证提交流程正常, 请选手务必根据自己模型根据替换修改
    # label_path = os.path.join(dirname, "submit", "%s.txt"%(basenames[0]))
    # c = random.choice(['A','B','C','D'])
    # x1 = random.randint(0, 512)
    # y1 = random.randint(0, 512)
    # x2 = random.randint(x1, 512)
    # y2 = random.randint(y1, 512)
    # with open(label_path, "w") as fw:
    #     fw.write("%s %d %d %d %d %d %d %d %d 0.95"%(c, x1, y1, x2, y1, x2, y2, x1, y2))
    
    #! 将预测结果压缩打包
    os.system('cd %s && zip -r submit.zip submit/'%dirname)
    os.system(f'cd {dirname} && cp submit.zip {result_save_path}')



if __name__ == "__main__":

    
    # main("/home/uestc/workspace/submit_exp/", "/home/uestc/workspace/submit_exp/result_save_path")
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)
