import pandas as pd

def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

def yolo_preprocessing(dataset: pd.DataFrame, labelsPath: str):
    for i in range(len(dataset)):
        with open(labelsPath + dataset['file_name'][i][:-3] + "txt", 'w') as file:
            for j in range(len(dataset["bbox"][i])):
                category_id = dataset['category_id'][i][j]
                bbox = dataset['bbox'][i][j]
                file.write(f'{category_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')