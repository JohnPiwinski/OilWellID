import os
import pandas as pd
import ast
from PIL import Image
import shutil as sh
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

DATA_DIR = Path('./datasets/daqing_processed')
img_list = list(DATA_DIR.glob('images/*.jpg'))

img_dim_dict = {}

for img in img_list:
    img_name = str(img).split('/')
    img_name = img_name[len(img_name) - 1]
    img_name = img_name.split('.')
    img_name = img_name[0]
    f_img = Image.open(img)
    img_dim_dict[img_name] = (f_img.width, f_img.height)

def literal_eval(x):
    return ast.literal_eval(x.rstrip('\r\n'))

df = pd.read_csv(DATA_DIR / "annotations.csv", converters={'bounds': literal_eval})
df.head(10)

image_ids = df['image_id'].unique()
train_image_ids, validation_image_ids = train_test_split(image_ids, test_size=0.2, random_state=0)
print(validation_image_ids)

HOME_DIR = './' 
DATASET_PATH = 'dataset/images'

def make_dir_safe(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print("The directory " + path + " already exists; any files already present will be overwritten")

make_dir_safe(HOME_DIR + "dataset")
make_dir_safe(HOME_DIR + DATASET_PATH)
make_dir_safe(HOME_DIR + DATASET_PATH + "/train2017")
make_dir_safe(HOME_DIR + DATASET_PATH + "/val2017")
make_dir_safe(HOME_DIR + DATASET_PATH + "/annotations")

RESIZE_WIDTH = 640
RESIZE_HEIGHT = 640

for img_path in img_list:
    folder = 'val' if img_path.stem in validation_image_ids else 'train'
    pil_img = Image.open(img_path, mode='r')
    resized_img = pil_img.resize((RESIZE_WIDTH, RESIZE_HEIGHT))
    resized_img.save(f'{HOME_DIR}{DATASET_PATH}/{folder}2017/{img_path.stem}.jpg')

print(f'Number of training files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/train2017/"))}')
print(f'Number of validation files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/val2017/"))}')

def save_annotation_json(json_annotation, filename):
    with open(filename, 'w') as f:
        output_json = json.dumps(json_annotation)
        f.write(output_json)

def dataset2coco(df):
    annotations_json = {"info": [],"licenses": [], "categories": [],"images": [],"annotations": []}
    
    info = {"year": "2022", "version": "1", "description": "Oil Well Detection Dataset - COCO format", "contributor": "", "url": "", "date_created": "2022-10-22T04:17:27+00:00"}
    annotations_json["info"].append(info)
    
    license = {"id": 1,"url": "","name": "Unknown"}
    annotations_json["licenses"].append(license)

    classes = {"id": 0, "name": "oil-well", "supercategory": "none"}
    annotations_json["categories"].append(classes)

    image_ids_dict = {k: v for v, k in enumerate(df['image_id'].unique())}
    for key, value in image_ids_dict.items():
        images = {"id": value, "license": 1, "file_name": key + '.jpg', "height": RESIZE_HEIGHT,
                  "width": RESIZE_WIDTH, "date_captured": "2022-10-22T04:17:27+00:00"}
        annotations_json["images"].append(images)

    for index, ann_row in enumerate(df.itertuples()):
        print("Adding row pandas dataframe", index, ann_row)
        x_min, y_min, x_max, y_max = ann_row.bounds
        IMAGE_WIDTH, IMAGE_HEIGHT = img_dim_dict[ann_row.image_id]
        RESIZE_WIDTH_RATE  = RESIZE_WIDTH / IMAGE_WIDTH
        RESIZE_HEIGHT_RATE = RESIZE_HEIGHT / IMAGE_HEIGHT
        b_width = (x_max - x_min) * RESIZE_WIDTH_RATE
        b_height = (y_max - y_min) * RESIZE_HEIGHT_RATE

        image_annotations = {"id": index,"image_id": image_ids_dict[ann_row.image_id],"category_id": 0,"bbox": [x_min * RESIZE_WIDTH_RATE, y_min * RESIZE_HEIGHT_RATE, b_width, b_height],"area": b_width * b_height,"segmentation": [],"iscrowd": 0}
        annotations_json["annotations"].append(image_annotations)

    print(f"COCO json format completed! Files: {len(df)}")
    return annotations_json

train_annotation_json = dataset2coco(df[df['image_id'].isin(train_image_ids)])
validation_annotation_json = dataset2coco(df[df['image_id'].isin(validation_image_ids)])
save_annotation_json(train_annotation_json, f"{HOME_DIR}{DATASET_PATH}/annotations/train.json")
save_annotation_json(validation_annotation_json,f"{HOME_DIR}{DATASET_PATH}/annotations/valid.json")

CONFIG_PATH='oil_rig_config.py'

coco_cls = '''
COCO_CLASSES = (
  "",
)
'''
with open('./YOLOX/yolox/data/datasets/coco_classes.py', 'w') as f:
    f.write(coco_cls)