import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"C:/Users/harsh/OneDrive/Desktop/college/maverick/AUTONOMOUS/yolo/trained_on_laptops/yolo11s-RGBT-midfusion-TMI_300epochs/weights/best.pt")
    model.val(data=r"E:/dataset_new/dataset_new/data.yaml",
              split='val',
              imgsz=640,
              batch=16,
              use_simotm="RGBT",
              channels=4,
              project='runs/val/66_epoch',
              name='66_epoch_valdation_niicu',
              )