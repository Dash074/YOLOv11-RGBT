import warnings
warnings.filterwarnings('ignore')
from YOLO_RGBT_STUFF.ultralytics import YOLO

if __name__ == '__main__':
    # image
    '''
        The source needs to be in the same directory as the train/val directories, and it must contain the "visible" field. 
        There is an "infrared" directory at the same level as the "visible" directory. 
        The principle is to replace "visible" with "infrared" and load the dual-spectrum data.
    '''

    #for 100epoch
    # model_path = r"C:/Users/harsh/OneDrive/Desktop/college/maverick/AUTONOMOUS/yolo/trained_on_laptops/100epoch_midfusion_b16/weights/best.pt"  # path to model weights
    # source_path = r"E:/dataset_new/dataset_new/visible/test"  # image folder
    # project_name = 'runs/100_epoch_evaluation'
    # run_name = '100_epoch_TMI'

    # model_path = r"C:/Users/harsh/OneDrive/Desktop/college/maverick/AUTONOMOUS/yolo/trained_on_laptops/runs____300EPOCH/200epoch_midfusion_b16/yolo11s-RGBT-midfusion-TMI25____/weights/best.pt"  # path to model weights
    # source_path = r"E:/dataset_new/dataset_new/visible/test"  # image folder
    # project_name = 'runs/144_epoch_evaluation'
    # run_name = '144_epoch_TMI'
    model_path = r"C:/Users/harsh/OneDrive/Desktop/college/maverick/AUTONOMOUS/yolo/trained_on_laptops/yolo11s-RGBT-midfusion-TMI_300epochs/weights/best.pt"  # path to model weights
    source_path = r"E:/DroneRGBT_new/visible/test"  # image folder
    project_name = 'runs/droneRGBT/66_epoch_trail_droneRGBT'
    run_name = '66_epoch_TMI_trail_DroneRGBT'


    model = YOLO(model_path)
    model.predict(
        source=source_path,
        imgsz=640,
        project=project_name,
        name=run_name,
        # show=True,
        save=True,
        save_txt=True,
        save_conf=True,
        use_simotm="RGBT",
        channels=4,
        conf=0.6,  # adjust threshold as needed
    )

    