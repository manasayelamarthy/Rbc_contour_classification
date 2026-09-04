from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR  = Path('F:/Livo/Data - 2026/Rbc/rbc_yolo')
DATASET_YAML = 'F:/Livo/Data - 2026/Rbc/yolo_dataset/dataset.yaml'

IMGSZ    = 1024
BATCH    = 4
EPOCHS   = 150
PATIENCE = 30
BASE_MODEL = 'yolo11s.pt'
RUN_NAME   = f'{Path(BASE_MODEL).stem}_{IMGSZ}_singlecls'


def main():
    model = YOLO(BASE_MODEL)
    result = model.train(
        data           = DATASET_YAML,
        epochs         = EPOCHS,
        patience       = PATIENCE,
        imgsz          = IMGSZ,
        batch          = BATCH,
        project        = str(PROJECT_DIR),
        name           = RUN_NAME,
        device         = 0,
        workers        = 4,
        optimizer      = 'AdamW',
        lr0            = 0.001,
        lrf            = 0.01,
        cos_lr         = True,
        warmup_epochs  = 5,
        single_cls     = True,   
        hsv_h          = 0.0,
        hsv_s          = 0.0,
        hsv_v          = 0.02,
        fliplr         = 0.5,
        flipud         = 0.5,
        degrees        = 15.0,
        scale          = 0.3,
        translate      = 0.1,
        mosaic         = 0.0,
        mixup          = 0.0,
        copy_paste     = 0.0,
        erasing        = 0.0,
        box            = 7.5,
        cls            = 0.5,
        dfl            = 1.5,
        save           = True,
        save_period    = 10,
        val            = True,
        plots          = True,
    )

    if hasattr(result, 'results_dict'):
        m = result.results_dict
        print('mAP50    :', m.get('metrics/mAP50(B)', 'n/a'))
        print('mAP50-95 :', m.get('metrics/mAP50-95(B)', 'n/a'))
        print('Precision:', m.get('metrics/precision(B)', 'n/a'))
        print('Recall   :', m.get('metrics/recall(B)', 'n/a'))


if __name__ == '__main__':
    main()
