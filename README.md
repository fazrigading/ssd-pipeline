# ssd-pipeline

Fine-tunes a pretrained [torchvision SSD300-VGG16](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html) (COCO) object detector for mushroom detection: `ganoderma` and `primordia`. 640 px input, Pascal VOC XML annotations, mAP evaluation via torchmetrics.

## Structure

```
ssd_pipeline/      # package: config, model builder, dataset + transforms
  config.py        # paths, classes, hyperparameters (edit here)
  model.py         # SSD300-VGG16 with custom classification head
  datasets.py      # VOC XML dataset + dataloaders
  custom_utils.py  # transforms, Averager, checkpoint/plot helpers
train.py           # training loop; saves best model + plots to outputs/
eval.py            # validation mAP evaluation; saves mAP.png
inference.py       # image directory detection; draws boxes to inference_outputs/
inference_video.py # video inference; writes annotated video
data/ganoderma/{train,valid}/   # images + XML annotations (not in repo)
outputs/           # model checkpoints and training plots
```

## Setup

```
python -m pip install torch torchvision albumentations opencv-python torchmetrics matplotlib tqdm
```

## Data layout

Each split directory holds images (`.jpg`, `.jpeg`, `.png`) with a matching Pascal VOC `.xml` per image:

```
data/ganoderma/train/
  image_1.jpg
  image_1.xml
```

Configure split paths and class list in `ssd_pipeline/config.py`.

## Usage

Train:
```
python train.py
```

Evaluate mAP on the validation set:
```
python eval.py
```

Run inference (images):
```
python inference.py -i data/ganoderma/valid --threshold 0.25
```

Run inference (video):
```
python inference_video.py -i data/video.mp4 --threshold 0.25
```

Inference scripts require `outputs/best_model.pth` from training. Use `--imgsz` to override the 640 px inference resolution.

## Configuration

All knobs live in `ssd_pipeline/config.py`: batch size, epochs, image size, data paths, class list, output dir.
