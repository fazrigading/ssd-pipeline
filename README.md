# ssd-pipeline

Fine-tunes a pretrained [torchvision SSD300-VGG16](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html) (COCO) object detector for mushroom detection: `ganoderma` and `primordia`. 640 px input, Pascal VOC XML annotations, mAP evaluation via torchmetrics.

## Structure

```
ssd_pipeline/                   # package: config, model builder, dataset + transforms
  config.py                     # paths, classes, hyperparameters (edit here)
  model.py                      # SSD300-VGG16 with custom classification head
  datasets.py                   # VOC XML dataset + dataloaders
  custom_utils.py               # transforms, Averager, checkpoint/plot helpers
train.py                        # training loop; saves best model + plots to outputs/
eval.py                         # validation mAP evaluation; saves mAP.png
inference.py                    # image directory detection; draws boxes to inference_outputs/
inference_video.py              # video inference; writes annotated video
data/ganoderma/{train,valid}/   # images + XML annotations (not in repo)
outputs/                        # model checkpoints and training plots
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

## Use your own dataset

### 1. Annotate

Annotate your images in [Pascal VOC XML format](https://towardsdatascience.com/a-guide-to-building-pascal-voc-dataset-in-under-one-hour-a-beginner-s-tutorial-3e0b54c9ab45) — one `.xml` per image, same base filename (`image_1.jpg` + `image_1.xml`). Any tool that exports Pascal VOC works: [LabelImg](https://github.com/HumanSignal/labelImg), [Roboflow](https://roboflow.com), CVAT. Each annotated object looks like:

```xml
<object>
    <name>ganoderma</name>
    <bndbox>
        <xmin>120</xmin><xmax>420</xmax>
        <ymin>85</ymin><ymax>400</ymax>
    </bndbox>
</object>
```

The `<name>` text must match a class in `CLASSES` (case-sensitive) exactly, or the dataset raises at load time.

### 2. Split

Images annotated, split them across two directories:

```
my_dataset/
  train/     # ~80% of images + matching .xml files
  valid/     # ~20%, no overlap with train
```

### 3. Configure

Edit `ssd_pipeline/config.py`:

```python
TRAIN_DIR = 'my_dataset/train'
VALID_DIR = 'my_dataset/valid'

# First entry must stay '__background__' (index 0 is background),
# every other class gets index 1, 2, ...
CLASSES = [
    '__background__', 'cat', 'dog'
]
```

Optional knobs: `RESIZE_TO` (input size, 640 default — inference must use the same, `--imgsz`), `BATCH_SIZE` (set by GPU memory), `NUM_EPOCHS` (100 default).

### 4. Sanity check

Verify annotations load before burning GPU hours:

```
python -c "from ssd_pipeline.datasets import CustomDataset; from ssd_pipeline.config import TRAIN_DIR, RESIZE_TO, CLASSES; d = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES); print(len(d), 'images'); print(d[0][1]['labels'])"
```

Expected: an image count and label tensor like `tensor([1, 2, 1])`. Or run `python datasets.py` for on-screen annotation overlay (original behavior; run from repo root, needs a display).

### 5. Train and check

```
python train.py     # saves outputs/best_model.pth + loss/mAP plots
python eval.py      # prints validation mAP
```

### 6. Inference

```
python inference.py -i my_dataset/valid --threshold 0.25
```

Video inference works with the same trained weights:

```
python inference_video.py -i data/video.mp4 --threshold 0.25
```

## Usage (original splits)

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

## License

[MIT](LICENSE). Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Author

Fazri Gading — [GitHub](https://github.com/fazrigading) · fazrigading@gmail.com

