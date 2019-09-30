from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
import skimage
from PIL import Image
import torch

image = skimage.io.imread("demo.jpg")
if image.shape[2] == 4:
	image = image[:, :, :3]
# image = torch.from_numpy(image).float()

predictions = coco_demo.run_on_opencv_image(image)
predictions = Image.fromarray(predictions)
predictions.save("demo_result.png")