from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import skimage
from PIL import Image
import torch
import os
import util
import json

config_file = "../configs/e2e_faster_rcnn_R_50_FPN_1x_alan.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
# cfg.MODEL.WEIGHT = "../weight/e2e_faster_rcnn_R_50_FPN_1x.pth"
cfg.MODEL.WEIGHT = "../weight/food_0042500.pth"

coco_test = COCODemo(
			cfg,
			min_image_size=400,
			confidence_threshold=0.3,
			)
# load image and then run prediction


# image = skimage.io.imread("000053.png")
# if image.shape[2] == 4:
# 	image = image[:, :, :3]
# # image = torch.from_numpy(image).float()

# pred_image, pred_bboxes = coco_test.run_on_opencv_image(image)
# pred_image = Image.fromarray(pred_image)
# pred_image.save("000053_result.png")

FOOD_DATA_PATH = "../datasets/food/"
def test_validation_set(data_split):
	result_path = "result/fasterrcnn_{}".format(data_split)
	if not os.path.isdir(result_path):
		os.makedirs(result_path)
	result_path += '/'

	val_file = open(FOOD_DATA_PATH + 'train_val/dimsum_{}_detection.csv'.format(data_split), 'r')
	val_paths = []
	for line in val_file:
		val_paths.append(line.strip().split(',')[0])

	num_gt, num_pred, num_tp = 0, 0, 0
	counting_error = [0 for _ in range(7)]
	counting_accuracy = [0 for _ in range(7)]
	num_image = [0 for _ in range(7)]

	for file_id, file_path in enumerate(val_paths[1:]):
		print ("{}/{}: ".format(file_id, len(val_paths) - 1))
		file_name = file_path.replace('/', '_')[:-4]
		pred_img_path = result_path + file_name + '_pred.png'
		json_file_path = result_path + file_name + '.json'

		img_path = FOOD_DATA_PATH + 'image/' + file_path
		label_path = FOOD_DATA_PATH + 'annotations/' + file_path.replace('/', '_').replace('.png', '.txt').replace(
				'.jpg', '.txt').strip()
		image = skimage.io.imread(img_path)
		if image.shape[2] == 4: image = image[:, :, :3]
		gt_bboxes = util.get_gt_bbox(label_path)


		pred_image, pred_bboxes = coco_test.run_on_opencv_image(image)
		pred_image = Image.fromarray(pred_image)
		pred_image.save(pred_img_path)

		pred_bboxes, json_bboxes = util.get_bbox_data(pred_bboxes)
		print (json_bboxes)
		with open(json_file_path, 'w') as json_file:
			json.dump(json_bboxes, json_file)
			

		#compute precision and recall
		_gt = len(gt_bboxes)
		_pred = len(pred_bboxes)
		_tp = 0
		for gt_bbox in gt_bboxes:
			for pred_bbox in pred_bboxes:
				if util.compute_IoU(pred_bbox, gt_bbox) > 0.5: #threshold for IoU
					_tp += 1
					break

		
		print ("precision = {:.4f}, recall = {:.4f}".format(_tp/max(1, _gt), _tp/max(1, _pred)))
		num_gt += _gt
		num_pred += _pred
		num_tp += _tp
		print ("avg_precision = {:.4f}, avg_recall = {:.4f}".format(num_tp / num_pred, num_tp / num_gt))

		counting_error[_gt] += abs(_gt - _pred)
		counting_accuracy[_gt] += int(_gt == _pred)
		num_image[_gt] += 1

	precision = num_tp / num_pred
	recall = num_tp / num_gt
	print ("precision = {:.4f}, recall = {:.4f}".format(precision, recall))

	for cnt_id in range(1, 7, 1):
		if num_image[cnt_id] == 0:
			num_image[cnt_id] += 1
		counting_error[cnt_id] /= num_image[cnt_id]
		counting_accuracy[cnt_id] /= num_image[cnt_id]

		print ('counting {}: mae = {:.2f}, accuracy = {:.2f}%'.format(cnt_id, counting_error[cnt_id], counting_accuracy[cnt_id] * 100))

	print ("overall: avg_mae = {:.2f}, avg_accuracy = {:.2f}%".format(sum(counting_error) / 6,
																	sum(counting_accuracy) * 100 / 6))






if __name__ == '__main__':
	test_validation_set('test')
