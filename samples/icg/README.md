# MASK R-CNN for Object Detection on Icg Dataset
The dataset should have the following folder structure :-
icg_dataset/
	-images/
		-train/
		-val/
	-images_seg/


To modify configurationg of the model, check IcgConfig in icg.py


To run model training, use the following command
python3 icg.py train --dataset /path/to/icg_dataset --weights "coco" --logs logs/

To run model evaluation, use the following command
python3 icg.py train --dataset /path/to/icg_dataset --weights "coco" --logs logs/



You may need to create weights and logs directory - create it within samples/icg directory
