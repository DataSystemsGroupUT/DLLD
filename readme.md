### LDLCT : An Instance-based Framework for Lesion Detection on Lung CT Scans

This framework performs Lesion detection on Lung CT Scans. The repository contains code and reference to data that has been used to train this system. The dataset used in this exeriment can be downloaded from : https://nihcc.app.box.com/v/DeepLesion/

The files in the code are arranged as :
Data/: 
	/subset_data.csv : List of files used from the original dataset in .csv format
Segmentation/:
	/extract_patches.py: Script to extract lesion and normal patches from the preprocessed images. Exports a .pickle used for training in train_test_segmentation.py
	/train_test_segmentation.py: Script to train the RBM and Random Forsest pipeline on extracted patches. Exports a trained model to be used in segmentation.py. Install the given version of tensorfow-rbm-master before using this script.
	/segmentation.py: Script to segment image files using trained model.
Postprocessing/:
	/train_filter.py: Script to extract training and test for training filter for postprocessing. Exports instances to train the svm on.
	/svm.py: Script to train the filter and exports a trained filter
	/evaluate.py: Script to output the final bounding box predictions. This also outputs the metrics such as sensitivity and false positive per image on the folder that the evaluation is performed.
Utilities/:
	/bounding_box.py: Put a bounding box on any data image as per the annotation in the meta-data file with the data.
tensorfow-rbm-master/:
			Modified version of Restricted Boltzmann Machine Library we use. Please use this version only as the activation functions has been
			modified from the origional library.

			To install: from the terminal in the folder tensorfow-rbm-master/ use the following command:

			python setup.py install


