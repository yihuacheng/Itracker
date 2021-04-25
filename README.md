# Itracker
The Pytorch Implementation of "Eye Tracking for Everyone". (updated in 2021/04/25)

This is the implementated version in our survey **"Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark"**.
Please refer our paper or visit our benchmark website <a href="http://phi-ai.org/project/Gazehub/" target="_blank">*GazeHub*</a> for more information.
The performance of this version is reported in them.

To know more detail about the method, please refer the origin paper.

We recommend you to use the data processing code provided in <a href="http://phi-ai.org/project/Gazehub/" target="_blank">*GazeHub*</a>.
You can use the processed dataset and this code for directly running.

## Introduction
Gazecapture dataset provides training, test and validation set.
We provide code for the evaluation with training-test mode.
In addition, cross validation is also a common evaluation strategy in gaze estimation. 
We also provide code for the evaluation with cross validation mode.

The project contains following files/folders.
- `model.py`, the model code.
- `train.py`, the entry for training with training-test mode.
- `test.py`, the entry for testing with training-test mode.
- `leave_train.py`, the entry for training with cross-validation mode.
- `leave_test.py`, the entry for training with cross-validation mode.
- `config/`, this folder contains the config of the experiment in each dataset. To run our code, **you should write your own** `config.yaml`. 
- `reader/`, the code for reading data. You can use the provided reader or write your own reader.
- `run.sh`, a shell file for cross validation.

## Getting Started
### Writing your own *config.yaml*

Normally, for training, you should change 
1. `train.save.save_path`, The model is saved in the `$save_path$/checkpoint/`.
2. `train.data.image`, This is the path of image.
3. `train.data.label`, This is the path of label.
4. `reader`, This indicates the used reader. It is the filename in `reader` folder, e.g., *reader/reader_mpii.py* ==> `reader: reader_mpii`.

For test, you should change 
1. `test.load.load_path`, it is usually the same as `train.save.save_path`. The test result is saved in `$load_path$/evaluation/`.
2. `test.data.image`, it is usually the same as `train.data.image`.
3. `test.data.label`, it is usually the same as `train.data.label`.
 
### Training

In the cross validation mode, you can run
```
python leave_train.py config/config_mpii.yaml 0
```
This means the code running with `config_mpii.yaml` and use the `0th` person as the test set.

You also can run
```
bash run.sh leave_train.py config/config_mpii.yaml
```
This means the code will perform cross validation automatically.   
`run.sh` performs iteration, you can change the iteration times in `run.sh` for different datasets, e.g., set the iteration times as `4` for four-fold validation.

In the training-test mode, you can run
```
python train.py config/config_gazecapture.yaml
```

### Testing
In the cross validation mode, you can run
```
python leave_test.py config/config_mpii.yaml 0
```
or
```
bash run.sh leave_test.py config/config_mpii.yaml
```

In the training-test mode, you can run
```
python test.py config/config_gazecapture.yaml
```

### Result
After training or test, you can find the result from the `save_path` in `config_mpii.yaml`. 


## Citation
```
@InProceedings{Krafka_2016_CVPR,
	author = {Krafka, Kyle and Khosla, Aditya and Kellnhofer, Petr and Kannan, Harini and Bhandarkar, Suchendra and Matusik, Wojciech and Torralba, Antonio},
	title = {Eye Tracking for Everyone},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2016}
}


@inproceedings{Cheng2021Survey,
	title={Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark},
	author={Yihua Cheng, Haofei Wang, Yiwei Bao, Feng Lu},
	booktitle={arxiv}
	year={2021}
}
```
## Contact 
Please email any questions or comments to yihua_c@buaa.edu.cn.
