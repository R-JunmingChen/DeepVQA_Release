# DeepVQA
We propose a convolutional neural networks (CNN) based FR-VQA model, named Deep Video Quality Assessment (DeepVQA), 
where the behavior of the HVS (visual masking effect of the video) is learned from the underlying data distribution of VQA databases.

> KIM, Woojae, et al. Deep video quality assessor: From spatio-temporal visual sensitivity to a convolutional neural aggregation network. In: Proceedings of the European Conference on Computer Vision (ECCV). 2018. p. 219-234.

## Prerequisites
This code was developed and tested with Theano 1.0.2, CUDA 9.0, and Windows python.

## Environment setting
### Setting database path:
For each database, set `BASE_PATH` to the actual root path of each database in the following file:
`IQA_DeepQA_FR_release/data_load/LIVE_VQA.py`.
`IQA_DeepQA_FR_release/data_load/CSIQ_VQA.py`.

## Training DeepVQA
We provide the demo code for training a DeepVQA model.
```bash
python example_fr_VQA.py
```

- `tr_te_file`: Store the randomly divided (training and testing) reference video indices in this file.
- `snap_path`: This indicates the path to store snapshot files.
- `LIVE_VQA.txt`: This .txt contains each databases' sequence names with MOS/DMOS, fps(fps∙times), and resolution including reference data. Note that, when the training step, the test set does not include reference data for a fair comparison.
- The model has stage1 and stage2, we recommend that user firstly train only stage 1, and by using trained weights (snap_file), train stage 2. Training both stage 1 and 2 is not currently trainable as mentioned in the paper.

## Quantitative results
DeepVQA was tested on the full-sets (randomly divided 10 sets) of LIVE VQA, CSIQ VQA databases. During the experiment, we randomly divided the reference images into two subsets, 80% for training and 20% for testing. The correlation coefﬁcients were averaged after the procedure was repeated ~10 times while dividing the training and testing sets randomly.

### Performance of DeepVQA (Stage 1)
Note that the performance can be slightly different by training/testing sets.

|Database |SRCC  |PLCC  |
|---------|:----:|:----:|
|LIVE VQA |0.891 | 0.881|
|CSIQ VQA |0.904 | 0.901|
