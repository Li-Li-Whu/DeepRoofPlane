  # DeepRoofPlane
This is the PyToch implementation of the following manuscript:
> A boundary-aware point clustering approach in Euclidean and embedding spaces for roof plane segmentation
>
> Li Li, Qingqing Li, Guozheng Xu, Pengwei Zhou, Jingmin Tu, Jie Li, Mingming Li, Jian Yao
>
This manuscript has been accepted by ISPRS Journal.

## Synthetic and real dataset

To train and test DeepRoofPlane, we construct a new synthetic dataset of roof building 3D reconstruction. To further evaluate the performance of the proposed DeepRoofPlane, we also construct a small real dataset. 
The real building point clouds are selected from [RoofN3D](https://github.com/sarthakTUM/roofn3d).
On this basis, we manually labeled and reproduced the roofs of 1,874 buildings.

The synthetic and real dataset can be downloaded from [[baiduyun](https://pan.baidu.com/s/14nS9D2XMhj-UpsSCXL5-4w?pwd=e9r4) :e9r4]. 
You can directly unzip the file to obtain all train and test sets. 


 
## Usage
The environment requires torch1.8+cu111 with python=3.7. The requirements list is shown as follow:
- pandas
- shapely
- scikit-learn
- tensorboardX
- pyyaml
- libboost
- tqdm

If the above requirements are incomplete, you can install the missed dependencies according to the compilation errors.

## Train and test
After downloading our dataset and code, you need to prepare your train.txt and test.txt.
We have provided the train.txt and test.txt used in dataset.

Then, your can run train.py and test.py for training and testing the model. Notably, some parameter settings required by the model are stored in the .yaml file.The roofseg1.yaml is for Roofpc3d, and the roofseg3.yaml is for RoofN3d. The models are stored in the output folder.
An example for training command (--extrag_tag $model_save_dir$):
```shell script
python train.py --dataset Roofpc3d --cfg_file ./roofseg1.yaml --extra_tag train_roofpc3d_ckpt --batch_size 16
``` 
An example for testing command (only write model outputs; --test_tag $test_model_dir$):
```shell script
python test.py --dataset Roofpc3d --cfg_file ./roofseg1.yaml --test_tag test_roofpc3d_dir --batch_size 1
```
If you want clustering-based post-processing, add "--cluster" to testing command. It should be noted that the clustering algorithm actually is implemented by C++. Here, we provide the Python code of clustering algorithm.



## Results
The model output, such as pred_sem_label, offset, feature, etc., will be saved in ./results.
Several examples are provided in ./results.

If you want visualize the output features, we recommend Google's open-source software Embedding Projector. Run txt2tsv.py for format conversion.
```shell script
python txt2tsv.py
```
For the quantative evaluation of the experimental results, you can check ./quanti_eval_res.txt.



## Citation

If you find our work useful for your research, please consider citing our paper.
> A boundary-aware point clustering approach in Euclidean and embedding spaces for roof plane segmentation
>
> Li Li, Qingqing Li, Guozheng Xu, Pengwei Zhou, Jingmin Tu, Jie Li, Mingming Li, Jian Yao

In addition, if you use the RoofN3d dataset, please also consider citing the following paper:

```shell script
@article{wichmann2019roofn3d,
  title={RoofN3D: A database for 3D building reconstruction with deep learning},
  author={Wichmann, Andreas and Agoub, Amgad and Schmidt, Valentina and Kada, Martin},
  journal={Photogrammetric Engineering \& Remote Sensing},
  volume={85},
  number={6},
  pages={435--443},
  year={2019},
  publisher={American Society for Photogrammetry and Remote Sensing}
}
``` 

## Contact:
Li Li (li.li@whu.edu.cn)