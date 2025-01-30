# A Benchmark and Evaluation for Real-World Out-of-Distribution Detection Using Vision-Language Models
|🎉 Introducing our new benchmarks for out-of-distribution (OOD) detection!  It's more challenging & closer to real-world scenarios, paving the way for real-world OOD detection 🚀|
|-----------------------------------------|

<div align="center">
    Shiho Noda<sup>1</sup></a>&emsp;
    <a href='https://atsumiyai.github.io/' target='_blank'>Atsuyuki Miyai<sup>1</sup></a>&emsp;
    <a href='https://yu1ut.com/' target='_blank'>Qing Yu<sup>1,2</sup></a>&emsp;
    <a href='https://scholar.google.co.jp/citations?hl=ja&user=2bCSG1AAAAAJ&view_op=list_works&authuser=1&sortby=pubdate' target='_blank'>Go Irie<sup>3</sup></a>&emsp;
    <a href='https://scholar.google.co.jp/citations?user=CJRhhi0AAAAJ&hl=en' target='_blank'>Kiyoharu Aizawa<sup>1</sup></a>
</div>
<div align="center">
    <sup>1</sup>The University of Tokyo&emsp;
    <sup>2</sup>LY Corporation&emsp;
    <sup>3</sup>Tokyo University of Science&emsp;
    <br>
</div>

![ood_x_benchmark](figures/ood_x_benchmark.png)





## Abstract
Conventional benchmarks have reached performance saturation, making it difficult to compare recent OOD detection methods. To address this challenge, we introduce **three novel OOD detection benchmarks** that enable a deeper understanding of method characteristics and reflect real-world conditions. First, we present **ImageNet-X**, designed to evaluate performance under challenging semantic shifts. Second, we propose **ImageNet-FS-X** for full-spectrum OOD detection, assessing robustness to covariate shifts (feature distribution shifts). Finally, we propose **Wilds-FS-X**, which extends these evaluations to real-world datasets, offering a more comprehensive testbed. Our experiments reveal that recent CLIP-based OOD detection methods struggle to varying degrees across the three proposed benchmarks, and none of them consistently outperforms the others. We hope the community goes beyond specific benchmarks and includes more challenging conditions reflecting real-world scenarios. 

## Get Started

### Installation


This benchmark now supports installation via poetry.

```
# install third party sources

mkdir third_party && cd third_party
git clone git@github.com:KaiyangZhou/Dassl.pytorch.git

# installation via poetry
poetry lock
poetry install
poetry shell

```


### Data

Please create `data` folder and download the following ID and OOD datasets to `data`.
Then, prepare the dataset as follows.


The overall file structure is as follows:


```
root
|-- data
    |-- imagenet
        |-- imagenet-classes.txt
        |-- images/
            |--train/ # contains 1,000 folders like n01440764, n01443537, etc.
            |-- val/ # contains 1,000 folders like n01440764, n01443537, etc.
    |-- imagenet_v2
    |-- imagenet_r
    |-- imagenet_c
    |-- iwildcam_v2.0
    |-- fmow_v1.1
    ...
```


#### ImageNet-1k

Download datas. 

1. Create a folder named `imagenet/` under `data` folder.
2. Create `images/` under `imagenet/`.
3. Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/images`.

Download metadatas and put under  `data/imagenet` folder.

- `classnames.txt`: https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view
- `wordnet_is_a.txt`: http://www.image-net.org/data/wordnet.is_a.txt
- `words.txt`: http://www.image-net.org/data/words.txt


#### ImageNet variant datasets

Download ImageNet-V2, ImageNet-R, ImageNet-C.
The composition of this test set is based on [OpenOOD v1.5](https://arxiv.org/abs/2306.09301).

```
cd data_managment/imagenet
python download_variance.py
```


#### Wilds

Download iWilCam and FMoW.

```
cd data_managment/wilds
python download.py
```


### Training and evaluation

We provide training and evaluation scripts in `scripts` folder.
After navigating to the directory containing the relevant script file, execute the script.


## Supported Benchmarks


### ImageNet-X


#### About


ImageNet-X is a benchmark that splits the ImageNet-1k.
This benchmark enables a precise evaluation of sensitivity to semantic shifts by separating semantically similar labels into ID and OOD categories.
Based on the WordNet hierarchy, closely related labels were separated into ID and OOD.
(Refer to `data_managment/imagenet/make_split_list.py` for the split method.)


#### Run scripts


```
bash imagenet_x.sh
```


### ImageNet-FS-X
#### About


ImageNet-FS-X incorporates covariate shift into ImageNet-X.
Derived datasets from ImageNet-1k are used as data with different covariate distributions. 


#### Run scripts

Since the training process is shared between ImageNet-X and ImageNet-FS-X, the pre-trained model from one can be used, allowing you to skip the training phase.

```
bash imagenet_fs_x.sh
```

### Wilds-FS-X (iWildCam)
#### About


Wilds-FS-X is a benchmark that brings the problem setting of ImageNet-FS-X closer to real-world scenarios by utilizing [Wilds](https://wilds.stanford.edu/datasets/).
iWildCam is a dataset of wildlife photos taken by camera traps.
Semantic shift corresponds to different animal species, and covariate shift corresponds to changes in camera locations of camera photos.

#### Run scripts


```
bash iwildcam_fs_x.sh
```

### Wilds-FS-X (FMoW)
#### About


FMoW is a RGB satellite image dataset. 
Semantic shift corresponds to the building or land use categories, and covariate shift corresponds to changes in the year when the images were captured.

#### Run scripts


```
bash fmow_fs_x.sh
```


## Supported Methods

This part lists all the methods we include in this codebase.

- [MCM](https://arxiv.org/abs/2211.13445)
- [GL-MCM](https://arxiv.org/abs/2304.04521)
- [CoOp](https://arxiv.org/abs/2109.01134)
- [LoCoOp](https://arxiv.org/abs/2306.01293)


### How to add custom trainer

If you want to add a new trainer, please refer to `trainers/trainer_templete.py` and implement the addition accordingly.




## Citation
If you find our paper helpful for your research, please consider citing the following paper:
```bibtex
@article{noda2025oodx,
  title={A Benchmark and Evaluation for Real-World Out-of-Distribution Detection Using Vision-Language Models},
  author={Noda, Shiho and Miyai, Atsuyuki and Yu, Qing and Irie, Go and Aizawa, Kiyoharu},
  journal={hogehgeo},
  year={2025}
}
```


Besides, please also consider citing our other projects that are closely related to this paper.    


```bibtex

# GL-MCM (Zero-shot OOD detection)
@article{miyai2025gl,
  title={GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot Out-of-Distribution Detection},
  author={Miyai, Atsuyuki and Yu, Qing and Irie, Go and Aizawa, Kiyoharu},
  journal={International Journal of Computer Vision},
  pages={1--11},
  year={2025},
  publisher={Springer}
}


# LoCoOp (Few-shot OOD detection, Concurrent work with PEFT-MCM)
@inproceedings{miyai2023locoop,
  title={LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning},
  author={Miyai, Atsuyuki and Yu, Qing and Irie, Go and Aizawa, Kiyoharu},
  booktitle={NeurIPS},
  year={2023}
}

# Survey on OOD detection in VLM era
@article{miyai2024generalized2,
  title={Generalized Out-of-Distribution Detection and Beyond in Vision Language Model Era: A Survey},
  author={Miyai, Atsuyuki and Yang, Jingkang and Zhang, Jingyang and Ming, Yifei and Lin, Yueqian and Yu, Qing and Irie, Go and Joty, Shafiq and Li, Yixuan and Li, Hai and Liu, Ziwei and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
  journal={arXiv preprint arXiv:2407.21794},
  year={2024}
}
```
