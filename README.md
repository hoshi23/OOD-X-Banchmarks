# OOD-X-Banchmarks


## Get Started

### Installation


This benchmark now supports installation via poetry.

```
# install third party sources

mkdir third_party && cd third_party
git clone git@github.com:KaiyangZhou/Dassl.pytorch.git

# installation via poetry
poetry config virtualenvs.in-project true
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

- `imagenet-classes.txt`: https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view
- `wordnet_is_a.txt`: http://www.image-net.org/data/wordnet.is_a.txt
- `words.txt`: http://www.image-net.org/data/words.txt


#### ImageNet derived datasets

Download ImageNet-V2, ImageNet-R, ImageNet-C.

```
cd data_managment/imagenet
python download_variance.py
```


#### Wilds

Download iWilCam and FMoW by using official library.

```
cd data_managment/wilds
python download.py
```


