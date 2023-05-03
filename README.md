# PointNu-Net

## PointNu-Net: Keypoint-assisted Convolutional Neural Network for Simultaneous Multi-tissue Histology Nuclei Segmentation and Classification. [ArXiv](https://arxiv.org/pdf/2111.01557.pdf)
Kai Yao, Kaizhu Huang, Jie Sun, Amir Hussain, Curran Jude \
Both University of Liverpool and Xi'an Jiaotong-liverpool University 

**Abstract**

Automatic nuclei segmentation and classification play a vital role in digital pathology. However, previous works are mostly built on data with limited diversity and small sizes, making the results questionable or misleading in actual downstream tasks. In this paper, we aim to build a reliable and robust method capable of dealing with data from the ‘the clinical wild’. Specifically, we study and design a new method to simultaneously detect, segment, and classify nuclei from Haematoxylin and Eosin (H\&E) stained histopathology data, and evaluate our approach using the recent largest dataset: PanNuke. We address the detection and classification of each nuclei as a novel semantic keypoint estimation problem to determine the center point of each nuclei. Next, the corresponding class-agnostic masks for nuclei center points are obtained using dynamic instance segmentation. Meanwhile, we proposed a novel Joint Pyramid Fusion Module (JPFM) to model the cross-scale dependencies, thus enhancing the local feature for better nuclei detection and classification. By decoupling two simultaneous challenging tasks and taking advantage of JPFM, our method can benefit from class-aware detection and class-agnostic segmentation, thus leading to a significant performance boost. We demonstrate the superior performance of our proposed approach for nuclei segmentation and classification across 19 different tissue types, delivering new benchmark results.

## News:

\[2023/5/1\] We release the training and inference code, and the training instruction.


## 1. Installation

Clone this repo.
```bash
git clone https://github.com/Kaiseem/PointNu-Net.git
cd PointNu-Net/
```

This code requires PyTorch 1.10+ and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```


## 2. Data preparation

For small dataset Kumar and CoNSeP, we conduct datasets preparation following [Hover-Net](https://github.com/vqdang/hover_net).

We provide the [processed Kumar and CoNSeP datasets](https://drive.google.com/file/d/1_eI_ii6xcNe_77NWx7Qo8_KndK5UwPBO/view?usp=sharing). 

The [PanNuKe](https://arxiv.org/pdf/2003.10778v7.pdf) datasets can be found [here](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke)

Download and unzip all the files where the folder structure should look this:

```none
PointNu-Net
├── ...
├── datasets
│   ├── kumar
│   │   ├── train
│   │   ├── test
│   ├── CoNSeP
│   │   ├── train
│   │   ├── test
│   ├── PanNuKe
│   │   ├── images
│   │   │   ├── fold1
│   │   │   │   ├── images.npy
│   │   │   │   ├── types.npy
│   │   │   ├── fold2
│   │   │   │   ├── images.npy
│   │   │   │   ├── types.npy
│   │   │   ├── fold3
│   │   │   │   ├── images.npy
│   │   │   │   ├── types.npy
│   │   ├── masks
│   │   │   ├── fold1
│   │   │   │   ├── masks.npy
│   │   │   ├── fold2
│   │   │   │   ├── masks.npy
│   │   │   ├── fold3
│   │   │   │   ├── masks.npy
├── ...
```

## 3. Training and Inference
To reproduce the performance, you need one 3090 GPU at least


<details>
  <summary>
    <b>1) Kumar Dataset</b>
  </summary>
  
run the command to train the model
```bash
python train.py --name=kumar_exp --seed=888 --config=configs/kumar_notype_large.yaml
```

run the command to inference
```bash
python inference.py --name=kumar_exp
```
</details>

<details>
  <summary>
    <b>2) CoNSeP Dataset</b>
  </summary>
  
run the command to train the model
```bash
python train.py --name=consep_exp --seed=888 --config=configs/consep_type_large.yaml
```

run the command to inference
```bash
python inference.py --name=consep_exp
```
</details>


<details>
  <summary>
    <b>2)  PanNuKe Dataset</b>
  </summary>
  
run the command to train the model
```bash
python train_pannuke.py --name=pannuke_exp --seed=888 --train_fold={} --val_fold={} --test_fold={}
```
[train_fold, val_fold, test_fold] should be selected from {[1, 2, 3], [2, 1, 3], [3, 2, 1]}

run the command to inference the model
```bash
python infer_pannuke.py --name=pannuke_exp --train_fold={} --test_fold={}
```

run the command to evaluate the performance
```bash
python eval_pannuke.py --name=pannuke_exp --train_fold={} --test_fold={}
```

</details>

The pretrained model shall be released soon.

## Citation
If our work or code helps you, please consider to cite our paper. Thank you!

```
@article{yao2021pointnu,
  title={PointNu-Net: Keypoint-assisted Convolutional Neural Network for Simultaneous Multi-tissue Histology Nuclei Segmentation and Classification},
  author={Yao, Kai and Huang, Kaizhu and Sun, Jie and Hussain, Amir and Jude, Curran},
  journal={arXiv preprint arXiv:2111.01557},
  year={2021}
}
```