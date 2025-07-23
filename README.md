# scMomer


## This repo is being updated and will be ready shortly.

## scMomer: A modality-aware pretraining framework for single-cell multi-omics modeling under missing modality conditions

We develop scMomer, a modality-aware pretraining framework designed for multi-modal representation learning under missing modality conditions. scMomer adopts a three-stage pretraining strategy that learns unimodal cell representations, models joint representations from paired multi-omics data, and distills multi-modal knowledge to enable multi-omics-like representations from unimodal input. Its modality-specific architecture and three-stage pretraining strategy enable effective learning under missing modality conditions and help capture cellular heterogeneity. Through extensive experiments, scMomer generates biologically meaningful embeddings and outperforms state-of-the-art unimodal approaches across diverse gene-level and cell-level downstream tasks, including cross-modality translation, gene function prediction, cell annotation, drug response prediction, and perturbation prediction. Overall, these results demonstrate that scMomer serves as a robust, generalizable, and scalable foundation for single-cell multi-modal analysis under missing modality conditions.

<p align="center">
<img src="https://github.com/nobody927/scMomer/blob/main/fig/main.png">
</p>

**Table of Contents**

* [Datasets](#Datasets)
* [Installation](#Installation)
* [Tutorials](#Tutorials)
# * [Citation](#Citation)

## Datasets

For pretraining, we used the atlas-level multi-modal dataset, available at https://www.dropbox.com/scl/fi/llehgmu928ii83u7jc8u9/fetal.h5mu?rlkey=e6h8d5l8fma7m2pzhxk8wqec7&dl=0.

For gene-level tasks, the datasets used in scMomer are as follows:
	- The gene property prediction datasets are avilable at Geneformer (https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/gene_classification)
  - The gene-gene interaction dataset is available at https://github.com/jingcheng-du/Gene2vec/tree/master/predictionData. The protein-protein interaction dataset is available at https://interactome-atlas.org/download.

For cell-level tasks, we make use of the following datasets:
	- human brain multimodal dataset: 
    - It is available at https://cf.10xgenomics.com/samples/cell-arc/2.0.0/human_brain_3k/human_brain_3k_filtered_feature_bc_matrix.h5.
	- Cardiomyocyte dataset: 
		- We used the dataset curated by Chen et al., which consists of a random 10% subset of the original dataset, available at [this google drive folder](https://drive.google.com/drive/folders/1LgFvJqWNq9BqHbuxB2tYf62kXs9KqL4t?usp=share_link).	
  - Aorta dataset:
		- We used the dataset curated by Chen et al., which consists of a 20% random subset of the original dataset ([link](https://drive.google.com/drive/folders/1LgFvJqWNq9BqHbuxB2tYf62kXs9KqL4t?usp=share_link)).
  - Drug response dataset is available at https://github.com/kimmo1019/DeepCDR.

  - Perturbation datasets is available at https://github.com/snap-stanford/GEARS.



## Installation

To reproduce **scMomer**, we suggest first creating a conda environment by:

~~~shell
conda create -n scMomer python=3.9
conda activate scMomer
~~~

and then install the required packages below:


- numpy=1.19.*
- pandas=1.1.5
- scikit_learn=0.24.2
- transformers=4.6.1
- torch=1.9.0
- scanpy=1.10
- muon>=0.1.2
- pytorch-lightning<2.0
- scipy>=1.7.3
- einops>=0.6

## Usage

### Data preprocessing

Following scBERT, the single-cell transcriptomics test data must first be pre-processed by updating the gene symbols and then normalized using `sc.pp.normalize_total` followed by `sc.pp.log1p`; details are provided in  `preprocess.py`.

### - Zero-shot embedding


### - Fine-tune for downstream tasks


## Tutorials

The tutorials will be available soon.

