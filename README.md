# scMomer

## scMomer: A modality-aware pretraining framework for single-cell multi-omics modeling under missing modality conditions

We develop scMomer, a modality-aware pretraining framework designed for multi-modal representation learning under missing modality conditions.


<p align="center">
<img src="https://github.com/nobody927/scMomer/blob/main/fig/main.png">
</p>

**Table of Contents**

* [Datasets](#Datasets)
* [Installation](#Installation)
* [Tutorials](#Tutorials)


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

### Three step pretraining

#### I. Single modality pretraining
In the first stage, we adopt masked modeling strategies to capture intra-modality interactions (Fig. 1b,c). 

For RNA, this involves predicting the expression of masked genes based on the context of co-expressed genes within the same cell. 

~~~shell
python -m torch.distributed.run pretrain_rna.py --data_path "pretrain_data_path"
~~~

For ATAC, the model predicts the accessibility of masked chromatin patches using information from surrounding genomic regions.

~~~shell
python -m torch.distributed.run pretrain_atac.py --data_path "pretrain_data_path" 
~~~

#### II. Multimodal interaction learning
The pretrained unimodal encoders are jointly fine-tuned using pseudo-paired RNA–ATAC profiles collected from atlas-scale datasets.

~~~shell
python -m torch.distributed.run pretrain_multimodal.py --data_path "pretrain_data_path" --atac_model_path "pretrained_atac_model_path" --rna_model_path "pretrained_rna_model_path"
~~~

#### III. Missing modality adaptation

The student is trained to approximate the missing modal embeddings produced by the modality-specific encoder, enabling inference when only one modality is available.

~~~shell
python get_distill.py --data_path "pretrain_data_path" --model_path "pretrained_model_path" 
~~~

~~~shell
python -m torch.distributed.run pretrain_missing_mod.py --data_path "pretrain_data_path" --model_path "pretrained_model_path" --distil_train "distilled_knowledge_train" --distil_val "distilled_knowledge_val"
~~~


## Tutorials

The tutorials will be available soon.

