# scMomer

## scMomer: A generic foundation model for single-cell multi-omics data with modality incompleteness

We develop scMomer, a generic foundation model for multi-modal cellular representation learning, with a particular focus on handling incomplete modalities in single-cell multi-omics data.

**Table of Contents**

* [Datasets](#Datasets)
* [Installation](#Installation)
* [Tutorials](#Tutorials)


## Datasets

### Pretraining Data
*  The atlas-level multi-modal dataset used in this study is available at https://www.dropbox.com/scl/fi/llehgmu928ii83u7jc8u9/fetal.h5mu?rlkey=e6h8d5l8fma7m2pzhxk8wqec7&dl=0.

### Gene-level Tasks
For gene-level tasks, the datasets used in scMomer are as follows:
*  The gene property prediction datasets are avilable at Geneformer (https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/gene_classification)
*  The gene-gene interaction dataset is available at https://github.com/jingcheng-du/Gene2vec/tree/master/predictionData.
*  The protein-protein interaction dataset is available at https://interactome-atlas.org/download.

### Cell-level Tasks
For cell-level tasks, the datasets used in scMomer are as follows:
*  Human brain multimodal dataset: It is available at https://cf.10xgenomics.com/samples/cell-arc/2.0.0/human_brain_3k/human_brain_3k_filtered_feature_bc_matrix.h5.
*  Cardiomyocyte dataset: We used the dataset curated by Chen et al., which consists of a random 10% subset of the original dataset, available at [this google drive folder](https://drive.google.com/drive/folders/1LgFvJqWNq9BqHbuxB2tYf62kXs9KqL4t?usp=share_link).	
*  Aorta dataset: We used the dataset curated by Chen et al., which consists of a 20% random subset of the original dataset ([link](https://drive.google.com/drive/folders/1LgFvJqWNq9BqHbuxB2tYf62kXs9KqL4t?usp=share_link)).
*  Drug response dataset is available at https://github.com/kimmo1019/DeepCDR.
* Perturbation datasets is available at https://github.com/snap-stanford/GEARS.



## Installation

To reproduce scMomer, we suggest first creating a conda environment by:

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

## Tutorials

### Data Preprocessing

Following scBERT, the single-cell transcriptomics data must first be pre-processed by updating the gene symbols and then normalized using `sc.pp.normalize_total` followed by `sc.pp.log1p`; details are provided in `preprocess.py`.

### Pretraining Workflow

#### Step I. Single Modality Pretraining
In the first stage, we adopt masked modeling strategies to capture intra-modality interactions (Fig. 1b,c). 

For RNA, this involves predicting the expression of masked genes based on the context of co-expressed genes within the same cell. 

~~~shell
python -m torch.distributed.run pretrain_rna.py --data_path "Pretrain_data_path" --ckpt_dir “Directory of checkpoint to save” --model_name “Pretrained model name”
~~~

For ATAC, the model predicts the accessibility of masked chromatin patches using information from surrounding genomic regions.

~~~shell
python -m torch.distributed.run pretrain_atac.py --data_path "Pretrain_data_path" --ckpt_dir “Directory of checkpoint to save” --model_name “Pretrained model name”
~~~

#### Step II. Multimodal Interaction Learning
The pretrained unimodal encoders are jointly fine-tuned using paired RNA–ATAC profiles:

~~~shell
python -m torch.distributed.run pretrain_multimodal.py --data_path "Pretrain_data_path" --atac_model_path "Pretrained_atac_model_path" --rna_model_path "Pretrained_rna_model_path" --ckpt_dir “Directory of checkpoint to save” --model_name “Pretrained model name”
~~~

#### Step III. Missing Modality Adaptation
To enable inference when one modality is absent, a student encoder is trained to approximate the embeddings of the missing modality. Specifically, we compute ATAC embeddings using the pretrained multimodal model as distilled knowledge. These input–output pairs are then used as training data to retrain the student ATAC encoder, with the mean squared error serving as the loss function.

First, we extract ATAC embeddings using the pretrained multimodal model:
~~~shell
python -m torch.distributed.run get_latent_emb.py --data_path "Pretrain_data_path" --model_path "Pretrained_multimodel_path" --ckpt_dir “Directory of ATAC embedding to save”
~~~

Then, train the student encoder with distilled knowledge:

~~~shell
python -m torch.distributed.run pretrain_missing_mod.py --data_path "Pretrain_data_path" --model_path "Pretrained_multimodal_model_path" --dir_train "Distilled_knowledge_train" --dir_val "Distilled_knowledge_val" --ckpt_dir “Directory of checkpoint to save” --model_name “Pretrained unimodal model name”
~~~

### Downstream Application
Take cell annotation as example, run 'train_missing_celltype_func.py' for finetuning:

~~~shell
python -m torch.distributed.run train_missing_celltype_func.py --data_path "Finetune_data_path" --pretrained_model_path "Pretrained_model_path" --ckpt_dir “Directory of checkpoint to save” --model “Finetuned model name”
~~~

Then, predict using finetuned models:

~~~shell
python predict_cell_type.py --data_path "data_path" --model_path "Finetuned model name" --ckpt_dir "Directory of finetuned model"
~~~

Due to a server attack, the system is now recovering; additional examples will be updated once it stabilizes.
