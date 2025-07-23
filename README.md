# scMomer

## scMomer: A modality-aware pretraining framework for single-cell multi-omics modeling under missing modality conditions

Foundation models offer new opportunities to capture cellular behavior from large-scale single-cell data. However, their development has been greatly constrained due to the limited availability of multi-omics profiles. Consequently, most models are trained on a single modality (e.g. scRNA-seq, or scATAC-seq, etc.) and fail to capture the diversity of heterogeneous biological systems. Here, we introduce scMomer, a modality-aware pretraining framework designed for multi-modal representation learning under missing modality conditions. scMomer adopts a three-stage pretraining strategy that learns unimodal cell representations, models joint representations from paired multi-omics data, and distills multi-modal knowledge to enable multi-omics-like representations from unimodal input. Its modality-specific architecture and three-stage pretraining strategy enable effective learning under missing modality conditions and help capture cellular heterogeneity. Through extensive experiments, scMomer generates biologically meaningful embeddings and outperforms state-of-the-art unimodal approaches across diverse gene-level and cell-level downstream tasks, including cross-modality translation, gene function prediction, cell annotation, drug response prediction, and perturbation prediction. Overall, these results demonstrate that scMomer serves as a robust, generalizable, and scalable foundation for single-cell multi-modal analysis under missing modality conditions.

<p align="center">
<img src="https://github.com/nobody927/fig/main.png">
</p>
