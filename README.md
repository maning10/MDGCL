# **MDGCL**
## Multi-view dynamic graph contrastive learning for complex disease-noncoding RNA association prediction

* MDGCL is a multi-view dynamic graph contrastive learning model designed for predicting associations between complex diseases and noncoding RNAs. It integrates biological heterogeneous graph construction, multi-view representation extraction, dynamic feature aggregation, and contrastive-supervised joint learning to capture attribute and topology information for enhanced generalization and prediction performance.

## Key Features
* (1)Construction of the biological heterogeneous graph; 2) Multi-view representation extraction module;
* 3)Dynamic feature aggregation module; 4) Contrastive and supervised joint learning.

## Project Structure
1. code/
2.     data_preparation.py is used to calculate lncRNA/miRNA k-mer features and construct knn graph (attribute graph) of lncRNA/miRNA/disease.
3.     calculating_similarity.py is used to calclulate lncRNA/miRNA/disease GIPK similarities and obtain the intra-edges in the topology graph;
4.     parms_setting.pycontains hyperparmeters;
5.     utils.py contains preprocessing function of the data;
6.     data_preprocess.py contains the preprocess of data;
7.     layer.py contains MDGCL's model layer;
8.     train.py contains training and testing code;
9.     main.py runs MDGCL;

## Requirements
1. numpy == 1.21.1
2. torch == 2.0.0+cu118
3. sklearn == 0.24.1
4. torch-geometric == 2.3.0

* Install dependencies:
* pip install torch numpy scipy scikit-learn tqdm
* pip install torch-geometric

## Quick Start

* Here we provide a example to predict the lncRNA-disease association scores on dataset 1:
* Run main.py (in file-- dataset1/LDA.edgelist, neg_sample-- dataset1/non_LDA.edgelist, task_type--LDAl)

1. Default dataset: dataset1
2.     data/ contains the dataset 1 and dataset 2 used in the paper, with dataset 1 as an example;
3.     lnc(mi)_dis_association_new2.txt and lnc_mi_interaction_new2.txt contain known lncRNA(miRNA)-disease associations and lncRNA-miRNA interactions, respectively;
4.     LDA/MDA/LMI.edgelist contain known LDA, MDA, and LMI pairs, respectively; no_LDA/MDA/LMI.edgelist contain unknown LDA, MDA, LMI pairs;
5.     lncRNA/miRNA_sequences2.xlsx contain lncRNA and miRNA sequences, lncRNA sequences are from NCBI, miRNA sequences are from miRBase;
6.     disease_name.xlsx contains disease names and their DOID numbers;
7.     dis_sem_sim.txt contains disease semantic similarity data:

## Evaluation Metrics
* AUROC
* AUPR

