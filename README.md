# -QinggeLab_BICOP24
Generalizing deep learning models with Continual learning

Repository for my current work on Generallizing Deep Learning Models for COVID-19 Detection with Transfer and Continual Learning

Datasets:
Data Source

    https://www.kaggle.com/datasets/luisblanche/covidct (Generalization-dataset3)
    https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset (Generalization-dataset4)
    https://www.kaggle.com/datasets/mehradaria/covid19-lung-ct-scans (Training1)
    https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset (Training2)

    NB: 
    - Datasets 1 and 2 are merged toghter to train the initial model as the TASK1.
    - Datasets 3 and 4 are merged for TASK 2 where we seek to develop deep learning models which adapts to TASK2 whilst retaining knowledge of TASK 1 (via Transfer and Continual Learning)
    
Main files: 
- Cvd19_pred_using_CT_scan_CapsuleNet.ipynb: Generalizing Capsule Net on TASK1 and TASK2 using transfer learning. Thus, pre-trained weights of TASK1 model is used to generalize on TASK 2 model. TASK 2 model after transfer learning is evaluated on both TASKs 1 and 2.
- Cvd19_pred_using_CT_scan_CapsuleNet_EWC.ipynb:Continual learning approach using simplified EWC is applied to generalize Capsule Network model on TASKs 1 and 2 to mitigate castastrophic forgetting.
- Cvd19_pred_using_CT_scan_ViT.ipynb:Generalizing Vision Transformer (ViT) model on TASK1 and TASK2 using transfer learning. Thus, pre-trained weights of TASK1 model is used to generalize on TASK 2 model. TASK 2 model after transfer learning is evaluated on both TASKs 1 and 2.
- Cvd19_pred_using_CT_scan_ViT_EWC:Continual learning approach using simplified EWC is applied to generalize ViT model on TASKs 1 and 2 to mitigate castastrophic forgetting.
- Cvd19_pred_using_CT_scan_CNN.ipynb:Generalizing Convolutional Neural Network (CNN) model on TASK1 and TASK2 using transfer learning. Thus, pre-trained weights of TASK1 model is used to generalize on TASK 2 model. TASK 2 model after transfer learning is evaluated on both TASKs 1 and 2.
- Cvd19_pred_using_CT_scan_CNN_EWC:Continual learning approach using simplified EWC is applied to generalize CNN model on TASKs 1 and 2 to mitigate castastrophic forgetting.

Dependencies: 
    python 3.9
    tensorflow < 2.11
    MS Visual Studio 2019
    CUDA v.1.1X
    cuDNN v.8.1
    miniconda

    https://www.tensorflow.org/install/pip

    Installing tensorflow<2.11 with conda, which is the last version of tensorflow that can run on Windows Native
    As such, requires older versions of MS Visual Studio (2019), CUDA Toolkit 11.2, and cuDNN SDK 8.1.0

conda install:
    Install miniconda from the internet

    conda create --name tf python=3.9
    conda activate tf
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    pip install --upgrade pip
    pip install "tensorflow<2.11" 

