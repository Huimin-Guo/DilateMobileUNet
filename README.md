# DilateMobileU-Net: An Efficient Hybrid Segmentation Model for Polyp Diagnoses
> Authors: Huimin Guo, Yin Gu, Wu Du, Boyang Chen, Taiwei Jiao, Wei Qian and He Ma
>

# 1. Overview
Accurate polyp segmentation is significant in colon cancer diagnoses. While deep learning has enhanced polyp segmentation in recent years, these advancements often rely on models with complex structures and numerous parameters. Rapid and precise lesion detection is crucial for aiding doctors during endoscopic examinations, presenting significant challenges in terms of the segmentation model’s inference speed and parameter efficiency. This paper proposes an efficient hybrid segmentation model called DilateMobileU-Net for polyp diagnoses. MobileNetv2 (MV2) and DilateMobileFormer (DMF) blocks are stacked in the encoder to promote the learning ability of the local and global representations. To assess the model’s robustness under varying conditions, we utilize three different-depth encoders. The decoder, comprising the convolutional layers with the residual blocks, decodes the multi-scale features and generates the final polyp segmentation. Extensive experiments were conducted on two public datasets, i.e., Kvasir-SEG and CVC-ClinicDB datasets. Experimental results have proven that DilateMobileU-Net achieves the best results in terms of segmentation accuracy on the Kvasir-SEG dataset, with DSC and IoU of 92.56 % and 88.57 %, and reaches the competitive performance on the CVC-ClinicDB, with DSC and IoU of 92.45 % and 87.58 %, respectively. The model balances inference speed with accuracy, achieving precise polyp segmentation with minimal model parameters.


# 2. Usage
In this instance, we utilize the Kvasir-SEG dataset as our example.

## 2.1 Train
```shell
. ./scripts/kv_seg/train_dilatemobile_256.sh
```

## 2.2 Test
```shell
. ./scripts/kv_seg/test_dilatemobile_256.sh
```

## 3. License
The source code is free for research and education use only. Any commercial use should get formal permission first.
