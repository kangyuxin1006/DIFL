# Domain Gap Reducing for Generalizable Medical Image Segmentation via Controlling Inductive Bias 

## Background

Deep Convolutional Neural Networks (DCNNs) have exhibited exceptional capability in medical image segmentation. However, their performance might significantly deteriorate when confronting testing data with the new distribution. Recent research indicates that a primary reason for this issue is the strong inductive bias of DCNNs, which tend to prioritize domain-specific features, such as superficial textures, over domain-invariant features like object shapes.

![Background](https://github.com/kangyuxin1006/DIJL/blob/main/Background.png)

## Method

We propose a novel method, named Domain Invariant Joint Learning (DIJL), aimed at enhancing the generalization ability of DCNNs to unseen datasets by controlling the inductive bias. Specifically, DIJL firstly blends the domain-specific information of training data to obtain the new features representation, thereby expanding the range of domains accessible for DCNNs training. Subsequently, leveraging this new features representation, we carefully devise a dual-branches domain invariant joint learning strategy to preserve predictions based on domain-invariant feature rather than domain-specific.

![Method](https://github.com/kangyuxin1006/DIJL/blob/main/Method.png)

## Result

Moreover, we validate the effectiveness of the proposed method on two typical medical segmentation tasks and perform in-depth analysis. Compared to other methods, it is evident that by minimizing the domain-specific inductive bias, our method produces predictions with smoother contours that are more closely aligned with the ground truth.

![Result](https://github.com/kangyuxin1006/DIJL/blob/main/Result.png)

## Usage

1. Clone the repository and download the [dataset](https://drive.google.com/drive/folders/1oxG-sDFBLkdvr8xqIs-KljCLzn410czO?usp=drive_link) into your own folder and change `--data-dir` correspondingly.

2. Train the model.

    ``` bash
    python train.py 
    ```
3. Test the model.

    ``` bash
    python test.py
    ```