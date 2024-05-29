![image](https://github.com/changzy00/pytorch-attention/blob/master/images/logo.jpg)
# This codebase is a PyTorch implementation of various attention mechanisms, CNNs, Vision Transformers and MLP-Like models.

![](https://img.shields.io/badge/python->=v3.0-yellowgreen)

![](https://img.shields.io/badge/pytorch->=v1.5-yellowgreen)

If it is helpful for your work, please⭐

# Updating...

# Install
```shell
git clone https://github.com/changzy00/pytorch-attention.git
cd pytorch-attention
```
# Content

- [Attention Mechanisms](#Attention-mechanisms)
    - [1. Squeeze-and-Excitation Attention](#1-squeeze-and-excitation-attention)
    - [2. Convolutional Block Attention Module](#2-convolutional-block-attention-module)
    - [3. Bottleneck Attention Module](#3-Bottleneck-Attention-Module)
    - [4. Double Attention](#4-Double-Attention)
    - [5. Style Attention](#5-Style-Attention)
    - [6. Global Context Attention](#6-Global-Convtext-Attention)
    - [7. Selective Kernel Attention](#7-Selective-Kernel-Attention)
    - [8. Linear Context Attention](#8-Linear-Context-Attention)
    - [9. Gated Channel Attention](#9-gated-channel-attention)
    - [10. Efficient Channel Attention](#10-efficient-channel-attention)
    - [11. Triplet Attention](#11-Triplet-Attention)
    - [12. Gaussian Context Attention](#12-Gaussian-Context-Attention)
    - [13. Coordinate Attention](#13-coordinate-attention)
    - [14. SimAM](#14-SimAM)
    - [15. Dual Attention](#15-dual-attention)
 
- [Vision Transformers](#vision-transformers)
    - [1. ViT Model](#1-ViT-Model)
    - [2. XCiT Model](#2-XCiT-model)
    - [3. PiT Model](#3-pit-model)
    - [4. CvT Model](#4-cvt-model)
    - [5. PvT Model](#5-pvt-model)
    - [6. CMT Model](#6-cmt-model)
    - [7. PoolFormer Model](#7-poolformer-model)
    - [8. KVT Model](#8-kvt-model)
    - [9. MobileViT Model](#9-mobilevit-model)
    - [10. P2T Model](#10-p2t-model)
    - [11. EfficientFormer Model](#11-EfficientFormer-Model)
    - [12. ShiftViT Model](#12-shiftvit-model)
    - [13. CSWin Model](#13-CSWin-Model)
    - [14. DilateFormer Model](#14-DilateFormer-Model)
    - [15. BViT Model](#15-bvit-model)
    - [16. MOAT Model](#16-moat-model)
    - [17. SegFormer Model](#17-segformer-model)
    - [18. SETR Model](#18-setr-model)
    
- [Convolutional Neural Networks(CNNs)](#convolutional-neural-networks(cnns))
    - [1. NiN Model](#1-nin-model)
    - [2. ResNet Model](#2-resnet-model)
    - [3. WideResNet Model](#3-wideresnet-model)
    - [4. DenseNet Model](#4-densenet-model)
    - [5. PyramidNet Model](#5-pyramidnet-model)
    - [6. MobileNetV1 Model](#6-mobilenetv1-model)
    - [7. MobileNetV2 Model](#7-mobilenetv2-model)
    - [8. MobileNetV3 Model](#8-mobilenetv3-model)
    - [9. MnasNet Model](#9-mnasnet-model)
    - [10. EfficientNetV1 Model](#10-efficientnetv1-model)
    - [11. Res2Net Model](#11-res2net-model)
    - [12. MobileNeXt Model](#12-mobilenext-model)
    - [13. GhostNet Model](#13-ghostnet-model)
    - [14. EfficientNetV2 Model](#14-efficientnetv2-model)
    - [15. ConvNeXt Model](#15-convnext-model)
    - [16. Unet Model](#16-unet-model)
    - [17. ESPNet Model](#17-espnet-model)
    
- [MLP-Like Models](#mlp-like-models)
    - [1. MLP-Mixer Model](#1-mlp-mixer-model)
    - [2. gMLP Model](#2-gmlp-model)
    - [3. GFNet Model](#3-gfnet-model)
    - [4. sMLP Model](#4-smlp-model)
    - [5. DynaMixer Model](#5-dynamixer-model)
    - [6. ConvMixer Model](#6-convmixer-model)
    - [7. ViP Model](#7-vip-model)
    - [8. CycleMLP Model](#8-cyclemlp-model)
    - [9. Sequencer Model](#9-sequencer-model)
    - [10. MobileViG Model](#10-mobilevig-model)
    



## Attention Mechanisms
### 1. Squeeze-and-Excitation Attention
* #### Squeeze-and-Excitation Networks (CVPR 2018) [pdf](https://arxiv.org/pdf/1709.01507)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/senet.png)

* ##### Code
```python
import torch
from attention_mechanisms.se_module import SELayer

x = torch.randn(2, 64, 32, 32)
attn = SELayer(64)
y = attn(x)
print(y.shape)

```
Squeeze-and-Excitation (SE) Attention is a mechanism used in deep learning, particularly within convolutional neural networks (CNNs), to improve their representational power by enabling the network to perform dynamic channel-wise feature recalibration. This method was introduced by Jie Hu, Li Shen, and Gang Sun in their 2018 paper titled "Squeeze-and-Excitation Networks."

### How SE Attention Works

The SE block operates in two main steps: **squeeze** and **excitation**.

1. **Squeeze**:
   - In the squeeze step, global information is captured by applying global average pooling across the spatial dimensions of the feature maps.
   - For a given input feature map with shape \( H \times W \times C \) (height, width, and channels), the squeeze operation reduces it to a \( 1 \times 1 \times C \) feature vector by averaging each channel's spatial dimensions:
     \[
     z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{ijc}
     \]
   - Here, \( z_c \) is the c-th element of the squeezed feature vector, representing the global information of the c-th channel.

2. **Excitation**:
   - The excitation step involves learning a set of per-channel modulation weights to recalibrate the channels. This is achieved using a simple gating mechanism with a sigmoid activation function.
   - The squeezed feature vector \( z \) is passed through a small fully connected (FC) neural network with a bottleneck architecture (using two FC layers with a non-linear activation in between):
     \[
     s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))
     \]
   - Here, \( W_1 \) and \( W_2 \) are weight matrices, and \( \sigma \) denotes the sigmoid function, resulting in an output vector \( s \) of the same length \( C \).
   - The resulting vector \( s \) contains weights that highlight the importance of each channel.

3. **Recalibration**:
   - Finally, the original input feature map is recalibrated by scaling each channel with the corresponding weight from the excitation vector \( s \):
     \[
     \tilde{x}_{ijc} = s_c \cdot x_{ijc}
     \]
   - This scaling operation emphasizes important channels and suppresses less important ones, thereby improving the network's ability to model complex dependencies.

### Benefits of SE Attention

- **Improved Performance**: SE blocks have been shown to significantly enhance the performance of various CNN architectures on image classification tasks by allowing the network to focus on more informative features.
- **Lightweight and Easy to Integrate**: SE blocks are computationally lightweight and can be easily integrated into existing network architectures, making them a versatile addition.
- **Versatility**: While initially applied to image classification networks, SE blocks have been adapted for other tasks, such as object detection and segmentation.

### Applications

SE Attention has been successfully integrated into various CNN architectures, such as ResNet, Inception, and MobileNet, resulting in improved accuracy on standard benchmarks like ImageNet.

Overall, the Squeeze-and-Excitation mechanism provides an effective and efficient way to enhance the representational capabilities of CNNs by focusing on channel-wise feature interdependencies.

### 2. Convolutional Block Attention Module
* #### CBAM: convolutional block attention module (ECCV 2018) [pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/cbam.png)

* ##### Code
```python
import torch
from attention_mechanisms.cbam import CBAM

x = torch.randn(2, 64, 32, 32)
attn = CBAM(64)
y = attn(x)
print(y.shape)
```
The Convolutional Block Attention Module (CBAM) is an attention mechanism designed to enhance the representational power of convolutional neural networks (CNNs) by focusing on both the spatial and channel-wise features of the input. Introduced by Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon in their 2018 paper titled "CBAM: Convolutional Block Attention Module," CBAM integrates two sequential sub-modules: the Channel Attention Module and the Spatial Attention Module.

### How CBAM Works

CBAM applies attention in two main stages: **Channel Attention** and **Spatial Attention**.

1. **Channel Attention Module**:
   - The Channel Attention Module focuses on refining the importance of each channel by capturing the global information of the input feature map.
   - Two different pooling operations, average pooling and max pooling, are applied along the spatial dimension to generate two different spatial context descriptors.
   - These descriptors are then passed through a shared multi-layer perceptron (MLP) with one hidden layer. The outputs of the MLPs are combined and passed through a sigmoid function to obtain the channel attention map.
   - Formally, given an input feature map \( F \in \mathbb{R}^{H \times W \times C} \):
     \[
     M_c(F) = \sigma(\text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F)))
     \]
   - The channel attention map \( M_c \in \mathbb{R}^{1 \times 1 \times C} \) is then multiplied with the input feature map to produce a refined feature map.

2. **Spatial Attention Module**:
   - The Spatial Attention Module focuses on highlighting important spatial regions within the feature maps.
   - Similar to the Channel Attention Module, two different pooling operations (average pooling and max pooling) are applied along the channel dimension to generate two different channel context descriptors.
   - These descriptors are concatenated and passed through a convolution layer followed by a sigmoid function to generate the spatial attention map.
   - Formally, given the refined feature map from the Channel Attention Module \( F' \in \mathbb{R}^{H \times W \times C} \):
     \[
     M_s(F') = \sigma(f^{7 \times 7}([\text{AvgPool}(F'); \text{MaxPool}(F')]))
     \]
   - The spatial attention map \( M_s \in \mathbb{R}^{H \times W \times 1} \) is then multiplied with the refined feature map to produce the final output.

### Benefits of CBAM

- **Improved Performance**: CBAM enhances the performance of CNNs on various tasks by focusing on both channel and spatial features, leading to more discriminative feature representations.
- **Flexibility**: CBAM is a lightweight and flexible module that can be seamlessly integrated into different CNN architectures with minimal computational overhead.
- **Versatility**: While primarily used for image classification, CBAM has been successfully applied to other tasks, such as object detection and semantic segmentation.

### Applications

CBAM has been integrated into various state-of-the-art CNN architectures, such as ResNet, DenseNet, and MobileNet, demonstrating improved accuracy on benchmark datasets like ImageNet and COCO.

### Summary

The Convolutional Block Attention Module (CBAM) is a powerful attention mechanism that improves the representational capacity of CNNs by applying sequential channel and spatial attention. By refining feature maps to focus on more informative channels and spatial locations, CBAM helps CNNs achieve better performance on a range of computer vision tasks.

### 3. Bottleneck Attention Module
* #### Bam: Bottleneck attention module(BMVC 2018) [pdf](http://bmvc2018.org/contents/papers/0092.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/bam.png)

* ##### Code
```python
import torch
from attention_mechanisms.bam import BAM

x = torch.randn(2, 64, 32, 32)
attn = BAM(64)
y = attn(x)
print(y.shape)
```

The Bottleneck Attention Module (BAM) is an attention mechanism designed to enhance the representational power of convolutional neural networks (CNNs) by incorporating attention in a lightweight manner. BAM was introduced by Jongchan Park, Sanghyun Woo, Joon-Young Lee, and In So Kweon in their 2018 paper titled "BAM: Bottleneck Attention Module."

### How BAM Works

BAM integrates attention into CNNs by using both spatial and channel attention mechanisms in a bottleneck architecture. The key idea is to refine feature representations by focusing on important spatial regions and channels while maintaining computational efficiency.

BAM consists of two main components: **Channel Attention** and **Spatial Attention**, which are applied sequentially to the input feature maps.

1. **Channel Attention Module**:
   - The Channel Attention Module aims to highlight important channels.
   - Global average pooling is applied to the input feature map to generate a channel descriptor.
   - The descriptor is passed through a small multi-layer perceptron (MLP) with one hidden layer and a ReLU activation function, followed by a sigmoid function to produce the channel attention map.
   - Formally, given an input feature map \( F \in \mathbb{R}^{H \times W \times C} \):
     \[
     M_c(F) = \sigma(\text{MLP}(\text{AvgPool}(F)))
     \]
   - The channel attention map \( M_c \in \mathbb{R}^{1 \times 1 \times C} \) is then used to modulate the input feature map by channel-wise multiplication.

2. **Spatial Attention Module**:
   - The Spatial Attention Module focuses on important spatial locations.
   - Average pooling and max pooling are applied along the channel dimension to generate spatial context descriptors.
   - These descriptors are concatenated and passed through a convolutional layer with a \( k \times k \) filter (typically \( k = 7 \)), followed by a sigmoid function to produce the spatial attention map.
   - Formally, for the modulated feature map \( F' \in \mathbb{R}^{H \times W \times C} \) from the Channel Attention Module:
     \[
     M_s(F') = \sigma(f^{k \times k}([\text{AvgPool}(F'); \text{MaxPool}(F')]))
     \]
   - The spatial attention map \( M_s \in \mathbb{R}^{H \times W \times 1} \) is then used to modulate the feature map spatially.

### Sequential Application

The final output of the BAM is obtained by sequentially applying the channel and spatial attention mechanisms:
   \[
   \hat{F} = F \times M_c(F) \times M_s(F')
   \]
Where \( \hat{F} \) is the refined feature map, \( F \) is the original input feature map, \( M_c(F) \) is the channel attention map, and \( M_s(F') \) is the spatial attention map applied to the feature map modulated by channel attention.

### Benefits of BAM

- **Improved Performance**: BAM enhances the representational capacity of CNNs by focusing on both important channels and spatial locations, leading to better performance on various tasks.
- **Computational Efficiency**: BAM is designed to be lightweight, making it suitable for integration into existing CNN architectures without significant computational overhead.
- **Ease of Integration**: BAM can be easily incorporated into different layers of CNNs, providing flexibility in network design.

### Applications

BAM has been shown to improve the performance of CNNs on various tasks, including image classification, object detection, and segmentation, when integrated into architectures like ResNet and DenseNet.

### Summary

The Bottleneck Attention Module (BAM) is a powerful attention mechanism that enhances CNN performance by focusing on critical channels and spatial regions within feature maps. By combining channel and spatial attention in a computationally efficient manner, BAM provides a practical solution for improving the representational power of deep learning models.

### 4. Double Attention
* #### A2-nets: Double attention networks (NeurIPS 2018) [pdf](https://arxiv.org/pdf/1810.11579)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/a2net.png)

* ##### Code
```python
import torch
from attention_mechanisms.double_attention import DoubleAttention

x = torch.randn(2, 64, 32, 32)
attn = DoubleAttention(64, 32, 32)
y = attn(x)
print(y.shape)
```
Double Attention, also known as A2-Nets (Augmented Attention Networks), is an advanced attention mechanism designed to enhance the ability of neural networks to capture and model long-range dependencies in data. Introduced by Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He in their 2018 paper "Non-local Neural Networks," Double Attention focuses on improving the efficiency and effectiveness of attention mechanisms, particularly in handling large-scale feature maps common in computer vision tasks.

### How Double Attention Works

Double Attention employs a two-step process to compute attention: **Gathering** and **Distributing**. This mechanism ensures that the network captures both spatial and channel-wise dependencies more effectively.

1. **Gathering**:
   - In the gathering step, feature information from all spatial locations is aggregated to form a compact representation.
   - Given an input feature map \( X \in \mathbb{R}^{H \times W \times C} \), three different projections are applied to create key \( K \), query \( Q \), and value \( V \) feature maps.
   - The attention map is computed by multiplying the query \( Q \) with the key \( K \):
     \[
     A = \text{softmax}(QK^T)
     \]
   - This results in an attention map \( A \in \mathbb{R}^{HW \times HW} \), capturing the relationships between all pairs of spatial locations.

2. **Distributing**:
   - In the distributing step, the aggregated information is distributed back to each spatial location to enhance the feature representation.
   - The value feature map \( V \) is multiplied by the attention map \( A \):
     \[
     O = AV
     \]
   - The result \( O \in \mathbb{R}^{HW \times C} \) is then reshaped back to the original feature map dimensions \( \mathbb{R}^{H \times W \times C} \).

3. **Combining with Input**:
   - The final output is obtained by combining the attention-modulated feature map \( O \) with the original input feature map \( X \):
     \[
     Y = X + O
     \]
   - This residual connection ensures that the model retains the original features while incorporating the attention-enhanced information.

### Benefits of Double Attention

- **Capturing Long-Range Dependencies**: Double Attention is particularly effective at modeling long-range dependencies, which are crucial for tasks requiring a global understanding of the input data.
- **Efficient Computation**: Despite its complexity, Double Attention is designed to be computationally efficient, making it suitable for high-resolution inputs common in vision tasks.
- **Enhanced Representational Power**: By focusing on both spatial and channel-wise dependencies, Double Attention provides a richer feature representation, leading to improved performance on various tasks.

### Applications

Double Attention has been successfully applied in various domains, including:

- **Image Classification**: Enhancing the performance of CNN architectures by capturing more comprehensive feature relationships.
- **Object Detection and Segmentation**: Improving the ability to detect and segment objects by modeling the dependencies between different regions of the image.
- **Video Understanding**: Capturing temporal dependencies in video data for tasks such as action recognition.

### Summary

Double Attention (A2-Nets) is an advanced attention mechanism that enhances neural networks' ability to model long-range dependencies by using a two-step process of gathering and distributing feature information. This mechanism is particularly useful in computer vision tasks, where it helps to capture complex relationships within the data efficiently and effectively. By integrating Double Attention into existing architectures, significant performance improvements can be achieved across various applications.

### 5. Style Attention
* #### Srm : A style-based recalibration module for convolutional neural networks (ICCV 2019)  [pdf](https://arxiv.org/pdf/1903.10829)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/srm.png)

* ##### Code
```python
import torch
from attention_mechanisms.srm import SRM

x = torch.randn(2, 64, 32, 32)
attn = SRM(64)
y = attn(x)
print(y.shape)
```

Style Attention is an attention mechanism that is designed to enhance the transfer and manipulation of styles in neural networks, particularly within the context of style transfer tasks. Style Attention focuses on learning and applying style features effectively from one domain (e.g., an artwork) to another (e.g., a photograph), ensuring that the stylized output retains the essential content of the original image while adopting the desired stylistic characteristics.

### Key Concepts of Style Attention

1. **Style Feature Extraction**:
   - Style Attention mechanisms first extract features that capture the stylistic elements of an image. These features are typically derived from pre-trained convolutional neural networks (CNNs) known to be effective at capturing image textures and patterns.
   - Common layers used for extracting style features are from deep networks like VGG-19, where lower layers capture fine details (e.g., brush strokes), and higher layers capture more abstract patterns.

2. **Attention Mechanism**:
   - The core idea of Style Attention is to create an attention map that emphasizes the most important style features and modulates the content image based on these features.
   - This involves computing correlations between style features and the corresponding regions in the content image. The attention map identifies which style features should be applied to which parts of the content image.

3. **Feature Modulation**:
   - Once the attention map is computed, it is used to modulate the content image's feature maps. This process involves applying the style features selectively, guided by the attention map, to ensure that the style transfer is coherent and visually appealing.
   - This selective application ensures that the content structure is preserved while the style characteristics are superimposed effectively.

### How Style Attention Works

Here's a simplified breakdown of how a Style Attention mechanism might be implemented:

1. **Extract Style and Content Features**:
   - Given a style image and a content image, extract their respective feature maps using a pre-trained CNN.
   - Let \( F_s \) be the style feature map and \( F_c \) be the content feature map.

2. **Compute Attention Map**:
   - Calculate the attention map \( A \) by measuring the similarity between \( F_s \) and \( F_c \). This can be done using various methods, such as dot-product attention or more complex correlation measures.
   - The attention map \( A \) highlights which parts of the style feature map \( F_s \) correspond to which parts of the content feature map \( F_c \).

3. **Modulate Content Features**:
   - Use the attention map \( A \) to modulate the content features. This involves combining \( F_c \) with \( F_s \) in a way that the style features are applied according to the attention map.
   - The modulation can be expressed as:
     \[
     F_{modulated} = A \cdot F_s + (1 - A) \cdot F_c
     \]
   - Here, \( F_{modulated} \) represents the new feature map that combines content and style features.

4. **Generate Stylized Output**:
   - Finally, the modulated feature map \( F_{modulated} \) is passed through a decoder network to generate the final stylized image.

### Applications of Style Attention

- **Artistic Style Transfer**: Applying the visual style of famous artworks to everyday photographs while preserving the essential content of the photos.
- **Image-to-Image Translation**: Translating images from one domain to another, such as converting daytime photos to nighttime scenes or converting sketches to realistic images.
- **Content-Aware Image Manipulation**: Enhancing images by applying stylistic elements in a content-aware manner, such as adding artistic filters to photos.

### Benefits of Style Attention

- **Enhanced Stylization**: Provides a more nuanced and coherent style transfer by focusing on important style features and their appropriate application to the content image.
- **Content Preservation**: Maintains the structural integrity of the content image while applying stylistic elements, leading to more visually appealing results.
- **Flexibility**: Can be integrated into various neural network architectures and used for a wide range of image manipulation tasks.

### Summary

Style Attention is a specialized attention mechanism used to improve the quality and effectiveness of style transfer in neural networks. By focusing on important style features and applying them selectively to content images, Style Attention ensures that the stylized outputs are both visually appealing and true to the desired artistic style. This mechanism has broad applications in artistic style transfer, image-to-image translation, and other content-aware image manipulation tasks.
### 6. Global Context Attention
* #### Gcnet: Non-local networks meet squeeze-excitation networks and beyond (ICCVW 2019) [pdf](https://arxiv.org/pdf/1904.11492)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/gcnet.png)

* ##### Code
```python
import torch
from attention_mechanisms.gc_module import GCModule

x = torch.randn(2, 64, 32, 32)
attn = GCModule(64)
y = attn(x)
print(y.shape)
```

Global Context Attention (GCA) is an advanced attention mechanism designed to enhance neural networks' ability to capture and utilize global contextual information. GCA is particularly useful in computer vision tasks, where understanding the broader context of an image can significantly improve the performance of models on tasks such as image classification, object detection, and semantic segmentation.

### Key Concepts of Global Context Attention

1. **Global Context Modeling**:
   - GCA aims to capture the global dependencies and contextual information across the entire feature map, rather than just focusing on local or pairwise relationships.
   - By incorporating global context, GCA helps the model understand the broader scene, which is crucial for tasks that require holistic understanding.

2. **Contextual Feature Aggregation**:
   - This mechanism involves aggregating features from the entire spatial domain to create a global context representation.
   - Aggregated features provide a summary of the important information from different regions of the image, enhancing the model's ability to make informed predictions based on global context.

### How Global Context Attention Works

Global Context Attention typically involves the following steps:

1. **Feature Extraction**:
   - Extract features from an input image using a convolutional neural network (CNN). Let the feature map be \( F \in \mathbb{R}^{H \times W \times C} \), where \( H \) and \( W \) are the height and width, and \( C \) is the number of channels.

2. **Context Aggregation**:
   - Perform global average pooling or a similar operation to aggregate the features across the spatial dimensions. This produces a global context vector \( g \in \mathbb{R}^{C} \):
     \[
     g = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{ij}
     \]
   - This vector \( g \) captures the overall context of the entire feature map.

3. **Attention Generation**:
   - Generate an attention map using the global context vector \( g \). This involves passing \( g \) through a series of transformations, such as fully connected layers followed by non-linear activations and a sigmoid function.
   - The attention map \( A \in \mathbb{R}^{1 \times 1 \times C} \) is obtained, which represents the importance of each channel in the global context.

4. **Feature Modulation**:
   - Modulate the original feature map \( F \) using the attention map \( A \). This is typically done by element-wise multiplication:
     \[
     \hat{F}_{ij} = F_{ij} \cdot A
     \]
   - Here, \( \hat{F}_{ij} \) is the refined feature map that incorporates global contextual information.

5. **Combining Features**:
   - The modulated feature map \( \hat{F} \) can be combined with the original feature map \( F \) to enhance the features further, often using a residual connection:
     \[
     F_{out} = \hat{F} + F
     \]
   - This ensures that the original features are preserved while enhancing them with global context.

### Benefits of Global Context Attention

- **Improved Contextual Understanding**: By capturing global dependencies, GCA enhances the model's ability to understand the broader context of the input, leading to better performance on complex tasks.
- **Enhanced Feature Representation**: Modulating feature maps with global context helps in refining the feature representations, making them more robust and informative.
- **Flexibility**: GCA can be easily integrated into various neural network architectures, making it a versatile tool for different computer vision applications.

### Applications

Global Context Attention has been successfully applied in various areas, including:

- **Image Classification**: Enhancing classification models by providing a global understanding of the scene.
- **Object Detection**: Improving detection models by considering the context in which objects appear, leading to better localization and recognition.
- **Semantic Segmentation**: Refining segmentation models by incorporating global context, which helps in distinguishing between different classes more effectively.

### Summary

Global Context Attention (GCA) is an attention mechanism that enhances neural networks by incorporating global contextual information into feature representations. By aggregating and utilizing global features, GCA improves the model's ability to understand the broader scene, leading to better performance in tasks such as image classification, object detection, and semantic segmentation. GCA's ability to capture and leverage global dependencies makes it a powerful addition to modern deep learning architectures.

### 7. Selective Kernel Attention

* #### Selective Kernel Networks (CVPR 2019) [pdf](https://arxiv.org/abs/1903.06586)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/sknet.png)

* ##### Code
```python
import torch
from attention_mechanisms.sk_module import SKLayer

x = torch.randn(2, 64, 32, 32)
attn = SKLayer(64)
y = attn(x)
print(y.shape)
```
Selective Kernel Attention (SKA) is an advanced attention mechanism designed to improve the flexibility and adaptability of convolutional neural networks (CNNs) by allowing them to dynamically select different convolutional kernels for each input. This mechanism was introduced by Xiang Li, Wenhai Wang, Xiaolin Hu, and Jian Yang in their 2019 paper "Selective Kernel Networks."

### Key Concepts of Selective Kernel Attention

1. **Dynamic Kernel Selection**:
   - SKA enables the network to adaptively select appropriate convolutional kernels based on the input features. This dynamic selection allows the network to better capture various patterns and features in the input data.

2. **Multi-branch Convolution**:
   - The core idea of SKA involves using multiple branches with different convolutional kernels. Each branch has a distinct receptive field, capturing different aspects of the input features.

3. **Attention-based Selection**:
   - An attention mechanism is used to generate selection weights for each branch. These weights determine how much each branch contributes to the final output, allowing the network to focus on the most relevant features for each input.

### How Selective Kernel Attention Works

Selective Kernel Attention typically involves the following steps:

1. **Multi-branch Convolution**:
   - Given an input feature map \( F \in \mathbb{R}^{H \times W \times C} \), it is processed by multiple convolutional branches. Each branch \( i \) applies a different convolutional kernel \( K_i \):
     \[
     F_i = K_i * F
     \]
   - This results in a set of feature maps \( \{F_1, F_2, \ldots, F_n\} \), where \( n \) is the number of branches.

2. **Global Feature Aggregation**:
   - To generate the selection weights, global context information is aggregated from the input feature map. This is typically done using global average pooling, producing a global context vector \( g \):
     \[
     g = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{ij}
     \]
   - This vector \( g \) summarizes the important information from the entire input feature map.

3. **Attention Generation**:
   - The global context vector \( g \) is passed through a series of fully connected layers (MLPs) to produce the selection weights for each branch. A softmax function ensures that the weights sum to one:
     \[
     \alpha = \text{softmax}(\text{MLP}(g))
     \]
   - Here, \( \alpha = [\alpha_1, \alpha_2, \ldots, \alpha_n] \) are the selection weights for the branches.

4. **Feature Aggregation**:
   - The final output feature map is obtained by a weighted sum of the feature maps from each branch, using the selection weights \( \alpha \):
     \[
     F_{out} = \sum_{i=1}^{n} \alpha_i \cdot F_i
     \]
   - This weighted sum combines the contributions of each branch according to their relevance, as determined by the attention mechanism.

### Benefits of Selective Kernel Attention

- **Adaptive Receptive Fields**: By dynamically selecting different kernels, SKA allows the network to adapt its receptive fields to better capture relevant features for each input.
- **Improved Feature Representation**: The mechanism enhances the representational power of the network by enabling it to focus on the most informative features dynamically.
- **Flexibility and Robustness**: SKA can be integrated into various CNN architectures, improving their flexibility and robustness in handling diverse inputs.

### Applications

Selective Kernel Attention has been applied in various areas, including:

- **Image Classification**: Enhancing the performance of classification models by enabling them to adapt to different scales and patterns in images.
- **Object Detection**: Improving detection models by allowing them to better focus on relevant features across different object scales.
- **Semantic Segmentation**: Enhancing segmentation models by providing more adaptive and detailed feature representations.

### Summary

Selective Kernel Attention (SKA) is a powerful attention mechanism that enhances CNNs by enabling dynamic selection of convolutional kernels. By using multiple branches with different receptive fields and an attention mechanism to select the most relevant features, SKA improves the adaptability, flexibility, and overall performance of neural networks. This makes it a valuable tool for various computer vision tasks, including image classification, object detection, and semantic segmentation.

### 8. Linear Context Attention
* #### Linear Context Transform Block (AAAI 2020) [pdf](https://arxiv.org/pdf/1909.03834v2)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/lct.png)

* ##### Code
```python
import torch
from attention_mechanisms.lct import LCT

x = torch.randn(2, 64, 32, 32)
attn = LCT(64, groups=8)
y = attn(x)
print(y.shape)
```
Linear Context Attention (LCA) is an attention mechanism designed to efficiently capture and utilize contextual information in neural networks, particularly in the context of computer vision tasks. The primary goal of LCA is to enhance the representational capacity of models while maintaining computational efficiency.

### Key Concepts of Linear Context Attention

1. **Linear Complexity**:
   - LCA is designed to operate with linear complexity relative to the number of input elements, making it more efficient compared to traditional attention mechanisms that have quadratic complexity.

2. **Contextual Information**:
   - LCA focuses on capturing the global context by aggregating information from the entire input feature map. This global context is then used to modulate the features, allowing the model to focus on relevant parts of the input.

3. **Efficiency**:
   - By using linear operations, LCA ensures that the computational and memory overhead remains low, making it suitable for applications with large inputs or limited resources.

### How Linear Context Attention Works

Here's a simplified breakdown of how Linear Context Attention typically functions:

1. **Feature Extraction**:
   - Extract features from an input image using a convolutional neural network (CNN). Let the feature map be \( F \in \mathbb{R}^{H \times W \times C} \), where \( H \) and \( W \) are the height and width, and \( C \) is the number of channels.

2. **Context Aggregation**:
   - Perform a global pooling operation (such as global average pooling) to aggregate the features across the spatial dimensions. This results in a global context vector \( g \in \mathbb{R}^{C} \):
     \[
     g = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{ij}
     \]
   - This global context vector \( g \) summarizes the important information from the entire feature map.

3. **Attention Generation**:
   - Generate attention weights using the global context vector. This involves passing \( g \) through a series of linear transformations, typically using fully connected layers followed by a softmax function to ensure the weights sum to one:
     \[
     \alpha = \text{softmax}(Wg + b)
     \]
   - Here, \( \alpha \in \mathbb{R}^{C} \) are the attention weights for each channel.

4. **Feature Modulation**:
   - Modulate the original feature map \( F \) using the attention weights \( \alpha \). This is done by element-wise multiplication:
     \[
     F_{modulated} = F \times \alpha
     \]
   - The modulated feature map \( F_{modulated} \in \mathbb{R}^{H \times W \times C} \) incorporates the global context information.

5. **Combining Features**:
   - The final output is often obtained by combining the modulated feature map with the original feature map, typically using a residual connection:
     \[
     F_{out} = F + F_{modulated}
     \]
   - This ensures that the original features are preserved while enhancing them with the global context.

### Benefits of Linear Context Attention

- **Computational Efficiency**: LCA's linear complexity makes it more efficient than traditional attention mechanisms, which is particularly beneficial for large-scale inputs and resource-constrained environments.
- **Enhanced Contextual Understanding**: By capturing global context, LCA improves the model's ability to understand the broader scene, leading to better performance on tasks requiring holistic understanding.
- **Flexibility**: LCA can be easily integrated into various neural network architectures, providing a versatile tool for improving feature representations.

### Applications

Linear Context Attention can be applied to a range of tasks, including:

- **Image Classification**: Enhancing classification models by providing a global understanding of the scene.
- **Object Detection**: Improving detection models by incorporating contextual information to better localize and recognize objects.
- **Semantic Segmentation**: Refining segmentation models by incorporating global context, which helps distinguish between different classes more effectively.

### Summary

Linear Context Attention (LCA) is an efficient attention mechanism designed to capture and utilize global contextual information in neural networks with linear complexity. By aggregating global context and using it to modulate feature representations, LCA enhances the model's ability to understand the broader scene while maintaining computational efficiency. This makes it a valuable tool for various computer vision tasks, including image classification, object detection, and semantic segmentation.

### 9. Gated Channel Attention
* #### Gated Channel Transformation for Visual Recognition (CVPR 2020) [pdf](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/gate_channel.png)

* ##### Code
```python
import torch
from attention_mechanisms.gate_channel_module import GCT

x = torch.randn(2, 64, 32, 32)
attn = GCT(64)
y = attn(x)
print(y.shape)
```
Gated Channel Attention (GCA) is an attention mechanism designed to enhance the representational power of neural networks, particularly convolutional neural networks (CNNs), by focusing on the most relevant feature channels. GCA selectively emphasizes or suppresses feature channels based on their importance to the specific task at hand, improving the overall performance of the network.

### Key Concepts of Gated Channel Attention

1. **Channel-wise Attention**:
   - GCA operates at the channel level, assigning different levels of importance to each feature channel. This allows the network to focus on the most informative channels while ignoring less relevant ones.

2. **Gating Mechanism**:
   - The gating mechanism controls the flow of information through the channels. It uses learned gates to modulate the importance of each channel dynamically, enhancing the network's ability to focus on critical features.

3. **Context Aggregation**:
   - GCA aggregates global context information to inform the gating mechanism, ensuring that the attention weights reflect the overall importance of each channel in the context of the entire input.

### How Gated Channel Attention Works

Here’s a detailed breakdown of how Gated Channel Attention typically functions:

1. **Feature Extraction**:
   - Given an input feature map \( F \in \mathbb{R}^{H \times W \times C} \) (where \( H \) and \( W \) are the height and width, and \( C \) is the number of channels), extract features using a convolutional neural network (CNN).

2. **Global Context Aggregation**:
   - Perform global average pooling to aggregate features across the spatial dimensions, producing a global context vector \( g \in \mathbb{R}^{C} \):
     \[
     g = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{ij}
     \]
   - This vector \( g \) captures the overall importance of each channel.

3. **Gating Mechanism**:
   - Pass the global context vector \( g \) through a gating mechanism. This typically involves a series of fully connected layers (MLPs) followed by a sigmoid activation function to produce the gating weights \( \alpha \):
     \[
     \alpha = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot g + b_1) + b_2)
     \]
   - Here, \( \alpha \in \mathbb{R}^{C} \) represents the gating weights, \( W_1 \) and \( W_2 \) are weight matrices, and \( b_1 \) and \( b_2 \) are biases.

4. **Feature Modulation**:
   - Modulate the original feature map \( F \) using the gating weights \( \alpha \). This is done by element-wise multiplication:
     \[
     F_{modulated} = F \times \alpha
     \]
   - Each channel of the feature map \( F \) is scaled by its corresponding gating weight in \( \alpha \).

5. **Combining Features**:
   - The final output feature map is often obtained by combining the modulated feature map with the original feature map using a residual connection:
     \[
     F_{out} = F + F_{modulated}
     \]
   - This residual connection helps retain the original features while enhancing them with the gated attention.

### Benefits of Gated Channel Attention

- **Enhanced Feature Representation**: By selectively emphasizing the most important feature channels, GCA improves the quality of the feature representations.
- **Dynamic Adaptation**: The gating mechanism allows the network to adaptively focus on different features depending on the input, improving performance across various tasks.
- **Computational Efficiency**: GCA adds relatively little computational overhead, making it suitable for integration into existing CNN architectures without significantly increasing their complexity.

### Applications

Gated Channel Attention can be applied to a wide range of tasks, including:

- **Image Classification**: Enhancing the discriminative power of classification models by focusing on the most informative channels.
- **Object Detection**: Improving detection models by emphasizing relevant features for accurate localization and recognition of objects.
- **Semantic Segmentation**: Refining segmentation models by enhancing the representation of important features, leading to better differentiation between classes.

### Summary

Gated Channel Attention (GCA) is an attention mechanism that enhances neural network performance by dynamically modulating the importance of feature channels. By using a gating mechanism informed by global context, GCA selectively emphasizes the most relevant features, improving the representational capacity and adaptability of the network. This makes GCA a valuable tool for various computer vision tasks, including image classification, object detection, and semantic segmentation.

### 10. Efficient Channel Attention
* #### Ecanet: Efficient channel attention for deep convolutional neural networks (CVPR 2020) [pdf](https://arxiv.org/pdf/1910.03151)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/ecanet.png)

* ##### Code
```python
import torch
from attention_mechanisms.eca import ECALayer

x = torch.randn(2, 64, 32, 32)
attn = ECALayer(64)
y = attn(x)
print(y.shape)
```

Efficient Channel Attention (ECA) is an attention mechanism designed to improve the performance of convolutional neural networks (CNNs) by focusing on important feature channels efficiently. ECA aims to enhance the representational capacity of CNNs without significantly increasing computational complexity. This mechanism was introduced to address the limitations of other channel attention methods, such as the Squeeze-and-Excitation (SE) block, which can be computationally expensive.

### Key Concepts of Efficient Channel Attention

1. **Channel-wise Attention**:
   - ECA operates at the channel level, assigning different levels of importance to each feature channel. This helps the network focus on the most informative channels.

2. **Efficiency**:
   - ECA is designed to be computationally efficient, avoiding the complexity of fully connected layers used in other attention mechanisms. It achieves this by using a more straightforward approach to compute attention weights.

3. **Local Cross-channel Interaction**:
   - ECA employs a local cross-channel interaction strategy, which captures the dependencies between channels without requiring a large number of parameters or complex computations.

### How Efficient Channel Attention Works

Here is a detailed breakdown of how Efficient Channel Attention typically functions:

1. **Feature Extraction**:
   - Given an input feature map \( F \in \mathbb{R}^{H \times W \times C} \) (where \( H \) and \( W \) are the height and width, and \( C \) is the number of channels), the features are extracted using a convolutional neural network (CNN).

2. **Global Average Pooling**:
   - Perform global average pooling to aggregate the features across the spatial dimensions, producing a global context vector \( g \in \mathbb{R}^{C} \):
     \[
     g_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{ijc}
     \]
   - This vector \( g \) summarizes the information from the entire feature map for each channel.

3. **1D Convolution**:
   - Instead of using fully connected layers, ECA applies a 1D convolution over the channel dimension. This 1D convolution uses a kernel size \( k \) to capture local cross-channel interactions:
     \[
     \alpha = \sigma(\text{Conv1D}(g, k))
     \]
   - Here, \( \sigma \) is a sigmoid activation function, and \( \alpha \in \mathbb{R}^{C} \) are the attention weights. The kernel size \( k \) is typically chosen based on the channel dimension to balance the trade-off between complexity and capturing sufficient dependencies.

4. **Feature Modulation**:
   - Modulate the original feature map \( F \) using the attention weights \( \alpha \) by element-wise multiplication:
     \[
     F_{modulated} = F \times \alpha
     \]
   - Each channel of the feature map \( F \) is scaled by its corresponding attention weight in \( \alpha \).

5. **Combining Features**:
   - The final output feature map is often obtained by combining the modulated feature map with the original feature map using a residual connection:
     \[
     F_{out} = F + F_{modulated}
     \]
   - This residual connection helps retain the original features while enhancing them with the attention mechanism.

### Benefits of Efficient Channel Attention

- **Computational Efficiency**: ECA significantly reduces the computational overhead compared to other attention mechanisms by using 1D convolutions instead of fully connected layers.
- **Enhanced Feature Representation**: By focusing on important channels, ECA improves the quality of the feature representations, leading to better performance in various tasks.
- **Simplicity**: The simplicity of ECA makes it easy to integrate into existing CNN architectures without adding much complexity.

### Applications

Efficient Channel Attention can be applied to a variety of tasks, including:

- **Image Classification**: Enhancing the discriminative power of classification models by focusing on the most informative channels.
- **Object Detection**: Improving detection models by emphasizing relevant features for accurate localization and recognition of objects.
- **Semantic Segmentation**: Refining segmentation models by enhancing the representation of important features, leading to better differentiation between classes.

### Summary

Efficient Channel Attention (ECA) is an attention mechanism designed to improve the performance of CNNs by focusing on important feature channels in a computationally efficient manner. By employing global average pooling and 1D convolutions to capture local cross-channel interactions, ECA enhances the representational capacity of neural networks without significantly increasing computational complexity. This makes ECA a valuable tool for various computer vision tasks, including image classification, object detection, and semantic segmentation.

### 11. Triplet Attention

* #### Rotate to Attend: Convolutional Triplet Attention Module (WACV 2021) [pdf](http://arxiv.org/pdf/2010.03045)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/triplet.png)

* ##### Code
```python
import torch
from attention_mechanisms.triplet_attention import TripletAttention

x = torch.randn(2, 64, 32, 32)
attn = TripletAttention(64)
y = attn(x)
print(y.shape)
```
Triplet Attention is an advanced attention mechanism designed to enhance neural networks' ability to capture spatial and channel-wise interdependencies by processing the input features through three distinct attention branches. Unlike traditional attention mechanisms that might focus on a single aspect of the input data (such as channel attention or spatial attention), Triplet Attention simultaneously considers multiple dimensions, leading to more comprehensive feature representation.

### Key Concepts of Triplet Attention

1. **Multi-branch Structure**:
   - Triplet Attention consists of three separate branches, each focusing on a different aspect of the input features: channel attention, spatial attention along the height dimension, and spatial attention along the width dimension.

2. **Dimensionality Reduction**:
   - Each branch performs dimensionality reduction to capture essential features while keeping computational complexity manageable.

3. **Comprehensive Feature Interaction**:
   - By considering interactions across different dimensions (channels, height, width), Triplet Attention provides a richer and more detailed representation of the input features.

### How Triplet Attention Works

Here's a detailed breakdown of how Triplet Attention typically functions:

1. **Feature Extraction**:
   - Given an input feature map \( F \in \mathbb{R}^{H \times W \times C} \), where \( H \) and \( W \) are the height and width, and \( C \) is the number of channels, features are extracted using a convolutional neural network (CNN).

2. **Three Attention Branches**:
   - The input feature map is processed through three branches: 
     1. **Channel Attention Branch**: Captures channel-wise dependencies.
     2. **Height Attention Branch**: Captures spatial dependencies along the height dimension.
     3. **Width Attention Branch**: Captures spatial dependencies along the width dimension.

3. **Channel Attention Branch**:
   - Perform global average pooling along the spatial dimensions to produce a channel descriptor \( g_c \in \mathbb{R}^{C} \):
     \[
     g_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{ijc}
     \]
   - This vector \( g_c \) is used to compute channel attention weights, typically using a fully connected layer and a sigmoid activation function:
     \[
     \alpha_c = \sigma(W_c g_c + b_c)
     \]
   - The channel attention weights \( \alpha_c \in \mathbb{R}^{C} \) modulate the original feature map:
     \[
     F_c = F \cdot \alpha_c
     \]

4. **Height Attention Branch**:
   - Perform global average pooling along the width dimension to produce a height descriptor \( g_h \in \mathbb{R}^{H \times C} \):
     \[
     g_h = \frac{1}{W} \sum_{j=1}^{W} F_{ijc}
     \]
   - Apply convolutional layers followed by a sigmoid activation function to generate height attention weights \( \alpha_h \in \mathbb{R}^{H \times 1 \times C} \):
     \[
     \alpha_h = \sigma(\text{Conv}(g_h))
     \]
   - The height attention weights \( \alpha_h \) modulate the feature map along the height dimension:
     \[
     F_h = F \cdot \alpha_h
     \]

5. **Width Attention Branch**:
   - Perform global average pooling along the height dimension to produce a width descriptor \( g_w \in \mathbb{R}^{W \times C} \):
     \[
     g_w = \frac{1}{H} \sum_{i=1}^{H} F_{ijc}
     \]
   - Apply convolutional layers followed by a sigmoid activation function to generate width attention weights \( \alpha_w \in \mathbb{R}^{1 \times W \times C} \):
     \[
     \alpha_w = \sigma(\text{Conv}(g_w))
     \]
   - The width attention weights \( \alpha_w \) modulate the feature map along the width dimension:
     \[
     F_w = F \cdot \alpha_w
     \]

6. **Combining Features**:
   - Combine the outputs from the three attention branches to form the final output feature map. This can be done using element-wise addition or multiplication:
     \[
     F_{out} = F_c + F_h + F_w
     \]

### Benefits of Triplet Attention

- **Comprehensive Feature Representation**: By considering channel-wise, height-wise, and width-wise attention, Triplet Attention provides a richer and more detailed feature representation.
- **Improved Performance**: The multi-branch attention mechanism helps in capturing diverse aspects of the input features, leading to improved performance in tasks such as image classification, object detection, and semantic segmentation.
- **Scalability**: Despite the enhanced attention mechanism, Triplet Attention is designed to be efficient and scalable, making it suitable for various neural network architectures.

### Applications

Triplet Attention can be applied to a wide range of tasks, including:

- **Image Classification**: Enhancing the discriminative power of classification models by focusing on multiple aspects of the input features.
- **Object Detection**: Improving detection models by capturing diverse spatial and channel-wise dependencies for better localization and recognition.
- **Semantic Segmentation**: Refining segmentation models by providing a more comprehensive understanding of the spatial context and feature interactions.

### Summary

Triplet Attention is a sophisticated attention mechanism that enhances neural networks' ability to capture spatial and channel-wise interdependencies through three distinct attention branches. By simultaneously considering channel attention, height attention, and width attention, Triplet Attention provides a comprehensive and detailed feature representation, leading to improved performance across various computer vision tasks. This makes it a valuable addition to modern deep learning architectures, enhancing their ability to understand and process complex visual data.

### 12. Gaussian Context Attention
* #### Gaussian Context Transformer (CVPR 2021) [pdf](http://openaccess.thecvf.com//content/CVPR2021/papers/Ruan_Gaussian_Context_Transformer_CVPR_2021_paper.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/gct.png)

* ##### Code
```python
import torch
from attention_mechanisms.gct import GCT

x = torch.randn(2, 64, 32, 32)
attn = GCT(64)
y = attn(x)
print(y.shape)
```
### 13. Coordinate Attention

* #### Coordinate Attention for Efficient Mobile Network Design (CVPR 2021) [pdf](https://arxiv.org/abs/2103.02907)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/coordinate.png)

* ##### Code
```python
import torch
from attention_mechanisms.coordatten import CoordinateAttention

x = torch.randn(2, 64, 32, 32)
attn = CoordinateAttention(64, 64)
y = attn(x)
print(y.shape)
```
### 14. SimAM
* SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks (ICML 2021) [pdf](http://proceedings.mlr.press/v139/yang21o/yang21o.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/simam.png)

* ##### Code
```python
import torch
from attention_mechanisms.simam import simam_module

x = torch.randn(2, 64, 32, 32)
attn = simam_module(64)
y = attn(x)
print(y.shape)
```
### 15. Dual Attention
* #### Dual Attention Network for Scene Segmentatio (CVPR 2019)  [pdf](https://arxiv.org/pdf/1809.02983.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/danet.png)

* ##### Code
```python
import torch
from attention_mechanisms.dual_attention import PAM, CAM

x = torch.randn(2, 64, 32, 32)
#attn = PAM(64)
attn = CAM()
y = attn(x)
print(y.shape
```
## Vision Transformers
### 1. ViT Model
* #### An image is worth 16x16 words: Transformers for image recognition at scale (ICLR 2021) [pdf](https://arxiv.org/pdf/2010.11929)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/vit.png)

* ##### Code
```python
import torch
from vision_transformers.ViT import VisionTransformer

x = torch.randn(2, 3, 224, 224)
model = VisionTransformer()
y = model(x)
print(y.shape) #[2, 1000]
```
### 2. XCiT Model

* #### XCiT: Cross-Covariance Image Transformer (NeurIPS 2021) [pdf](https://arxiv.org/pdf/2106.09681)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/xcit.png)

* ##### Code
```python
import torch
from vision_transformers.xcit import xcit_nano_12_p16
x = torch.randn(2, 3, 224, 224)
model = xcit_nano_12_p16()
y = model(x)
print(y.shape)
```
### 3. PiT Model

* #### Rethinking Spatial Dimensions of Vision Transformers (ICCV 2021) [pdf](https://arxiv.org/abs/2103.16302)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/pit.png)

* ##### Code
```python
import torch
from vision_transformers.pit import pit_ti
x = torch.randn(2, 3, 224, 224)
model = pit_ti()
y = model(x)
print(y.shape)
```
### 4. CvT Model

* #### CvT: Introducing Convolutions to Vision Transformers (ICCV 2021) [pdf](https://arxiv.org/abs/2103.15808)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/cvt.png)

* ##### Code
```python
import torch
from vision_transformers.cvt import cvt_13
x = torch.randn(2, 3, 224, 224)
model = cvt_13()
y = model(x)
print(y.shape)
```
### 5. PvT Model

* #### Pyramid vision transformer: A versatile backbone for dense prediction without convolutions (ICCV 2021) [pdf](https://arxiv.org/abs/2102.12122)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/pvt.png)

* ##### Code
```python
import torch
from vision_transformers.pvt import pvt_t
x = torch.randn(2, 3, 224, 224)
model = pvt_t()
y = model(x)
print(y.shape)
```
### 6. CMT Model

* #### CMT: Convolutional Neural Networks Meet Vision Transformers (CVPR 2022) [pdf](http://arxiv.org/pdf/2107.06263)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/cmt.png)

* ##### Code
```python
import torch
from vision_transformers.cmt import cmt_ti
x = torch.randn(2, 3, 224, 224)
model = cmt_ti()
y = model(x)
print(y.shape)
```
### 7. PoolFormer Model

* #### MetaFormer is Actually What You Need for Vision (CVPR 2022) [pdf](https://arxiv.org/abs/2111.11418)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/poolformer.png)

* ##### Code
```python
import torch
from vision_transformers.poolformer import poolformer_12
x = torch.randn(2, 3, 224, 224)
model = poolformer_12()
y = model(x)
print(y.shape)
```
### 8. KVT Model

* #### KVT: k-NN Attention for Boosting Vision Transformers (ECCV 2022) [pdf](https://arxiv.org/abs/2106.00515)
* ##### Code
```python
import torch
from vision_transformers.kvt import KVT
x = torch.randn(2, 3, 224, 224)
model = KVT()
y = model(x)
print(y.shape)
```
### 9. MobileViT Model

* #### MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (ICLR 2022) [pdf](https://arxiv.org/abs/2110.02178)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/mobilevit.png)

* ##### Code
```python
import torch
from vision_transformers.mobilevit import mobilevit_s
x = torch.randn(2, 3, 224, 224)
model = mobilevit_s()
y = model(x)
print(y.shape)
```
### 10. P2T Model

* #### Pyramid Pooling Transformer for Scene Understanding (TPAMI 2022) [pdf](https://arxiv.org/abs/2106.12011)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/p2t.png)

* ##### Code
```python
import torch
from vision_transformers.p2t import p2t_tiny
x = torch.randn(2, 3, 224, 224)
model = p2t_tiny()
y = model(x)
print(y.shape)
```
### 11. EfficientFormer Model

* #### EfficientFormer: Vision Transformers at MobileNet Speed (NeurIPS 2022) [pdf](https://arxiv.org/abs/2212.08059)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/efficientformer.png)

* ##### Code
```python
import torch
from vision_transformers.efficientformer import efficientformer_l1
x = torch.randn(2, 3, 224, 224)
model = efficientformer_l1()
y = model(x)
print(y.shape)
```
### 12. ShiftViT Model

* #### When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism (AAAI 2022) [pdf](https://arxiv.org/abs/2201.10801)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/shiftvit.png)

* ##### Code
```python
import torch
from vision_transformers.shiftvit import shift_t
x = torch.randn(2, 3, 224, 224)
model = shift_t()
y = model(x)
print(y.shape)
```
### 13. CSWin Model

* #### CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows (CVPR 2022) [pdf](https://arxiv.org/pdf/2107.00652.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/cswin.png)

* ##### Code
```python
import torch
from vision_transformers.cswin import CSWin_64_12211_tiny_224
x = torch.randn(2, 3, 224, 224)
model = CSWin_64_12211_tiny_224()
y = model(x)
print(y.shape)
```
### 14. DilateFormer Model

* #### DilateFormer: Multi-Scale Dilated Transformer for Visual Recognition (TMM 2023) [pdf](https://arxiv.org/abs/2302.01791)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/dilateformer.png)

* ##### Code
```python
import torch
from vision_transformers.dilateformer import dilateformer_tiny
x = torch.randn(2, 3, 224, 224)
model = dilateformer_tiny()
y = model(x)
print(y.shape)
```
### 15. BViT Model

* #### BViT: Broad Attention based Vision Transformer (TNNLS 2023) [pdf](https://arxiv.org/abs/2202.06268)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/bvit.png)

* ##### Code
```python
import torch
from vision_transformers.bvit import BViT_S
x = torch.randn(2, 3, 224, 224)
model = BViT_S()
y = model(x)
print(y.shape)
```
### 16. MOAT Model

* #### MOAT: Alternating Mobile Convolution and Attention Brings Strong Vision Models (ICLR 2023) [pdf](https://arxiv.org/pdf/2210.01820.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/moat.png)

* ##### Code
```python
import torch
from vision_transformers.moat import moat_0
x = torch.randn(2, 3, 224, 224)
model = moat_0()
y = model(x)
print(y.shape)
```
### 17. SegFormer Model

* #### SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (NeurIPS 2021) [pdf](https://arxiv.org/abs/2105.15203)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/segformer.png)

* ##### Code
```python
import torch
from vision_transformers.moat import SegFormer
x = torch.randn(2, 3, 512, 512)
model = SegFormer(num_classes=50)
y = model(x)
print(y.shape)
```
### 18. SETR Model

* #### Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers (CVPR 2021) [pdf](https://arxiv.org/abs/2012.15840)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/setr.png)

* ##### Code
```python
import torch
from vision_transformers.setr import SETR
x = torch.randn(2, 3, 480, 480)
model = SETR(num_classes=50)
y = model(x)
print(y.shape)
```
## Convolutional Neural Networks(CNNs)
### 1. NiN Model
* #### Network In Network (ICLR 2014) [pdf](https://arxiv.org/pdf/1312.4400v3)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/nin.png)

* ##### Code
```python
import torch
from cnns.NiN import NiN 
x = torch.randn(2, 3, 224, 224)
model = NiN()
y = model(x)
print(y.shape)
```
### 2. ResNet Model
* #### Deep Residual Learning for Image Recognition (CVPR 2016) [pdf](https://arxiv.org/abs/1512.03385)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/resnet.png)

* ##### Code
```python
import torch
from cnns.resnet import resnet18 
x = torch.randn(2, 3, 224, 224)
model = resnet18()
y = model(x)
print(y.shape)
```
### 3. WideResNet Model
* #### Wide Residual Networks (BMVC 2016) [pdf](https://arxiv.org/pdf/1605.07146)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/wideresnet.png)

* ##### Code
```python
import torch
from cnns.wideresnet import wideresnet
x = torch.randn(2, 3, 224, 224)
model = wideresnet()
y = model(x)
print(y.shape)
```
### 4. DenseNet Model
* #### Densely Connected Convolutional Networks (CVPR 2017) [pdf](http://arxiv.org/abs/1608.06993v5)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/densenet.png)

* ##### Code
```python
import torch
from cnns.densenet import densenet121
x = torch.randn(2, 3, 224, 224)
model = densenet121()
y = model(x)
print(y.shape)
```
### 5. PyramidNet Model
* #### Deep Pyramidal Residual Networks (CVPR 2017) [pdf](https://arxiv.org/pdf/1610.02915)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/yramidnet.png)

* ##### Code
```python
import torch
from cnns.pyramidnet import pyramidnet18
x = torch.randn(2, 3, 224, 224)
model = densenet121()
y = model(x)
print(y.shape)
```
### 6. MobileNetV1 Model
* #### MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (CVPR 2017) [pdf](https://arxiv.org/pdf/1704.04861.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/mobilenetv1.png)

* ##### Code
```python
import torch
from cnns.mobilenetv1 import MobileNetv1
x = torch.randn(2, 3, 224, 224)
model = MobileNetv1()
y = model(x)
print(y.shape)
```
### 7. MobileNetV2 Model
* #### MobileNetV2: Inverted Residuals and Linear Bottlenecks (CVPR 2018) [pdf](https://arxiv.org/abs/1801.04381)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/mobilenetv2.png)

* ##### Code
```python
import torch
from cnns.mobilenetv2 import MobileNetv2
x = torch.randn(2, 3, 224, 224)
model = MobileNetv2()
y = model(x)
print(y.shape)
```
### 8. MobileNetV3 Model
* #### Searching for MobileNetV3 (ICCV 2019) [pdf](https://arxiv.org/pdf/1905.02244)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/mobilenetv3.png)

* ##### Code
```python
import torch
from cnns.mobilenetv3 import mobilenetv3_small
x = torch.randn(2, 3, 224, 224)
model = mobilenetv3_small()
y = model(x)
print(y.shape)
```
### 9. MnasNet Model
* #### MnasNet: Platform-Aware Neural Architecture Search for Mobile (CVPR 2019) [pdf](http://arxiv.org/pdf/1807.11626)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/mnasnet.png)

* ##### Code
```python
import torch
from cnns.mnasnet import MnasNet
x = torch.randn(2, 3, 224, 224)
model = MnasNet()
y = model(x)
print(y.shape)
```
### 10. EfficientNetV1 Model
* #### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (ICML 2019) [pdf](https://arxiv.org/abs/1905.11946)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/efnetv1.png)

* ##### Code
```python
import torch
from cnns.efficientnet import EfficientNet
x = torch.randn(2, 3, 224, 224)
model = EfficientNet()
y = model(x)
print(y.shape)
```
### 11. Res2Net Model
* #### Res2Net: A New Multi-scale Backbone Architecture (TPAMI 2019) [pdf](https://arxiv.org/pdf/1904.01169)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/res2net.png)

* ##### Code
```python
import torch
from cnns.res2net import res2net50
x = torch.randn(2, 3, 224, 224)
model = res2net50()
y = model(x)
print(y.shape)
```
### 12. MobileNeXt Model
* #### Rethinking Bottleneck Structure for Efficient Mobile Network Design (ECCV 2020) [pdf](https://arxiv.org/pdf/2007.02269.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/mobilenext.png)

* ##### Code
```python
import torch
from cnns.mobilenext import MobileNeXt
x = torch.randn(2, 3, 224, 224)
model = MobileNeXt()
y = model(x)
print(y.shape)
```
### 13. GhostNet Model
* #### GhostNet: More Features from Cheap Operations (CVPR 2020) [pdf](https://arxiv.org/abs/1911.11907)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/ghost.png)

* ##### Code
```python
import torch
from cnns.ghostnet import ghostnet
x = torch.randn(2, 3, 224, 224)
model = ghostnet()
y = model(x)
print(y.shape)
```
### 14. EfficientNetV2 Model
* #### EfficientNetV2: Smaller Models and Faster Trainin (ICML 2021) [pdf](https://arxiv.org/abs/2104.00298)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/efnetv2.png)

* ##### Code
```python
import torch
from cnns.efficientnet import EfficientNetV2
x = torch.randn(2, 3, 224, 224)
model = EfficientNetV2()
y = model(x)
print(y.shape)
```
### 15. ConvNeXt Model
* #### A ConvNet for the 2020s (CVPR 2022) [pdf](https://arxiv.org/abs/2201.03545)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/convnext.png)

* ##### Code
```python
import torch
from cnns.convnext import convnext_18
x = torch.randn(2, 3, 224, 224)
model = convnext_18()
y = model(x)
print(y.shape)
```
### 16. Unet Model
* #### U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015) [pdf](https://arxiv.org/pdf/1505.04597.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/unet.png)

* ##### Code
```python
import torch
from cnns.unet import Unet
x = torch.randn(2, 3, 512, 512)
model = Unet(10)
y = model(x)
print(y.shape)
```
### 17. ESPNet Model
* #### ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation (ECCV 2018) [pdf]( https://arxiv.org/abs/1803.06815)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/espnet.png)

* ##### Code
```python
import torch
from cnns.espnet import ESPNet
x = torch.randn(2, 3, 512, 512)
model = ESPNet(10)
y = model(x)
print(y.shape)
```
## MLP-Like Models
### 1. MLP-Mixer Model
* #### MLP-Mixer: An all-MLP Architecture for Vision (NeurIPS 2021) [pdf](https://arxiv.org/pdf/2105.01601.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/mlpmixer.png)

* ##### Code
```python
import torch
from mlps.mlp_mixer import MLP_Mixer
x = torch.randn(2, 3, 224, 224)
model = MLP_Mixer()
y = model(x)
print(y.shape)
```
### 2. gMLP Model
* #### Pay Attention to MLPs (NeurIPS 2021) [pdf]( https://arxiv.org/pdf/2105.08050)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/gmlp.png)

* ##### Code
```python
import torch
from mlps.gmlp import gMLP
x = torch.randn(2, 3, 224, 224)
model = gMLP()
y = model(x)
print(y.shape)
```
### 3. GFNet Model
* #### Global Filter Networks for Image Classification (NeurIPS 2021) [pdf](https://arxiv.org/abs/2107.00645)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/gfnet.png)

* ##### Code
```python
import torch
from mlps.gfnet import GFNet
x = torch.randn(2, 3, 224, 224)
model = GFNet()
y = model(x)
print(y.shape)
```
### 4. sMLP Model
* #### Sparse MLP for Image Recognition: Is Self-Attention Really Necessary? (AAAI 2022) [pdf](https://arxiv.org/abs/2109.05422)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/smlp.png)

* ##### Code
```python
import torch
from mlps.smlp import sMLPNet
x = torch.randn(2, 3, 224, 224)
model = sMLPNet()
y = model(x)
print(y.shape)
```
### 5. DynaMixer Model
* #### DynaMixer: A Vision MLP Architecture with Dynamic Mixing (ICML 2022) [pdf](https://arxiv.org/pdf/2201.12083)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/dynamixer.png)

* ##### Code
```python
import torch
from mlps.dynamixer import DynaMixer
x = torch.randn(2, 3, 224, 224)
model = DynaMixer()
y = model(x)
print(y.shape)
```
### 6. ConvMixer Model
* #### Patches Are All You Need? (TMLR 2022) [pdf](https://arxiv.org/pdf/2201.09792)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/convmixer.png)

* ##### Code
```python
import torch
from mlps.convmixer import ConvMixer
x = torch.randn(2, 3, 224, 224)
model = ConvMixer(128, 6)
y = model(x)
print(y.shape)
```
### 7. ViP Model
* #### Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition (TPAMI 2022) [pdf](https://arxiv.org/abs/2106.12368)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/vip.png)

* ##### Code
```python
import torch
from mlps.vip import vip_s7
x = torch.randn(2, 3, 224, 224)
model = vip_s7()
y = model(x)
print(y.shape)
```
### 8. CycleMLP Model
* #### CycleMLP: A MLP-like Architecture for Dense Prediction (ICLR 2022) [pdf](https://arxiv.org/abs/2107.10224)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/cyclemlp.png)

* ##### Code
```python
import torch
from mlps.cyclemlp import CycleMLP_B1
x = torch.randn(2, 3, 224, 224)
model = CycleMLP_B1()
y = model(x)
print(y.shape)
```
### 9. Sequencer Model
* #### Sequencer: Deep LSTM for Image Classification (NeurIPS 2022) [pdf](https://arxiv.org/abs/2205.01972)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/sequencer.png)

* ##### Code
```python
import torch
from mlps.sequencer import sequencer_s
x = torch.randn(2, 3, 224, 224)
model = sequencer_s()
y = model(x)
print(y.shape)
```
### 10. MobileViG Model
* #### MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications (CVPRW 2023) [pdf](https://arxiv.org/pdf/2307.00395.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/mobilevig.png)

* ##### Code
```python
import torch
from mlps.mobilevig import mobilevig_s
x = torch.randn(2, 3, 224, 224)
model = mobilevig_s()
y = model(x)
print(y.shape)
```
