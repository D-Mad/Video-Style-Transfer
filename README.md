<h1 align="center"> Instant Photorealistic Style Transfer:<br /> A Lightweight and Adaptive Approach </h1>
<p align="center">
<a style="color:#000000;" href="https://rongliu-leo.github.io/">Rong Liu</a>&nbsp;&nbsp;&nbsp;&nbsp;
<a style="color:#000000;" href="https://www.linkedin.com/in/enyu-zhao-564566250/">Enyu Zhao</a>&nbsp;&nbsp;&nbsp;&nbsp;
<a style="color:#000000;" href="https://www.linkedin.com/in/liuzy98/">Zhiyuan Liu</a>&nbsp;&nbsp;&nbsp;&nbsp;
<a style="color:#000000;" href="https://viterbi.usc.edu/directory/faculty/Easley/Scott">Scott John Easley</a>
</p>

<p align="center">
<a href="https://www.usc.edu/">University of Southern California</a> 
</p>

<p align="center">
<a href="https://rongliu-leo.github.io/Video-Style-Transfer/">Project Page</a>
<a>  |  </a>
<a href="https://www.usc.edu/">Paper</a> 
</p>

 


# About
Instant Photorealistic Style Transfer (IPST) is designed to achieve real-time photorealistic style transfer on 4K-resolution images and videos without the need for pre-training using pair-wise datasets. It utilizes a lightweight style network to enable instant photorealistic style transfer from a style image to a content image or video while preserving spatial information. It also has an instance-adaptive optimization method to accelerate convergence, resulting in rapid training completion within seconds.




# Quickstart
The quickstart will help you install IPST and be familiar with the command.

## Installation

### Prerequisites

An NVIDIA video card with installed [CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

### Clone repository
```bash
git clone https://github.com/RongLiu-Leo/Video-Style-Transfer.git
cd Video-Style-Transfer
```

### Create environment

```bash
conda create --name IPST -y python=3.8
conda activate IPST
pip install --upgrade pip
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Image style transfer
For a content-style image pair, run
```bash
python transfer.py --content-image {content_image_path} --style-image {style_image_path}
```
For multiple content-style image pairs, please put them into two folders and run
```bash
python transfer.py --content-image-folder {content_image_folder_path} --style-image-folder {style_image_folder_path}
```
### Video style transfer
```bash
python transfer.py --content-video {content_image_path} --style-image {style_image_path}
```
By default, IPST will load and transfer all video frames, which performs extreme speed. However, if the number of frames is too large, it may lead to memory issues.

Try to use ```--frame-by-frame``` to solve memory issues:
```bash
python transfer.py --content-video {content_video_path} --style-image {style_image_path} --frame-by-frame
```
If it is still not working, try to split the whole video into smaller videos.

# Citation

# Contact
If you are interested in this project or have questions about it, feel free to contact Rong Liu (<rliu8691@usc.edu>) and Scott Easley (<seasley@usc.edu>).