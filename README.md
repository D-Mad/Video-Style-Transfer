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
The quickstart will help you get started with the default vanilla NeRF trained on the classic Blender Lego scene. For more complex changes


## How to use
    conda create --name IST -y python=3.8
    conda activate IST
    pip install -r requirements.txt

    python transfer.py --content-image ./dataset/input/1.png --style-image ./dataset/style/1.png




### Contact
If you are interested in this project or have questions about it, feel free to contact Rong Liu (<rliu8691@usc.edu>) and Scott Easley (<seasley@usc.edu>).