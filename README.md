# Video Style Transfer


<!-- Content             |  Style | Transfer
:-------------------------:|:-------------------------:|:---:
<img src="./videos/video.gif" width="250" height="140"/>  |  <img src="/styles/tar49.png" width="250" height="140"/> | <img src="./videos/output2.gif" width="250" height="140"/> -->
 
Video Style Transfer is a CSCI 590 Directed Research project supervised by Prof. [Scott Easley](https://viterbi.usc.edu/directory/faculty/Easley/Scott). This project aims to build a pipeline with new cost-effective and professional-looking cinematic imagery that transfers styles of the given image into a video.

### Motivation

Professional cinematic lighting costs thousands of dollars, as does color correction after a film is developed. This research is a way to see if a lot of that can be circumvented by simply choosing an existing image as an input and having the algorithm do all the relighting.

[Neural Style Transfer](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), regardless of how fancy it is, depreciates the image into a garbled series of lines and colors. So while it is initially interesting, it has very limited applications in a film. [Photo Style Transfer](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luan_Deep_Photo_Style_CVPR_2017_paper.pdf) can be used to color-correct images that are shot well but are not lit well or have uncorrected color. However, due to computing Matting Laplacian matrices for retaining the photorealistic semantic, Photo Style Transfer runs painfully slow and is not able to process a video with hundreds and thousands of image frames. 

### Method



## How to use
    conda create --name IST -y python=3.8
    conda activate IST
    pip install -r requirements.txt

    python transfer.py --content-image ./dataset/input/1.png --style-image ./dataset/style/1.png




### Contact
If you are interested in this project or have questions about it, feel free to contact Rong Liu (<rliu8691@usc.edu>) and Scott Easley (<seasley@usc.edu>).