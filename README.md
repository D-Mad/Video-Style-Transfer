# Video Style Transfer
 
Video Style Transfer is a CSCI 590 Directed Research project supervised by Prof. [Scott Easley](https://viterbi.usc.edu/directory/faculty/Easley/Scott). The goal of this project is to build up a pipeline with new cost-effective and professional-looking cinematic imagery that transfers styles of the given image into a video.


Professional cinematic lighting costs thousands of dollars, as does color correction after a film is developed. This research is a way to see if a lot of that can be circumvented by simply choosing an existing image as an input and having the algorithm do all the relighting.

[Neural Style Transfer](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), regardless of how fancy it is, depreciates the image into a garbled series of lines and colors. So while it is initially interesting, it has very limited applications in a film. [Photo Style Transfer](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luan_Deep_Photo_Style_CVPR_2017_paper.pdf) can be used to color-correct images that are shot well, but are not lit well or have uncorrected color. However, due to computing Matting Laplacian matrices for retaining the photorealistic semantic, Photo Style Transfer runs painfully slow and is not able to process a video with hundreds and thousands of image frames. 

This project adopts [HR-Net](https://ieeexplore.ieee.org/abstract/document/9052469?casa_token=y1aLdMGcewkAAAAA:UH78gcYDmeq6umHIbLCK9-py4U4cFYzRAgWOG9ltR7ozb4X7_q-5DPMM9wRXJCWhE3VoxjyqVw) to keep the semantic information and uses the same way from Neural Style Transfer to abstract image styles by minimizing the Mean Squared Error between content and style activation maps. Similar to Neural Style Transfer, there is a content-style trade-off we need to fine-tune carefully. Hence we first train the HR-Net based on a single video frame to verify whether the transfer result satisfies one's expectation. Once getting a decent transferred result, we can save the model and apply it to all the rest video frames with similar semantic information.

Notice that Neural Style Transfer and Photo Style Transfer are model-free methods. They start with a noisy image or the content image and gradually transfer the style on it instead of learning semantic and style information. So every time facing a new transfer task, they have to go through the whole process, which means that the computational time linearly goes up as the number of video frames increases. However, when training an HRNet model, which is a neural network model, it learns semantic and style patterns from the content and style images. The whole inference time reduces effectively. You only need to fine-tune the content-style trade-off on a pair of content and style images and the rest of the time inferencing other frames won't annoy you.

## How to use

The demo video:
![](video.gif)


1. use ffmpeg command to split the video into [frames](/frames/).

```sh
ffmpeg -i video.mp4 %d.jpg
```
2. open the interactive [Jupyter notebook](/frame_transfer.ipynb) to fine-tune the content-style trade-off and save the satisfied model. The instruction of how to use this notebook is inside.

**Note:** You can also use this notebook to address photorealistic transfer tasks. Some examples are shown below.

![](imgs/3.png)
![](imgs/16.png)
![](imgs/24.png)

3.  apply the saved model to the rest of the frames.

```py
python 
```

4.  use ffmpeg command to integrate frames into the video.
```sh
ffmpeg -f image2 -i %d.jpg -vcodec libx264 -pix_fmt yuv420p output.mp4
``` 
