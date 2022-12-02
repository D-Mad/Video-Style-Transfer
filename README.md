# Video Style Transfer
 
Video Style Transfer is a CSCI 590 Directed Research project supervised by Prof. Scott Easley. The goal of this project is to build up a pipeline with new cost-effective and professional-looking cinematic imagery that transfers styles of the given image into a video.


Professional cinematic lighting costs thousands of dollars, as does color correction after a film is developed. This research is a way to see if a lot of that can be circumvented by simply choosing an existing image as an input and having the algorithm do all the relighting.

Neural Style Transfer, regardless of how fancy it is, depreciates the image into a garbled series of lines and colors. So while it is initially interesting, it has very limited applications in a film. Photo Style Transfer can be used to color-correct images that are shot well, but are not lit well or have uncorrected color. However, due to Laplacian computation for retaining the photorealistic semantic, Photo Style Transfer runs painfully slow and is not able to process a video with hundreds and thousands of image frames. 

This project adopts HR-Net to keep the semantic information and uses the same way from Neural Style Transfer to abstract image styles by . Similar to Neural Style Transfer, there is a content-style trade-off we need to fine-tune carefully. Hence we first train the HR-Net based on a single video frame to verify whether the transfer result satisfies one's expectation. Once getting a decent transferred result, we can save the model and apply it to all the rest video frames with similar semantic information.

Notice that Neural Style Transfer and Photo Style Transfer are model-free methods. They start with a noisy image or the content image and gradually transfer the style on it instead of learning semantic and style information. So every time facing a new transfer task, they have to go through the whole process, which means that the computational time linearly goes up as the number of video frames increases. However, when training an HRNet model, which is a neural network model, it learns semantic and style patterns from the content and style images. The whole inference time reduces effectively. You only need to fine-tune the content-style trade-off on a pair of content and style images and the rest of the time inferencing other frames won't annoy you.

## How to use




