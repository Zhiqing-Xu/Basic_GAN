
# <center> **-- Deep Convolutional Generative Adversarial Network --** </center>


# <font size="6"> &#10148; </font> Introduction

- In this work, a deep convolutional generative adversarial network is trained. The model architecture is an expansion on the original DCGAN architecture in [<em>`Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks`</em>](https://arxiv.org/abs/1511.06434). With the enhancement in model architecture, the model is capable of generating images of 128 * 128 resolution (originally 64 * 64). We train the DCGAN on a fraction of the [<em> `Large-scale CelebFaces Attributes (CelebA) Dataset` </em>](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), a image dataset contains 202,599 number of face images, with binary attributes (labels are **NOT** used in this work).
- Images from the dataset are 178 * 218 , and is adjusted to 128 * 128 (via CenterCrop function from cv2). 
- The model is trained on a single RTX 3090 GPU for 120 epoches. It takes around 15 hours to finish the training. 
- Due to very limited time we have to finish this project, we didn't perform much hyperparameter tuning. We use the learning rate from reproduced original DCGAN found online.
- Images generated showed acceptable performance of the model.



# <font size="6"> &#10148; </font> Original DCGAN performance (64 x 64 Image Generator)

<div align="center">
    <video src="https://user-images.githubusercontent.com/47986787/227747455-554964bc-b5c2-4182-9693-9b0f4eb0ec10.mov" controls="controls" style="max-width: 900px;"> </video>
</div>


# <font size="6"> &#10148; </font> Enhanced DCGAN performance (128 x 128 Image Generator)

- Enhanced DCGAN's performance is NOT very stable since there is very limited time for tuning the hyperparameters of the model, including, 
    - size of feature map in discriminator
    - size of feature map in generator
    - size of latent space
    - learning rate (super sensitive)

- 



# <font size="6"> &#10148; </font> DCGAN & SRGAN
- Run_Image_Generator_Main.ipynb provides a pipeline to generate face images.
    - A trained DCGAN (that is trained using Train_Enhanced_DCGAN.ipynb) is loaded to generate low-resolution images.
    - A SRGAN model trained by others is used to scale-up the resolution of the generate images.











