# DCGAN

- Run_Image_Generator_Main.ipynb provides a pipeline to generate face images.
    - A trained DCGAN (that is trained using Train_Enhanced_DCGAN.ipynb) is loaded to generate low-resolution images.
    - A SRGAN model trained by others is used to scale-up the resolution of the generate images.
    

<video src="Sample_GAN_Training.mov" controls="controls" style="max-width: 730px;"> </video>