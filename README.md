# DeepSino

![ Architecture ]( ./Sample%20Images/Fig-1.png )

This is the code associated with our paper:

Navchetan Awasthi\*, Gaurav Jain\*, Sandeep Kumar Kalva, Manojit Pramanik, Phaneendra K. Yalavarthy, [Deep Neural-Network Based Sinogram Super-resolution and Bandwidth Enhancement for Limited Data Photoacoustic Tomography](https://doi.org/10.1109/TUFFC.2020.2977210), Published in _IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control (Special issue on Deep Learning in Medical Ultrasound)_, 2020 (in press). \* Co-first authors wth equal contribution

When using this code for your own projects, please cite this article.

## Project Description

Photoacoustic tomography (PAT) is a nonin2 vasive imaging modality combining the benefits of optical contrast at ultrasonic resolution. Analytical reconstruction algorithms for photoacoustic (PA) signals require a large number of data points for accurate image reconstruction. However, in practical scenarios, data are collected using the limited number of transducers along with data being often corrupted with noise resulting in only qualitative images. Furthermore, the collected boundary data are band10 limited due to limited bandwidth (BW) of the transducer, making the PA imaging with limited data being qualitative. In this work, a deep neural network-based model with loss function being scaled root-mean-squared error was proposed for super-resolution, denoising, as well as BW enhancement of the PA signals collected at the boundary of the domain. The proposed network has been compared with traditional as well as other popular deep-learning methods in numerical as well as experimental cases and is shown to improve the collected boundary data, in turn, providing superior quality reconstructed PA image. The improvement obtained in the Pearson correlation, structural similarity index metric, and root-mean-square error was as high as 35.62%, 33.81%, and 41.07%, respectively, for phantom cases and signal-to-noise ratio improvement in the reconstructed PA images was as high as 11.65 dB for in vivo cases compared with reconstructed image obtained using original limited BW data. Data available [here](https://sites.google.com/site/sercmig/home/dnnpat).


