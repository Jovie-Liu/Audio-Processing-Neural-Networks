# Audio Processing Neural Networks

Here we apply the time-frequency techniques in audio processing to design a specialized neural network for audio. The first type of neural network is based on Fourier transform and involves complex number operations. The complex derivatives are computed by [Wirtinger calculus](https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc). To aviod the complexity of complex numbers and restrict the computations in the real domain, we further introduce the discrete cosine transform architecture into the neural network design.

<img src="graphs/DFT_DCT.jpg" style="width:900px">
<caption><center> Figure 1. Basic Structures of DFT and DCT neural networks </center></caption>
