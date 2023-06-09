# nQuantGpp
<p align="justify">nQuantGpp includes top 10 color quantization algorithms for g++ producing high quality optimized images. I enhance each of the algorithms to support semi transparent images. 
nQuantGpp also provides a command line wrapper in case you want to use it from the command line.</p>

<p align="justify">nQuantGpp is ported from nQuantCpp which migrates to OpenCV to leverage the deep learning features of such popular library.<br />
PNG8 or PNG1 is used because it's the only widely supported format which can store partially transparent images.</p>
<p>For Windows users, it is assumed that you have downloaded and extracted OpenCV 4 to the public user folder %PUBLIC%.<br />
In addition, nQuantGpp depends on OpenCV library opencv_world4xx.dll, it also required opencv_videoio_ffmpeg4xx_64.dll to open gif files.</p>
<p align="justify">For Linux users, please refer to Dockerfile to install libopencv-dev and compile by CMake.</p>
<p align="justify">If you are using the command line. Assuming you are in the same directory as nQuantGpp.exe, you could enter: `nQuantGpp yourImage.jpg /m 16`. To avoid dot gain, `/d n` can set the dithering to false. However, false contours will be resulted for gradient color zones.<br />
nQuantGpp will quantize yourImage.jpg and create yourImage-PNNLABquant16.png in the same directory.</p>
Let's catch up with the races: Ready, Go!!!<p />
<p>Original photo of sailing<br /><img src="https://mcychan.github.io/PnnQuant.js/demo/img/sailing_2020.jpg" /></p>
<p>Reduced to 256 colors by NeuQuant Neural-Net Quantization Algorithm<br /><img src="https://github.com/mcychan/nQuantGpp/assets/26831069/e149efbc-0d82-4ee9-9cf8-f94168438079" /></p>
<p>Reduced to 256 colors by Fast pairwise nearest neighbor based algorithm<br /><img src="https://github.com/mcychan/nQuantGpp/assets/26831069/7eb32e78-b920-4868-be87-104c1220ce20" /></p>
<p>Reduced to 256 colors by Fast pairwise nearest neighbor based algorithm with CIELAB color space<br /><img src="https://github.com/mcychan/nQuantGpp/assets/26831069/d2786a4c-0b43-4837-8a90-99d2fa0f3945" /></p>
<p>Reduced to 256 colors by Xialoin Wu's fast optimal color Quantization Algorithm<br /><img src="https://github.com/mcychan/nQuantGpp/assets/26831069/7d8ed4cf-6dae-4bf6-8b7a-86558e19fc43" /></p>
<hr />
<p align="justify">Most color quantization algorithms are based on K-Means clustering, can you see the minor but significant color is loss for the NeuQuant Neural-Net Quantization Algorithm?<br />
However, keeping the minor but significant color which in terms giving rise to false contours because some dominant colors are not selected into palette.</p>
<p align="justify">To select the most fitted palette, firstly reducing the color depth to 16 bits, ARGB4444 suits best for semi-transparent images. This means the value of alpha less than 16 will be converted to transparent color. For images having alpha channel value either 255 or 0, ARGB1555 is preferred.</p>
<p align="justify">Fast pairwise nearest neighbor based algorithm is updated to use YUV channels rather than RGB channels. Then the resulted image become less visible artifacts.</p>
<p align="justify">Color quantization algorithms conduct palette selection, Spatial color quantization algorithm; Efficient, Edge-Aware, Combined Color Quantization and Dithering algorithm conduct dithering at the same time.<br />
The color dithering functions may not belong to such algorithms. However, dithering leads to regular artifacts are very common like hue shift and dot gain.</p>
<p align="justify">Color dithering using a generalized Hilbert ("gilbert") space-filling curve produces clustered approximations to images with less regular artifacts which is O(n) instead of O(n<sup>2</sup>) of the classical dithering algorithm. Most importantly, "gilbert" required to set the maximum error acceptance level to avoid undesired artifacts. To deal with the false contours, partial Blue noise distribution is used.</p>
<p align="justify">On top of color dithering, Blue noise dithered sampling use high-quality stratified sampling patterns, which minimizes the low-frequency content in the output noise. It correlates the pixel estimates in a way that fits the implicative patterns of the original image. Unfortunately, using Blue noise distribution entirely would reduce the effect of color dithering and cannot resolve the color banding in gradient problem during CQ.</p>
<p align="justify">There are many magic numbers in Fast pairwise nearest neighbor based algorithm with CIELAB color space. Due to different framework returns different pixel values, these magic numbers are fine tuned by evaluating the highest PSNR and SSIM on testing images. However, this will arouse the overfitting problem.</p>
<p align="justify">For the future development, it is hoped that the models can be saved to files and load them up again to make better CQ.</p>
