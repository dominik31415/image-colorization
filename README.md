## Teaser

I trained a CNN to reconstruct the color of provided monochrome images. Here are some pretty results, left the original grey image, right my network's colorization. The samples subfolder contains many more examples (including failed ones). 

<img src="/samples/single_images/grey_2_8.png" alt="drawing" width="200"/> | <img src="/samples/single_images/adverserial_2_8.png" alt="drawing" width="200"/> | <img src="/samples/single_images/grey_4_4.png" alt="drawing" width="200"/> | <img src="/samples/single_images/adverserial_4_4.png" alt="drawing" width="200"/> 
<img src="/samples/single_images/grey_5.png" alt="drawing" width="200"/> | <img src="/samples/single_images/adverserial_5.png" alt="drawing" width="200"/> | <img src="/samples/single_images/grey_5_1.png" alt="drawing" width="200"/> | <img src="/samples/single_images/adverserial_5_1.png" alt="drawing" width="200"/>  
<img src="/samples/single_images/grey_5_5.png" alt="drawing" width="200"/> | <img src="/samples/single_images/adverserial_5_5.png" alt="drawing" width="200"/> | <img src="/samples/single_images/grey_5_3.png" alt="drawing" width="200"/> | <img src="/samples/single_images/adverserial_5_3.png" alt="drawing" width="200"/> 



## Motivation

This is my first attempt to use a CNN for unassisted image colorization: Given only the grey channel of an image, the network is supposed to add the original (or any realistic) color channels. In principal, this is a great problem for machine learning: take any of the bazillion image data sets, split off the grey channel, and warm up your GPU...

The big drawback is though, that this problem is highly multi-modal, i.e. for most objects there are several possible colorizations, a flower can be any hue on the rainbow and even simple "stuff" like water can be anything from blue to green to brown to the reflection of the before mentioned flower. In those cases the “naive” approach using supervised learning with a loss function will often just decide on an average color and result in pale or outright grey images. 

A possible remedy are GANs, which are unfortunately hard to train. Straight out of the box I only could get my networks to converge for rather small images (like 128x128 or smaller). As a work-around I decided to employ a pre-trained generator, trained using a classical loss function, and then iteratively fine-tune it using my GAN critic.


## Results

My network ended up producing many [“soso” images](https://github.com/dominik31415/image-colorization/blob/master/samples/soso.png) (about 70%), a few [ugly images](https://github.com/dominik31415/image-colorization/blob/master/samples/ugly.png) (about 10%) and quite some [awesome images](https://github.com/dominik31415/image-colorization/blob/master/samples/good.png) (about 20%), as shown above. The critic was set to ignore the outmost 16px of each generated image, thus they usually are a bit off in color there.

The most common problem (the “soso” category) are incomplete re-colorizations, i.e. the network had the right idea but did not fill in the full structure, or it simply overlooked an object completely and instead merged it with the background. Longer training might have helped here. But training took almost a week already on my GPU and I had to stop it before results actually plateaued. All in all, I only pre-trained for less than 15 epochs, and used the GAN training for roughly 8 additional epochs.

Further, my network failed to treat more complex structures, like cluttered images and animals (most of those images go into the bad category).  Those instances typically exhibit big red blotches completely ignoring object boundaries.  To be fair, there are relatively few examples for these in my training data set, i.e. I think a more comprehensive set would be necessary here. Lastly a larger network architecture would have certainly improved results. For images with poor results, the generator often failed on a lower level to segment the image in the pre-training step, even though this is a task that had been demonstrated to be learnable with a classical loss function; from reference [2] we know that a well designed loss could take you quite far already.


## Brief description of method

(a) Work is done in La*b* color space. Given the L channel (closely resembling the grey channel) the neural net is tasked with guessing the missing color channels a/b

(b) The problem is inherently multi-modal. Reference [1] used a classical GAN, but I found WGAN-GP to exhibit superior convergence behavior

(c) But for 256x256px sized images this seems to overwhelm my networks, so I moved on to a pre-trained generator

(d) The generator is pre-trained as described in [2], i.e. using discrete a/b channels, a cross-entropy loss and "magic weights" to account for class inbalances

(e) The training had to be done "adiabatically", i.e. by starting with an almost completely frozen generator and then slowly un-freezing layers and fine-tuning it for a few epochs. 

(e) I used the places365 dataset (2016 edition, 256x256 pixels). Eventually I restricted myself to the wild_field subcategory, comprised of 54k and 27k images. I also used the typical data augmentation methods (flipping and rotating). 


## References
[1] Image Colorization with Generative Adversarial Networks, *Kamyar Nazeri, Eric Ng, Mehran Ebrahimi*, 2018

[2] Colorful Image Colorization, *Richard Zhang, Phillip Isola, Alexei A. Efros*, 2016


## Files

| File name        | Description| 
| ------------- |:-------------:| 
| model_utils.py | main file defining the models used |
| data_utils.py  | file defining+preprocessing the data source (using multiprocessing) and "magic weights" |
| definitions_XYZ.py | files to be imported in the various training steps |
| calc_magic_weights.jpynb | script to generate magic weights (step 0) |
| direct_train.jpynb | step 1 of training, uses cross-entropy to send the generator onto the right track |
| main_train.jpynb | step 2 of training, uses WGAN-GP to refine the generator |
| adverserial_ABC.png | final image generated after GAN training |
| grey_ABC.png | grey image input for generator |
| raw_ABC.png | initial color image (only included for reference purposes) |
