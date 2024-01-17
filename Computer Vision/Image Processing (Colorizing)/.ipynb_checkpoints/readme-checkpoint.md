Project 1: Colorizing the Prokudin-Gorskii Photo Collection
===========================================================

The goal of the project is to take the digitized Prokudin-Gorskii glass
plate images and apply image processing techniques to automatically
produce a color image. The basic method is to extract the three color
channel images, place them on top of each other, and align them so that
they form a single RGB color image.

Data
----

I used 17 digitized Prokudin-Gorskii glass plate images in this program.
The sizes ranged from smaller .jpg's to larger .tif files. Here is an
example of the raw data used:


![](Data/cathedral.jpg)

After seperating the image into its respective R, B, G channel images,
without using any metrics when you simply align the images you get the
following results. 

![](outputs/original/og_cathedral.jpg)
![](outputs/original/og_monastery.jpg)

Image Matching Metrics
----------------------

When naively aligned, the images produced are not sharp. In order to
better align the three channel images, we have to determine the optimal
displacement for each image relative to one another. To do this, I
exhaustively search over a [-15, 15] window of possible displacements. I
applied the following two metrics to each potential alignment and keep
the displacement that produced the best score.

#### Sum of Squared Differences (SSD)

[![](https://latex.codecogs.com/gif.latex?\sum&space;(image_{1}&space;-&space;image_{2})^{2} "\sum (image_{1} - image_{2})^{2}")](https://www.codecogs.com/eqnedit.php?latex=\sum&space;(image_{1}&space;-&space;image_{2})^{2})

#### Normalized Cross-Correlation (NCC)

[![](https://latex.codecogs.com/gif.latex?\frac{image_{1}}{\left&space;|&space;image_{1}&space;\right&space;|}\cdot&space;\frac{image_{2}}{\left&space;|&space;image_{2}&space;\right&space;|} "\frac{image_{1}}{\left | image_{1} \right |}\cdot \frac{image_{2}}{\left | image_{2} \right |}")](https://www.codecogs.com/eqnedit.php?latex=\frac{image_{1}}{\left&space;|&space;image_{1}&space;\right&space;|}\cdot&space;\frac{image_{2}}{\left&space;|&space;image_{2}&space;\right&space;|})

 

![](outputs/ssd_cathedral.jpg) ![](outputs/ncc_cathedral.jpg)

Same image with SSD applie (left) versus NCC applied (right)

Ultimately, I decided to use NCC over SSD for the final metric when
establishing displacements

Image Pyramids
--------------

While using NCC alone worked for the smaller images, when applying the
same methodology to larger images, the exhaustive search became to
expensive to run efficiently. To remedey this, I implemented a
coarse-to-fine pyramid speed up. To do so, I applied the same exhaustive
search as I did to smaller images to a scaled version of the larger
images. Usually, the larger image was scaled to somewhere between 1/24
to 1/48 the orginal size. As the image gets scaled back up to its
orginal size, the optimal displacments are carried through and applied
to the final image.

 

Primitive Results
-----------------

Here are the results from only using NCC and image pyramids. There has
been no pre or post-op cropping or adustments.

![](outputs/pyramid_only/po_cathedral.jpg)

Displacement: R(7, -1), G(1, -1), B(0, 0)

![](outputs/pyramid_only/po_monastery.jpg)

Displacement: R(9, 1), G(-6, 0), B(0, 0)

![](outputs/pyramid_only/po_tobolsk.jpg)

Displacement: R(6, 3), G(3, 2), B(0, 0)

![](outputs/pyramid_only/po_castle.jpg)

Displacement: R(6, 3), G(3, 2), B(0, 0)

![](outputs/pyramid_only/po_icon.jpg)

Displacement: R(73, 16), G(31, 12), B(0, 0)

![](outputs/pyramid_only/po_lady.jpg)

Displacement: R(106, -13), G(43, -5), B(0, 0)

![](outputs/pyramid_only/po_self_portrait.jpg)

Displacement: R(113, -3), G(38, -2), B(0, 0)

![](outputs/pyramid_only/po_melons.jpg)

Displacement: R(172, 0), G(65, 3), B(0, 0)

![](outputs/pyramid_only/po_three_generations.jpg)

Displacement: R(91, 5), G(39, 4), B(0, 0)

![](outputs/pyramid_only/po_emir.jpg)

Displacement: R(91, 13), G(-2, 5), B(0, 0)

![](outputs/pyramid_only/po_onion_church.jpg)

Displacement: R(91, 2), G(40, 16), B(0, 0)

![](outputs/pyramid_only/po_train.jpg)

Displacement: R(91, -1), G(92, -5), B(0, 0)

![](outputs/pyramid_only/po_harvesters.jpg)

Displacement: R(104, 3), G(100, -3), B(0, 0)

![](outputs/pyramid_only/po_workshop.jpg)

Displacement: R(53, -12), G(40, -3), B(0, 0)

Results on photos I personally pulled from collection:

![](outputs/pyramid_only/po_church.jpg)

Displacement: R(8, 0), G(4, 0), B(0, 0)

![](outputs/pyramid_only/po_old_gates.jpg)

Displacement: R(10, -1), G(9, -1), B(0, 0)

![](outputs/pyramid_only/po_fisherman.jpg)

Displacement: R(10, 0), G(4, 1), B(0, 0)

While applying only these techniques works relatively well, it is
apparent that not all the images are crisply aligned. After playing
around, I found that the easiest way to get cleaner photos is by
cropping the images before undergoing the alignment methods.

Auto-Cropping (Bells & Whistles)
--------------------------------

I choose to apply auto-cropping to all the images before they underwent
other the rest of the processing methods because this produced the
cleanist images, for the most part. Here are the results from applying
autocropping

![](outputs/auto_crop/ac_cathedral.jpg)

Displacement: R(12, 3), G(5, 2), B(0, 0)

![](outputs/auto_crop/ac_monastery.jpg)

Displacement: R(3, 2), G(-3, 2), B(0, 0)

![](outputs/auto_crop/ac_tobolsk.jpg)

Displacement: R(6, 3), G(3, 2), B(0, 0)

![](outputs/auto_crop/ac_castle.jpg)

Displacement: R(79, 2), G(26, 2), B(0, 0)

![](outputs/auto_crop/ac_icon.jpg)

Displacement: R(73, 17), G(30, 13), B(0, 0)

![](outputs/auto_crop/ac_lady.jpg)

Displacement: R(92, 5), G(41, -2), B(0, 0)

![](outputs/auto_crop/ac_self_portrait.jpg)

Displacement: R(0, -11), G(61, 22), B(0, 0)

![](outputs/auto_crop/ac_melons.jpg)

Displacement: R(164, 5), G(65, 7), B(0, 0)

![](outputs/auto_crop/ac_three_generations.jpg)

Displacement: R(92, 9), G(41, 10), B(0, 0)

![](outputs/auto_crop/ac_emir.jpg)

Displacement: R(40, 33), G(37, 18), B(0, 0)

![](outputs/auto_crop/ac_onion_church.jpg)

Displacement: R(91, 27), G(39, 19), B(0, 0)

![](outputs/auto_crop/ac_train.jpg)

Displacement: R(70, 24), G(33, 4), B(0, 0)

![](outputs/auto_crop/ac_harvesters.jpg)

Displacement: R(9, 13), G(46, 12), B(0, 0)

![](outputs/auto_crop/ac_workshop.jpg)

Displacement: R(86, -9), G(40, 0), B(0, 0)

Results on photos I personally pulled from collection:

![](outputs/auto_crop/ac_church.jpg)

Displacement: R(8, 0), G(4, 0), B(0, 0)

![](outputs/auto_crop/ac_old_gates.jpg)

Displacement: R(9, 0), G(2, 2), B(0, 0)

![](outputs/auto_crop/ac_fisherman.jpg)

Displacement: R(10, 0), G(4, 2), B(0, 0)

Conclusion
----------

Even after intrudcucing auto-cropping, some of the larger images are
still not aligned properly. This could be due to several factors,
including contrast, larger borders, varied color balance, and varied
brightness bewteen channel.
