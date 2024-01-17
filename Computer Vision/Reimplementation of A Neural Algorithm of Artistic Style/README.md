Reimplementation of A Neural Algorithm of Artistic Style
=========================================================================

Reimplement: A Neural Algorithm of Artistic Style

For this project, I used a PyTorch's pretrained VGG19 feature space. I changed all the maxpooling layers to avgpool, as described in the paper.

Here is the total feature space:

![](model.png)

For content activation, I used the second convolutional layer in the 4th block. For style activations, I used the first convolutional  layer in each of the 5 blocks. I initally used optim.lbfgs as our omtimizer but found better results once switching to optim.Adam. I also used a learning rate of 0.01 and found the best results between 1500 and 3000 iterations. A difference between my model and the proposed steps in the paper, is that I did not use a  hite noise image as my input. Instead, I found better results using the content image as my input into the model.

Here are the results of applying 3 different artistic styles to the content of Neckarfront in Tu ̈bingen, German

Content Image:

![](content_images/neckarfront.jpg)

Style Images:

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="center"><p><img src="style_images/starry_night.jpg" width = "300" height = "300"/></p>
<p>The Starry Night by Vincent can Gogh</p></td>
<td align="center"><p><img src="style_images/shipweck.jpg" width = "300" height = "300" /></p>
<p>The Shipwreck of the Minotaur by J.M.W. Turner</p></td>
<td align="center"><p><img src="style_images/the_scream.jpg" width = "300" height = "300" /></p>
<p>Der Scheri by Edvard Munch</p></td>
</tr>
</tbody>
</table>

Mixed content and style:

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="center"><p><img src="generated/necktarfront_x_starry_night.png" width = "300" height = "300"/></p>
<p>Starry Night x Neckarfront</p></td>
<td align="center"><p><img src="generated/necktarfront_x_shipwreck.png" width = "300" height = "300"/></p>
<p>Shipwreck x Neckarfront</p></td>
<td align="center"><p><img src="generated/nacktarfront_x_scream.png" width = "300" height = "300"/></p>
<p>Dan Sheri x Neckarfront</p></td>
</tr>
</tbody>
</table>

Compared to the images in the paper, I don't think I captured the style as strongly as I would have preferred, even when using an alpha/beta ratio \> 10\^6. 

Here are the results of different styles to my own images

Content Images:

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="center"><p><img src="content_images/soli.jpg" width = "300" height = "300" /></p>
<p>Morgan's Dog</p></td>
<td align="center"><p><img src="content_images/lastupper.jpg" width = "300" height = "300"/></p>
<p>The Last Supper by Leonardo</p></td>
<td align="center"><p><img src="content_images/choatic.jpg" width = "300" height = "300" /></p>
<p>Chaotic Space by Anastasiya</p></td>
</tr>
</tbody>
</table>

Style Images:

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="center"><p><img src="style_images/memory.jpg" width = "300" height = "300"/></p>
<p>Persistence of Memory by Salvidor Dali</p></td>
<td align="center"><p><img src="style_images/picasso.jpg" width = "300" height = "300"/></p>
<p>Three Musicians by Picasso</p></td>
<td align="center"><p><img src="content_images/soli.jpg"width = "300" height = "300" /></p>
<p>Morgan's Dog</p></td>
</tr>
</tbody>
</table>

Mixed Content and Style Images: Style Images

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="center"><p><img src="generated/sol_x_memory.png" width = "300" height = "300"/></p>
Dali x Sol</p></td>
<td align="center"><p><img src="generated/picasso_x_supper.png" width = "300" height = "300" /></p>
<p>Picasso x Last Supper</p></td>
<td align="center"><p><img src="generated/soli_x_chaos.png" width = "300" height = "300" /></p>
<p>Sol x Chaos</p></td>
</tr>
</tbody>
</table>


As you can see, the last image does not quite work. I believe this is caused by having too much chaos imbedded in the image, and my model cannot pick out and discern these nuances well. In addition to this. The "style" image has no real "style" to it, making it harder to apply to another image
