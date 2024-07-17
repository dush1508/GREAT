from PIL import Image, ImageEnhance
import numpy as np
import math
from typing import List
from matplotlib import pyplot as plt

# Function definitions (provided in your initial script)

# Ensure all the functions (cartesian_to_polar, polar_to_cartesian, etc.)
# and add_chromatic, add_jitter, blend_images are here.

def add_chromatic(im, strength: float = 1, no_blur: bool = False):
    """Splits <im> into red, green, and blue channels, then performs a
    1D Vertical Gaussian blur through a polar representation. Finally,
    it expands the green and blue channels slightly.
    <strength> determines the amount of expansion and blurring.
    <no_blur> disables the radial blur
    """
    r, g, b = im.split()
    rdata = np.asarray(r)
    gdata = np.asarray(g)
    bdata = np.asarray(b)
    if no_blur:
        # channels remain unchanged
        rfinal = r
        gfinal = g
        bfinal = b
    else:
        rpolar = cartesian_to_polar(rdata)
        gpolar = cartesian_to_polar(gdata)
        bpolar = cartesian_to_polar(bdata)

        bluramount = (im.size[0] + im.size[1] - 2) / 100 * strength
        if round(bluramount) > 0:
            rpolar = vertical_gaussian(rpolar, round(bluramount))
            gpolar = vertical_gaussian(gpolar, round(bluramount * 1.2))
            bpolar = vertical_gaussian(bpolar, round(bluramount * 1.4))

        rcartes = polar_to_cartesian(
            rpolar, width=rdata.shape[1], height=rdata.shape[0])
        gcartes = polar_to_cartesian(
            gpolar, width=gdata.shape[1], height=gdata.shape[0])
        bcartes = polar_to_cartesian(
            bpolar, width=bdata.shape[1], height=bdata.shape[0])

        rfinal = Image.fromarray(np.uint8(rcartes), 'L')
        gfinal = Image.fromarray(np.uint8(gcartes), 'L')
        bfinal = Image.fromarray(np.uint8(bcartes), 'L')

    # enlarge the green and blue channels slightly, blue being the most enlarged
    gfinal = gfinal.resize((round((1 + 0.018 * strength) * rdata.shape[1]),
                            round((1 + 0.018 * strength) * rdata.shape[0])), Image.Resampling.LANCZOS)
    bfinal = bfinal.resize((round((1 + 0.044 * strength) * rdata.shape[1]),
                            round((1 + 0.044 * strength) * rdata.shape[0])), Image.Resampling.LANCZOS)

    rwidth, rheight = rfinal.size
    gwidth, gheight = gfinal.size
    bwidth, bheight = bfinal.size
    rhdiff = (bheight - rheight) // 2
    rwdiff = (bwidth - rwidth) // 2
    ghdiff = (bheight - gheight) // 2
    gwdiff = (bwidth - gwidth) // 2

    # Centre the channels
    im = Image.merge("RGB", (
        rfinal.crop((-rwdiff, -rhdiff, bwidth - rwdiff, bheight - rhdiff)),
        gfinal.crop((-gwdiff, -ghdiff, bwidth - gwdiff, bheight - ghdiff)),
        bfinal.crop((0, 0, bwidth, bheight))))

    # Crop the image to the original image dimensions
    return im.crop((rwdiff, rhdiff, rwidth + rwdiff, rheight + rhdiff))

if __name__ == '__main__':
    im = Image.open("Python_Image_Failures/sim1.jpg")  # Provide the correct path to your image

    # Ensure width and height are odd numbers
    if (im.size[0] % 2 == 0 or im.size[1] % 2 == 0):
        if (im.size[0] % 2 == 0):
            im = im.crop((0, 0, im.size[0] - 1, im.size[1]))
            im.load()
        if (im.size[1] % 2 == 0):
            im = im.crop((0, 0, im.size[0], im.size[1] - 1))
            im.load()

    og_im = im.copy()
    img = Image.open("Python_Image_Failures/sim1.jpg")  # Provide the correct path to your image
    im = add_chromatic(im, strength=1, no_blur=True)

    # Display the original and adjusted images
    plt.figure(num='Chromatic Aberration Adjustment')
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(im), plt.title('Chromatic Aberration Applied')
    plt.xticks([]), plt.yticks([])
    plt.show()