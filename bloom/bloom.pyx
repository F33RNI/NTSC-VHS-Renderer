# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, optimize.use_switch=True
# encoding: utf-8
"""
MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

# Pygame is require
try:
    import pygame
    from pygame.transform import scale, smoothscale
    from pygame import BLEND_RGB_ADD
    from pygame.surfarray import pixels3d, array3d, pixels_alpha, make_surface
    from pygame.image import tostring

except ImportError:
    raise ImportError("\n<pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

# Cython is require
try:
    cimport cython
    from cython.parallel cimport prange
    from cpython cimport PyObject, PyObject_HasAttr, PyObject_IsInstance, PyObject_CallFunctionObjArgs
    from cpython.list cimport PyList_Append, PyList_GetItem, PyList_Size, PyList_SetItem
    from cpython.dict cimport PyDict_Values, PyDict_Keys, PyDict_Items, PyDict_GetItem, \
        PyDict_SetItem, PyDict_Copy

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

# Numpy is require
try:
    import numpy
    from numpy import asarray, uint8, float32, zeros, float64, empty, uint8, float32

except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

cimport numpy as np

DEF THREADS = 8
DEF METHOD = 'static'

cdef:
    float ONE_SIXTEEN  = 1.0 / 16.0
    float FOUR_SIXTEEN = 4.0 / 16.0
    float SIX_SIXTEEN  = 6.0 / 16.0

DEF ONE_HALF      = 1.0 / 2.0
DEF ONE_FOURTH    = 1.0 / 4.0
DEF ONE_EIGHTH    = 1.0 / 8.0
DEF ONE_SIXTEENTH = 1.0 / 16.0

from libc.math cimport exp, pi

__version__ = "1.0.1"

# Version 1.0.0 to version 1.0.1 changes
# + Renamed bloom function bloom_effect_array24 to bloom_effect24
# + Renamed bloom function bloom_effect_array32 to bloom_effect32
# + Renamed bloom_effect_array24_inplace to bloom_effect24_inplace
# + Renamed bloom_effect32_inplace to bloom_effect32_inplace
# + Added example.py file
# + Changed Readme.md file

# CREATE GAUSSIAN BLUR WITH DIFFERENT KERNEL
# MASK FOR FUNCTION BLUR

# ----------------------------------------------- GAUSSIAN BLUR METHODS ------------------------------------------------


cpdef np.ndarray[np.uint8_t, ndim=3] blur5x5_array24(
        unsigned char [:, :, :] rgb_array_, unsigned char [:, :] mask=None):
    """
    APPLY A GAUSSIAN BLUE 5x5 TO A 3D NUMPY ARRAY (containing RGB pixels)
    
    # Gaussian kernel 5x5
       # |1   4   6   4  1|
       # |4  16  24  16  4|
       # |6  24  36  24  6|  x 1/256
       # |4  16  24  16  4|
       # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param rgb_array_: numpy.ndarray type (w, h, 3) uint8 
    :param mask      : numpy.ndarray type (w, h) uint8 use for masking the pixels (NOT USED)
    :return          : Return 24-bit a numpy.ndarray type (w, h, 3) uint8, Return a numpy.array 
    without alpha channel 
    """
    return asarray(blur5x5_array24_c(rgb_array_, mask))

cpdef blur5x5_array32(
        unsigned char [:, :, :] rgba_array_, mask=None):
    """
    APPLY A GAUSSIAN BLUE 5x5 TO A 3D NUMPY ARRAY (containing RGBA pixels)
    
    # Gaussian kernel 5x5
       # |1   4   6   4  1|
       # |4  16  24  16  4|
       # |6  24  36  24  6|  x 1/256
       # |4  16  24  16  4|
       # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param rgba_array_: numpy.ndarray type (w, h, 4) uint8 
    :param mask      : numpy.ndarray type (w, h) uint8 use for masking the pixels (NOT USED)
    :return          : Return 32-bit a numpy.ndarray type (w, h, 4) uint8 containing alpha values
    """

    return asarray(blur5x5_array32_c(rgba_array_, mask=None))


cpdef void blur5x5_array24_inplace(object rgb_array_, object mask=None):
    """
    APPLY A GAUSSIAN BLUR 5x5 EFFECT (INPLACE) TO A 3D ARRAY (containing RGB values)
        
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels, please refer to pygame 
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)
    
    :param rgb_array_   : object; numpy.ndarray type (w, h, 3) uint8 
    :param mask         : object; numpy.ndarray default None (NOT USED)
    :return             : void 
    """
    cdef bint own_data_flag = rgb_array_.flags.owndata

    # use pixels3d instead
    if own_data_flag:
        raise ValueError(
            "The array rgb_array_ does not owns the memory it uses or borrows it from another object.")

    blur5x5_array24_inplace_c(rgb_array_, mask)



cpdef void blur5x5_array32_inplace(object rgba_array_, object mask=None):
    """
    APPLY A GAUSSIAN BLUR 5x5 EFFECT (INPLACE) TO A 3D ARRAY (containing RGBA values)

    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels, please refer to pygame 
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    :param rgba_array_  : object; numpy.ndarray type (w, h, 4) uint8 
    :param mask         : object; numpy.ndarray type (w, h); default None (NOT USED)
    :return             : void 
    """

    cdef bint own_data_flag = (<object>rgba_array_).flags.owndata

    # use pixels3d instead
    if own_data_flag:
        raise ValueError(
            "The array rgba_array_ does not owns the memory it uses or borrows it from another object.")

    blur5x5_array32_inplace_c(rgba_array_, mask)
# ----------------------------------------------- GAUSSIAN BLUR METHODS ------------------------------------------------


# --------------------------------------------------- BLOOM METHODS ----------------------------------------------------

cpdef bloom_effect24(surface_, threshold_, smooth_ = 1, mask_ = None, fast_ = False):
    """
    CREATE A BLOOM EFFECT ON A PYGAME.SURFACE (COMPATIBLE 24 BIT SURFACE)
    THIS METHOD IS A 3D NUMPY NDARRAY AS STRUCTURE (CONTAINING RGB VALUES)

    * the bloom x2 x4 x8 x16 will be added to the final image except if the original
      image size is not big enough

    definition:
        bloom is a computer graphics effect used in video games, demos,
        and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    1) First apply a bright pass filter to the pygame surface(sdl surface) using methods
       bpf24_b_c or bpf32_b_c (adjust the threshold value to get the best filter effect).
    2) Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale
       method (no need to use smoothscale (bilinear filtering method).
    3) Apply a gaussian blur 5x5 effect on each of the downsized bpf images, if smooth_ is > 1,
       then the gaussian filter 5x5 will by applied more than once.
    4) Re-scale all the bpf images using a bilinear filter (width and height of original image).
       using an un-filtered rescaling method will pixelate the final output image.
       for best performances sets smoothscale acceleration. A value of 'generic' turns off
       acceleration. MMX uses mmx instructions only. SSE allows sse extensions as well.
    5) Blit all the bpf images on the original surface, use pygame additive blend mode for
       a smooth and brighter effect.

    notes:
    The downscaling process of all sub-images could be done in a single process to increase performance.

    :param fast_     : bool; True | False. Speed up the bloom process using only the x16 surface and using
    an optimized bright pass filter (texture size downscale x4 prior processing)
    :param mask_     : object; Numpy 1d array used for masking the pixels (default no mask)
    :param surface_  : pygame.surface compatible 24 - 32 bit format surface
    :param threshold_: integer; threshold value used by the bright pass algorithm (default 128)
    :param smooth_   : integer; number of gaussian blur 5x5 to apply to downsized images.
    :return          : returns a pygame.surface with a bloom effect (24 bit surface)
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface must be a pygame.Surface type got %s " % type(surface_)
    assert 0 <= threshold_ <= 255, "\nArgument threshold must be in range [0 ... 255]"
    assert isinstance(fast_, bool), "\nArgument fast must be a bool type got %s " % type(fast_)
    assert smooth_ > 0, "\nArgument smooth cannot be <= 0"

    return bloom_effect_array24_c(surface_, threshold_, smooth_, mask_, fast_)

cpdef bloom_effect32(surface_, threshold_, smooth_ = 1, mask_ = None, fast_ = False):
    """
    CREATE A BLOOM EFFECT ON A PYGAME.SURFACE (COMPATIBLE 32 BIT SURFACE)
    THIS METHOD IS A 3D NUMPY NDARRAY AS STRUCTURE (CONTAINING RGBA VALUES)

    * the bloom x2 x4 x8 x16 will be added to the final image except if the original
      image size is not big enough

    definition:
        bloom is a computer graphics effect used in video games, demos,
        and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    1) First apply a bright pass filter to the pygame surface(sdl surface) using methods
       bpf24_b_c or bpf32_b_c (adjust the threshold value to get the best filter effect).
    2) Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale
       method (no need to use smoothscale (bilinear filtering method).
    3) Apply a gaussian blur 5x5 effect on each of the downsized bpf images, if smooth_ is > 1,
       then the gaussian filter 5x5 will by applied more than once.
    4) Re-scale all the bpf images using a bilinear filter (width and height of original image).
       using an un-filtered rescaling method will pixelate the final output image.
       for best performances sets smoothscale acceleration. A value of 'generic' turns off
       acceleration. MMX uses mmx instructions only. SSE allows sse extensions as well.
    5) Blit all the bpf images on the original surface, use pygame additive blend mode for
       a smooth and brighter effect.

    notes:
    The downscaling process of all sub-images could be done in a single process to increase performance.

    :param fast_     : bool; True | False. Speed up the bloom process using only the x16 surface and using
    an optimized bright pass filter (texture size downscale x4 prior processing)
    :param mask_     : object; Numpy 1d array used for masking the pixels (default no mask)
    :param surface_  : pygame.surface compatible 24 - 32 bit format surface
    :param threshold_: integer; threshold value used by the bright pass algorithm (default 128)
    :param smooth_   : integer; number of gaussian blur 5x5 to apply to downsized images.
    :return          : returns a pygame.surface with a bloom effect (32 bit surface)
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface must be a pygame.Surface type got %s " % type(surface_)
    assert 0 <= threshold_ <= 255, "\nArgument threshold must be in range [0 ... 255]"
    assert isinstance(fast_, bool), "\nArgument fast must be a bool type got %s " % type(fast_)
    assert smooth_ > 0, "\nArgument smooth cannot be <= 0"

    return bloom_effect_array32_c(surface_, threshold_, smooth_, mask_, fast_)

# TODO smooth for inplace method
cpdef void bloom_effect24_inplace(object surface_, object threshold_, object fast_ = False)except *:
    """
    BLOOM A PYGAME SURFACE (24 - 32 BIT) INPLACE 
    
    :param surface_   : pygame.Surface; compatible 24-32 bit format 
    :param threshold_ : integer; threshold value used by the bright pass algorithm (default 128)
    :param fast_      : bool; True | False; If True the bloom effect will be approximated and only the x16 subsurface 
    will be processed to maximize the overall processing time, default is False).
    :return           : void
    """
    assert isinstance(surface_, pygame.Surface), \
        "Argument surface_ must be a pygame.Surface type got %s " % type(surface_)
    assert 0 <= threshold_ <= 255, "Argument threshold must be in range [0 ... 255]"
    assert isinstance(fast_, bool), "Argument fast_ must be a type bool got %s " % type(fast_)

    bloom_effect_array24_inplace_c(surface_, threshold_, fast_)

# TODO smooth for inplace method
cpdef void bloom_effect32_inplace(object surface_, object threshold_, object fast_ = False)except *:
    """
    BLOOM A PYGAME SURFACE 32 BIT INPLACE

    :param surface_   : pygame.Surface; compatible 32 bit format 
    :param threshold_ : integer; threshold value used by the bright pass algorithm (default 128)
    :param fast_      : bool; True | False; If True the bloom effect will be approximated and only the x16 subsurface 
    will be processed to maximize the overall processing time, default is False).
    :return           : void
    """
    assert isinstance(surface_, pygame.Surface), \
        "Argument surface_ must be a pygame.Surface type got %s " % type(surface_)
    assert 0 <= threshold_ <= 255, "Argument threshold must be in range [0 ... 255]"
    assert isinstance(fast_, bool), "Argument fast_ must be a type bool got %s " % type(fast_)

    bloom_effect_array32_inplace_c(surface_, threshold_, fast_)

# --------------------------------------------------- BLOOM METHODS ----------------------------------------------------

# kernel 5x5 separable
cdef:
    float [5] KERNEL = \
        numpy.array(([ONE_SIXTEEN,
                      FOUR_SIXTEEN,
                      SIX_SIXTEEN,
                      FOUR_SIXTEEN,
                      ONE_SIXTEEN]), dtype=float32, copy=False)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] blur5x5_array24_c(
        unsigned char [:, :, :] rgb_array_,
        object mask = None):

    cdef int w, h, dim

    try:
        w, h, dim = (<object>rgb_array_).shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve  = empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = empty((w, h, 3), dtype=uint8)
        short int kernel_length = len(KERNEL)
        int x, y, xx, yy
        float r, g, b, s
        char kernel_offset
        unsigned char red, green,blue
        unsigned int h_1 = h - 1
        unsigned int w_1 = w - 1
        float *k
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3
        unsigned char *c4
        unsigned char *c5
        unsigned char *c6

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule=METHOD, num_threads=THREADS):

            c1 = &rgb_array_[0, y, 0]
            c2 = &rgb_array_[0, y, 1]
            c3 = &rgb_array_[0, y, 2]
            c4 = &rgb_array_[w_1, y, 0]
            c5 = &rgb_array_[w_1, y, 1]
            c6 = &rgb_array_[w_1, y, 2]

            for x in range(0, w):

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &KERNEL[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = c1[0], c2[0], c3[0]
                    elif xx > (w - 1):
                        red, green, blue = c4[0], c5[0], c6[0]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0], \
                                           rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule=METHOD, num_threads=THREADS):

            c1 = &convolve[x, 0, 0]
            c2 = &convolve[x, 0, 1]
            c3 = &convolve[x, 0, 2]
            c4 = &convolve[x, h_1, 0]
            c5 = &convolve[x, h_1, 1]
            c6 = &convolve[x, h_1, 2]

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &KERNEL[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = c1[0], c2[0], c3[0]
                    elif yy > (h - 1):
                        red, green, blue = c4[0], c5[0], c6[0]
                    else:
                        red, green, blue = convolve[x, yy, 0], \
                                           convolve[x, yy, 1], convolve[x, yy, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                convolved[x, y, 0], convolved[x, y, 1], convolved[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b

    return convolved



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] blur5x5_array32_c(
        unsigned char [:, :, :] rgba_array_,
        object mask = None):

    cdef int w, h, dim

    try:
        w, h, dim = rgba_array_.shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve  = empty((w, h, 3), dtype=uint8)  # NOTE depth = 3
        unsigned char [:, :, ::1] convolved = empty((w, h, 4), dtype=uint8)
        short int kernel_length = len(KERNEL)
        int x, y, xx, yy
        float r, g, b
        char kernel_offset
        unsigned int h_1 = h - 1
        unsigned int w_1 = w - 1
        unsigned char red, green, blue
        float *k
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3
        unsigned char *c4
        unsigned char *c5
        unsigned char *c6

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule=METHOD, num_threads=THREADS):

            c1 = &rgba_array_[0, y, 0]
            c2 = &rgba_array_[0, y, 1]
            c3 = &rgba_array_[0, y, 2]
            c4 = &rgba_array_[w_1, y, 0]
            c5 = &rgba_array_[w_1, y, 1]
            c6 = &rgba_array_[w_1, y, 2]

            for x in range(0, w):

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &KERNEL[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = c1[0], c2[0], c3[0]

                    elif xx > w_1:
                        red, green, blue = c4[0], c5[0], c6[0]

                    else:
                        red   = rgba_array_[xx, y, 0]
                        green = rgba_array_[xx, y, 1]
                        blue  = rgba_array_[xx, y, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule=METHOD, num_threads=THREADS):

            c1 = &convolve[x, 0, 0]
            c2 = &convolve[x, 0, 1]
            c3 = &convolve[x, 0, 2]
            c4 = &convolve[x, h_1, 0]
            c5 = &convolve[x, h_1, 1]
            c6 = &convolve[x, h_1, 2]

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &KERNEL[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = c1[0], c2[0], c3[0]

                    elif yy > h_1:
                        red, green, blue = c4[0], c5[0], c6[0]

                    else:
                        red   = convolve[x, yy, 0]
                        green = convolve[x, yy, 1]
                        blue  = convolve[x, yy, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                convolved[x, y, 0], convolved[x, y, 1],\
                convolved[x, y, 2], convolved[x, y, 3] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b, rgba_array_[x, y, 3]

    return convolved



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void blur5x5_array24_inplace_c(unsigned char [:, :, :] rgb_array_, object mask=None):

    cdef int w, h
    w, h = rgb_array_.shape[:2]

    # cdef bint own_data_flag = (<object>rgb_array_).flags.owndata
    # if not own_data_flag:
    #     raise ValueError(
    #     '\nThe array rgb_array_ does not owns the memory it uses or borrows it from another object. ')

    # kernel 5x5 separable
    cdef:

        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = empty((w, h, 3), dtype=uint8)
        short int kernel_length = len(KERNEL)
        int x, y, xx, yy
        float r, g, b, s
        char kernel_offset
        unsigned char red, green, blue
        float *k
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3
        unsigned char *c4
        unsigned char *c5
        unsigned char *c6
        unsigned int w_1 = w - 1
        unsigned int h_1 = h - 1

    with nogil:

        # horizontal convolution
        for y in prange(0, h, schedule='static', num_threads=THREADS):

            c1 = &rgb_array_[0, y, 0]
            c2 = &rgb_array_[0, y, 1]
            c3 = &rgb_array_[0, y, 2]
            c4 = &rgb_array_[w_1, y, 0]
            c5 = &rgb_array_[w_1, y, 1]
            c6 = &rgb_array_[w_1, y, 2]

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &KERNEL[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = c1[0], c2[0], c3[0]
                    elif xx > (w - 1):
                        red, green, blue = c4[0], c5[0], c6[0]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule='static', num_threads=THREADS):

            c1 = &convolve[x, 0, 0]
            c2 = &convolve[x, 0, 1]
            c3 = &convolve[x, 0, 2]
            c4 = &convolve[x, h_1, 0]
            c5 = &convolve[x, h_1, 1]
            c6 = &convolve[x, h_1, 2]

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &KERNEL[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = c1[0], c2[0], c3[0]
                    elif yy > (h -1):
                        red, green, blue = c4[0], c5[0], c6[0]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                rgb_array_[x, y, 0], rgb_array_[x, y, 1], rgb_array_[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void blur5x5_array32_inplace_c(unsigned char [:, :, :] rgba_array_, object mask=None):

    cdef int w, h
    w, h = rgba_array_.shape[:2]

    # cdef bint own_data_flag = (<object>rgba_array_).flags.owndata
    # if not own_data_flag:
    #     raise ValueError(
    #     '\nThe array rgba_array_ does not owns the memory it uses or borrows it from another object. ')

    # kernel 5x5 separable
    cdef:

        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = empty((w, h, 3), dtype=uint8)  # Note depth = 3
        unsigned char [:, :, ::1] convolved = empty((w, h, 4), dtype=uint8)
        short int kernel_length = len(KERNEL)
        int x, y, xx, yy
        float r, g, b, s
        char kernel_offset
        unsigned char red, green, blue
        float *k
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3
        unsigned char *c4
        unsigned char *c5
        unsigned char *c6
        unsigned int w_1 = w - 1
        unsigned int h_1 = h - 1

    with nogil:

        # horizontal convolution
        for y in prange(0, h, schedule='static', num_threads=THREADS):

            c1 = &rgba_array_[0, y, 0]
            c2 = &rgba_array_[0, y, 1]
            c3 = &rgba_array_[0, y, 2]
            c4 = &rgba_array_[w_1, y, 0]
            c5 = &rgba_array_[w_1, y, 1]
            c6 = &rgba_array_[w_1, y, 2]

            for x in range(0, w):

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &KERNEL[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = c1[0], c2[0], c3[0]
                    elif xx > (w - 1):
                        red, green, blue = c4[0], c5[0], c6[0]
                    else:
                        red, green, blue = rgba_array_[xx, y, 0],\
                            rgba_array_[xx, y, 1], rgba_array_[xx, y, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule='static', num_threads=THREADS):

            c1 = &convolve[x, 0, 0]
            c2 = &convolve[x, 0, 1]
            c3 = &convolve[x, 0, 2]
            c4 = &convolve[x, h_1, 0]
            c5 = &convolve[x, h_1, 1]
            c6 = &convolve[x, h_1, 2]

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &KERNEL[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = c1[0], c2[0], c3[0]
                    elif yy > (h -1):
                        red, green, blue = c4[0], c5[0], c6[0]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                rgba_array_[x, y, 0] = <unsigned char>r
                rgba_array_[x, y, 1] = <unsigned char>g
                rgba_array_[x, y, 2] = <unsigned char>b
                rgba_array_[x, y, 3] = rgba_array_[x, y, 3]


cpdef np.ndarray[np.float64_t, ndim=2] kernel_deviation(double sigma, unsigned short int kernel_size):
    """
    TESTING ONLY - DO NOT USE
    """
    return kernel_deviation_c(sigma, kernel_size)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t, ndim=2] kernel_deviation_c(double sigma, unsigned short int kernel_size):
    """
    Sample Gaussian matrix
    This is a sample matrix, produced by sampling the Gaussian filter kernel 
    (with σ = 0.84089642) at the midpoints of each pixel and then normalizing.
    Note that the center element (at [4, 4]) has the largest value, decreasing
     symmetrically as distance from the center increases.
    0.00000067	0.00002292	0.00019117	0.00038771	0.00019117	0.00002292	0.00000067
    0.00002292	0.00078633	0.00655965	0.01330373	0.00655965	0.00078633	0.00002292
    0.00019117	0.00655965	0.05472157	0.11098164	0.05472157	0.00655965	0.00019117
    0.00038771	0.01330373	0.11098164	0.22508352	0.11098164	0.01330373	0.00038771
    0.00019117	0.00655965	0.05472157	0.11098164	0.05472157	0.00655965	0.00019117
    0.00002292	0.00078633	0.00655965	0.01330373	0.00655965	0.00078633	0.00002292
    0.00000067	0.00002292	0.00019117	0.00038771	0.00019117	0.00002292	0.00000067

    :param sigma        : float; Kernel sigma value  
    :param kernel_size  : kernel size example 3x3, 5x5 etc 
    :return             : a numpy.ndarray kernel 

    """

    if sigma==0.0:
        raise ValueError('Argument sigma cannot be equal to zero.')
    if kernel_size <=0:
        raise ValueError('Argument kernel_size cannot be <=0.')

    # In two dimensions, it is the product of two such Gaussian functions, one in each dimension:
    # 1 / (2 * math.pi * (sigma ** 2)) * math.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    cdef:
        double g1, g2
        double [:, :] kernel = zeros((kernel_size, kernel_size), dtype=float64)
        int half_kernel = kernel_size >> 1
        int x, y

    g1 = 1.0 / (2.0 * pi * (sigma * sigma))
    for x in range(-half_kernel, half_kernel+1):
            for y in range(-half_kernel, half_kernel+1):
                    g2 = exp(-((x * x + y *y) / (2.0 * sigma * sigma)))
                    g = g1 * g2
                    kernel[x + half_kernel, y + half_kernel] = g
    return asarray(kernel)




cpdef object test_bpf24_c(unsigned char [:, :, :] input_array_, unsigned char threshold = 128):
    """
    TESTING ONLY - DO NOT USE 
    """
    return bpf24_c(input_array_, threshold)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object bpf24_c(unsigned char [:, :, :] input_array_, unsigned char threshold = 128):
    """
    BRIGHT PASS FILTER

    INPUT : numpy.ndarray shape (w, h, 3) containing RGB pixels type (uint8) 
    ________
    
    OUTPUT : pygame.Surface (w, h) compatible 24, 32 bit without alpha transparency 
    ________
    
    * Conserve only the brightest pixels in an array 

    * The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels, please refer to pygame 
      function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    :param input_array_: numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels 
    :param threshold   : float Bright pass threshold default 128 must be in range [0 ... 255]
    :return            :  Return a pygame Surface compatible 24-32 bit without per-pixel transparency 
    """

    cdef:
        int w, h

    w, h = input_array_.shape[:2]

    assert w > 0, "Array width cannot be null "
    assert h > 0, "Array height cannot be null"

    cdef:
        int i = 0, j = 0
        float lum, c
        unsigned char [:, :, :] output_array_ = zeros((h, w, 3), uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(0, h, schedule='static', num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &input_array_[i, j, 0]
                g = &input_array_[i, j, 1]
                b = &input_array_[i, j, 2]

                lum = r[0] * 0.299 + g[0] * 0.587 + b[0] * 0.114
                # no div by zero lum must be strictly > 0
                if lum > threshold:
                    c = (lum - threshold) / lum
                    output_array_[j, i, 0] = <unsigned char>(r[0] * c)
                    output_array_[j, i, 1] = <unsigned char>(g[0] * c)
                    output_array_[j, i, 2] = <unsigned char>(b[0] * c)

    return pygame.image.frombuffer(output_array_, (w, h), 'RGB')

cpdef void test_bpf24_inplace(unsigned char [:, :, :] input_array_, unsigned char threshold = 128):
    """
    TESTING ONLY - DO NOT USE 
    """
    bpf24_inplace_c(input_array_, threshold)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void bpf24_inplace_c(unsigned char [:, :, :] input_array_, unsigned char threshold = 128):
    """
    BRIGHT PASS FILTER

    INPUT : numpy.ndarray shape (w, h, 3) containing RGB pixels type (uint8) 
    ________

    OUTPUT : void
    ________

    * Conserve only the brightest pixels in an array 

    * The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels, please refer to pygame 
      function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    :param input_array_: numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels 
    :param threshold   : float Bright pass threshold default 128 must be in range [0 ... 255]
    :return            : void
    """

    cdef:
        int w, h

    w, h = input_array_.shape[:2]

    assert w > 0, "Array width cannot be null "
    assert h > 0, "Array height cannot be null"

    # cdef bint own_data_flag = input_array_.flags.owndata
    #
    # # use pixels3d instead
    # if own_data_flag:
    #     raise ValueError(
    #         "The array rgba_array_ does not owns the memory it uses or borrows it from another object.")

    cdef:
        int i = 0, j = 0
        float lum, c
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(0, h, schedule='static', num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &input_array_[i, j, 0]
                g = &input_array_[i, j, 1]
                b = &input_array_[i, j, 2]

                lum = r[0] * 0.299 + g[0] * 0.587 + b[0] * 0.114
                # no div by zero lum must be strictly > 0
                if lum > threshold:
                    c = (lum - threshold) / lum
                    input_array_[i, j, 0] = <unsigned char>(r[0] * c)
                    input_array_[i, j, 1] = <unsigned char>(g[0] * c)
                    input_array_[i, j, 2] = <unsigned char>(b[0] * c)
                else:
                    input_array_[i, j, 0] = 0
                    input_array_[i, j, 1] = 0
                    input_array_[i, j, 2] = 0



cpdef object test_bpf32_c(object image_, unsigned char threshold_ = 128):
    """
    TESTING ONLY - DO NOT USE 
    """
    return asarray(bpf32_c(image_, threshold_))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] bpf32_c(object image_, unsigned char threshold_ = 128):
    """
    BRIGHT PASS FILTER COMPATIBLE 32-BIT SURFACE

    INPUT      
    image_     : pygame.Surface compatible 32 bit with alpha transparency
    threshold_ : integer value in range [0 ... 255] bright pass threshold value
    ________
    
    OUTPUT 
    Return a MemoryViewSlice shape (w, h, 4) containing RGBA pixels (contiguous array), 
    filtered (only bright area of the image remains).
    ________

    Bright pass filter for 32-bit image (method using 3d array data structure)
    
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.

    :param image_    : pygame.Surface 32 bit format (RGBA) with per-pixel information
    :param threshold_: integer; Threshold to consider for filtering pixels luminance values,
    default is 128 range [0..255] unsigned char (python integer)
    :return          : Return a 3d numpy.ndarray type (w, h, 4) filtered (only bright area of the image remains).
    """

    assert isinstance(image_, pygame.Surface), \
           "\nExpecting pygame surface for argument image, got %s " % type(image_)

    # make sure the surface is 32-bit format RGB
    if not (image_.get_bitsize() == 32 and image_.get_bytesize() == 4):
        raise ValueError('Surface is not a 32-bit format or does not contain transparency layer.')

    try:
        rgba_array_  = pixels3d(image_)
        alpha_array_ = pixels_alpha(image_)

    except (pygame.error, ValueError) as error:
        raise ValueError('\nSurface is not 32 bit or surface is missing the alpha layer.\n%s ' % error)

    cdef:
        int w, h

    w, h = image_.get_size()

    assert w > 0, "Array width cannot be null "
    assert h > 0, "Array height cannot be null"

    cdef:
        unsigned char [:, :, :] rgba_array = rgba_array_
        unsigned char [:, :, ::1] out_rgba = zeros((w, h, 4), uint8)
        unsigned char [:, :] alpha_array   = alpha_array_
        int i = 0, j = 0
        float lum, c
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3

    with nogil:
        for j in prange(0, h, schedule=METHOD, num_threads=THREADS):
            for i in range(0, w):
                c1 = &rgba_array[i, j, 0]
                c2 = &rgba_array[i, j, 1]
                c3 = &rgba_array[i, j, 2]

                # ITU-R BT.601 luma coefficients
                lum = c1[0] * 0.299 + c2[0] * 0.587 + c3[0] * 0.114

                if lum > threshold_:
                    c = (lum - threshold_) / lum
                    out_rgba[i, j, 0] = <unsigned char>(c1[0] * c)
                    out_rgba[i, j, 1] = <unsigned char>(c2[0] * c)
                    out_rgba[i, j, 2] = <unsigned char>(c3[0] * c)
                    out_rgba[i, j, 3] = alpha_array[i, j]

    return out_rgba


cpdef void test_bpf32_inplace(object image_, unsigned char threshold_ = 128):
    """
    TESTING ONLY - DO NOT USE 
    """
    bpf32_inplace_c(image_, threshold_)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void bpf32_inplace_c(object image_, unsigned char threshold_ = 128):
    """
    BRIGHT PASS FILTER COMPATIBLE 32-BIT SURFACE

    INPUT      
    image_     : pygame.Surface compatible 32 bit with alpha transparency
    threshold_ : integer value in range [0 ... 255] bright pass threshold value
    ________

    OUTPUT 
    void
    ________

    Bright pass filter for 32-bit image (method using 3d array data structure)

    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.

    :param image_    : pygame.Surface 32 bit format (RGBA) with per-pixel information
    :param threshold_: integer; Threshold to consider for filtering pixels luminance values,
    default is 128 range [0..255] unsigned char (python integer)
    :return          : void
    """

    assert isinstance(image_, pygame.Surface), \
           "\nExpecting pygame surface for argument image, got %s " % type(image_)

    # make sure the surface is 32-bit format RGB
    if not (image_.get_bitsize() == 32 and image_.get_bytesize() == 4):
        raise ValueError('Surface is not a 32-bit format or does not contain transparency layer.')

    try:
        rgb_array_  = pixels3d(image_)
        alpha_array_ = pixels_alpha(image_)

    except (pygame.error, ValueError) as error:
        raise ValueError('\nSurface is not 32 bit or surface is missing the alpha layer.\n%s ' % error)

    cdef:
        int w, h

    w, h = image_.get_size()

    assert w > 0, "Array width cannot be null "
    assert h > 0, "Array height cannot be null"

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_array_
        unsigned char [:, :] alpha_array   = alpha_array_
        int i = 0, j = 0
        float lum, c
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3

    with nogil:
        for j in prange(0, h, schedule=METHOD, num_threads=THREADS):
            for i in range(0, w):
                c1 = &rgb_array[i, j, 0]
                c2 = &rgb_array[i, j, 1]
                c3 = &rgb_array[i, j, 2]

                # ITU-R BT.601 luma coefficients
                lum = c1[0] * 0.299 + c2[0] * 0.587 + c3[0] * 0.114

                if lum > threshold_:
                    c = (lum - threshold_) / lum
                    rgb_array[i, j, 0] = <unsigned char>(c1[0] * c)
                    rgb_array[i, j, 1] = <unsigned char>(c2[0] * c)
                    rgb_array[i, j, 2] = <unsigned char>(c3[0] * c)
                else:
                    rgb_array[i, j, 0] = 0
                    rgb_array[i, j, 1] = 0
                    rgb_array[i, j, 2] = 0

cpdef inline void filtering24(object surface_, float [:, :] mask_):
    """
    TESTING ONLY - DO NOT USE 
    """
    filtering24_inplace_c(surface_, mask_)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void filtering24_inplace_c(object surface_, float [:, :] mask_):
    """
    APPLY A MASK TO AN IMAGE (24 - 32 format with or without alpha transparency)
    
    * The mask is a numpy.ndarray shape (w, h) containing float values in range [0.0 ... 1.0]
      All the pixels are multiply by mask values. Wherever the mask value equal to 0.0, the final image 
      pixel will have its value set to 0   
    
    * The image can be 24-32 bit with or without per pixel transparency 
    * The mask must have the same dimension than the surface, identical width and height 

    :param surface_: object; pygame.Surface compatible 24-32 bit
    :param mask_   : numpy.ndarray; 2D array shape (w, h) type float, values in range [0.0 ... 1.0]
    :return        : void 
    """

    cdef:
        int w, h, w_, h_
    w, h = surface_.get_size()

    try:
        w_, h_ = mask_.shape[:2]

    except (ValueError, pygame.error):
       raise ValueError('\nArgument mask_ type not understood, '
           'expecting numpy.ndarray type (w, h) got %s ' % type(mask_))

    assert w == w_ and h == h_, \
        '\nSurface and mask size does not match (w:%s, h:%s), ' \
        '(w_:%s, h_:%s) ' % (w, h, w_, h_)

    try:
        rgb_array_ = pixels3d(surface_)

    except (ValueError, pygame.error):
        raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_array_
        float [:, :] mask = mask_
        int i, j
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3
        float c

    with nogil:
        for j in prange(0, h, schedule=METHOD, num_threads=THREADS):
            for i in range(w):

                c = mask[i, j]

                c1 = &rgb_array[i, j, 0]
                c2 = &rgb_array[i, j, 1]
                c3 = &rgb_array[i, j, 2]

                c1[0] = <unsigned char>(c1[0] * c)
                c2[0] = <unsigned char>(c2[0] * c)
                c3[0] = <unsigned char>(c3[0] * c)



cpdef np.ndarray[np.float32_t, ndim=2] build_mask_from_surface(object surface_, bint invert_mask = False):
    """
    BUILD A MASKING ARRAY SHAPE (W, H) TYPE FLOAT (NORMALIZED)
    
    :param surface_    : pygame.Surface; compatible 24 - 32 bit surface with or without transparency layer. The 
    surface size must be equivalent to the surface being bloomed
    :param invert_mask : bool; True | False invert the final mask values 
    :return            : Return a numpy.ndarray shape (w, h) type float32 (python float). All the values are 
    normalized 
    """

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_array_ = pixels3d(surface_)

    except (ValueError, pygame.error):
        raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_array_
        float [:, :] mask = empty((w, h), float32)
        int i, j
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3

    with nogil:
        for i in prange(0, w, schedule=METHOD, num_threads=THREADS):
            for j in range(h):

                c1 = &rgb_array[i, j, 0]
                c2 = &rgb_array[i, j, 1]
                c3 = &rgb_array[i, j, 2]

                mask[i, j] = (c1[0] + c2[0] + c3[0]) * 1.0 / 765.0  # create a grayscale and Normalized

                if invert_mask:
                    mask[i, j] = 1.0 - mask[i, j]

    return asarray(mask)


cpdef inline void filtering32(object surface_, float [:, :] mask_):
    """
    TESTING ONLY - DO NOT USE 
    """
    filtering32_inplace_c(surface_, mask_)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void filtering32_inplace_c(object surface_, float [:, :] mask_):
    """   
     APPLY A MASK TO AN IMAGE (32 BIT FORMAT WITH ALPHA TRANSPARENCY)
    
    * The mask is a numpy.ndarray shape (w, h) containing float values in range [0.0 ... 1.0]
      All the pixels are multiply by mask values. Wherever the mask value equal to 0.0, the final image 
      pixel will have its value set to 0   
    
    * The image can be 32 bit with per pixel transparency 
    * The mask must have the same dimension than the surface, identical width and height 

    :param surface_: object; pygame.Surface compatible 32 bit with alpha transparency 
    :param mask_   : numpy.ndarray; 2D array shape (w, h) type float, values in range [0.0 ... 1.0]
    :return        : void 
    """

    cdef int w, h, w_, h_
    w, h = surface_.get_size()

    try:
        w_, h_ = mask_.shape[:2]

    except (ValueError, pygame.error):
        raise ValueError('\nArgument mask_ type not understood, '
                         'expecting numpy.ndarray type (w, h) got %s ' % type(mask_))

    assert w == w_ and h == h_, \
        '\nSurface and mask size does not match (w:%s, h:%s), ' \
        '(w_:%s, h_:%s) ' % (w, h, w_, h_)

    try:
        rgb_array_ = pixels3d(surface_)
    except (ValueError, pygame.error):
        raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_array_
        float [:, :] mask = mask_
        int i, j
        unsigned char *c0
        unsigned char *c1
        unsigned char *c2
        # unsigned char *c3
        float c

    with nogil:
        for j in prange(0, h, schedule=METHOD, num_threads=THREADS):
            for i in range(w):
                c0 = &rgb_array[i, j, 0]
                c1 = &rgb_array[i, j, 1]
                c2 = &rgb_array[i, j, 2]
                # c3 = &rgb_array[i, j, 3]
                c = mask[i, j]
                c0[0] = <unsigned char>(c0[0] * c)
                c1[0] = <unsigned char>(c1[0] * c)
                c2[0] = <unsigned char>(c2[0] * c)
                # c3[0] = <unsigned char>(c3[0] * c)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void bloom_effect_array24_inplace_c(object surface_, unsigned int threshold_, bint fast_ = False):

    cdef:
        int w, h, bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    with nogil:
        w2, h2 = w >> 1, h >> 1
        w4, h4 = w2 >> 1, h2 >> 1
        w8, h8 = w4 >> 1, h4 >> 1
        w16, h16 = w8 >> 1, h8 >> 1

        if w2 > 0 and h2 > 0:
            x2 = True
        else:
            x2 = False

        if w4 > 0 and h4 > 0:
            x4 = True
        else:
            x4 = False

        if w8 > 0 and h8 > 0:
            x8 = True
        else:
            x8 = False

        if w16 > 0 and h16 > 0:
            x16 = True
        else:
            x16 = False

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False

    surface_cp = bpf24_c(pixels3d(surface_), threshold=threshold_)

    # FIRST SUBSURFACE DOWNSCALE x2
    # THIS IS THE MOST EXPENSIVE IN TERM OF PROCESSING TIME
    # cdef:
    #     unsigned char [:, :, :] s2_array
    #     unsigned char [:, :, :] s4_array
    #     unsigned char [:, :, :] s8_array
    #     unsigned char [:, :, :] s16_array

    if x2:
        s2 = scale(surface_cp, (w2, h2))
        s2_array = asarray(s2.get_view('3'), dtype=uint8)
        blur5x5_array24_inplace_c(s2_array)
        b2_blurred = make_surface(s2_array)
        s2 = smoothscale(b2_blurred, (w, h))
        surface_.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:
        s4 = scale(surface_cp, (w4, h4))
        s4_array = asarray(s4.get_view('3'), dtype=uint8)
        blur5x5_array24_inplace_c(s4_array)
        b4_blurred = make_surface(s4_array)
        s4 = smoothscale(b4_blurred, (w, h))
        surface_.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:
        s8 = scale(surface_cp, (w8, h8))
        s8_array = asarray(s8.get_view('3'), dtype=uint8)
        blur5x5_array24_inplace_c(s8_array)
        b8_blurred = make_surface(s8_array)
        s8 = smoothscale(b8_blurred, (w, h))
        surface_.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    # FOURTH SUBSURFACE DOWNSCALE x16
    # LESS SIGNIFICANT IN TERMS OF RENDERING AND PROCESSING TIME
    if x16:
        s16 = scale(surface_cp, (w16, h16))
        s16_array = asarray(s16.get_view('3'), dtype=uint8)
        blur5x5_array24_inplace_c(s16_array)
        b16_blurred = make_surface(s16_array)
        s16 = smoothscale(b16_blurred, (w, h))
        surface_.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void bloom_effect_array32_inplace_c(object surface_, unsigned int threshold_, bint fast_ = False):

    cdef:
        int w, h, bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    with nogil:
        w2, h2 = w >> 1, h >> 1
        w4, h4 = w2 >> 1, h2 >> 1
        w8, h8 = w4 >> 1, h4 >> 1
        w16, h16 = w8 >> 1, h8 >> 1

        if w2 > 0 and h2 > 0:
            x2 = True
        else:
            x2 = False

        if w4 > 0 and h4 > 0:
            x4 = True
        else:
            x4 = False

        if w8 > 0 and h8 > 0:
            x8 = True
        else:
            x8 = False

        if w16 > 0 and h16 > 0:
            x16 = True
        else:
            x16 = False

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False

    surface_cp = bpf24_c(pixels3d(surface_), threshold=threshold_)

    # FIRST SUBSURFACE DOWNSCALE x2
    # THIS IS THE MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x2:
        s2 = scale(surface_cp, (w2, h2))
        s2_array = asarray(s2.get_view('3'), dtype=uint8)
        blur5x5_array24_inplace_c(s2_array)
        b2_blurred = make_surface(s2_array)
        s2 = smoothscale(b2_blurred, (w, h))
        surface_.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:
        s4 = scale(surface_cp, (w4, h4))
        s4_array = asarray(s4.get_view('3'), dtype=uint8)
        blur5x5_array24_inplace_c(s4_array)
        b4_blurred = make_surface(s4_array)
        s4 = smoothscale(b4_blurred, (w, h))
        surface_.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:
        s8 = scale(surface_cp, (w8, h8))
        s8_array = asarray(s8.get_view('3'), dtype=uint8)
        blur5x5_array24_inplace_c(s8_array)
        b8_blurred = make_surface(s8_array)
        s8 = smoothscale(b8_blurred, (w, h))
        surface_.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    # FOURTH SUBSURFACE DOWNSCALE x16
    # LESS SIGNIFICANT IN TERMS OF RENDERING AND PROCESSING TIME
    if x16:
        s16 = scale(surface_cp, (w16, h16))
        s16_array = asarray(s16.get_view('3'), dtype=uint8)
        blur5x5_array24_inplace_c(s16_array)
        b16_blurred = make_surface(s16_array)
        s16 = smoothscale(b16_blurred, (w, h))
        surface_.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline bloom_effect_array24_c(
        object surface_,
        unsigned char threshold_,
        unsigned short int smooth_ = 1,
        object mask_               = None,
        bint fast_                 = False):

    surface_cp = surface_.copy()

    cdef:
        int w, h, bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    with nogil:
        w2, h2 = w >> 1, h >> 1
        w4, h4 = w2 >> 1, h2 >> 1
        w8, h8 = w4 >> 1, h4 >> 1
        w16, h16 = w8 >> 1, h8 >> 1

        if w2 > 0 and h2 > 0:
            x2 = True
        else:
            x2 = False

        if w4 > 0 and h4 > 0:
            x4 = True
        else:
            x4 = False

        if w8 > 0 and h8 > 0:
            x8 = True
        else:
            x8 = False

        if w16 > 0 and h16 > 0:
            x16 = True
        else:
            x16 = False

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False

    bpf_surface = bpf24_c(pixels3d(surface_), threshold=threshold_)

    cdef unsigned short int r
    # FIRST SUBSURFACE DOWNSCALE x2
    # THIS IS THE MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x2:
        s2 = scale(bpf_surface, (w2, h2))

        s2_array = asarray(s2.get_view('3'), dtype=uint8)
        for r in range(smooth_):
            s2_array = blur5x5_array24_c(s2_array)

        b2_blurred_surface = make_surface(asarray(s2_array))
        s2 = smoothscale(b2_blurred_surface, (w, h))
        surface_cp.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:
        s4 = scale(bpf_surface, (w4, h4))
        s4_array = asarray(s4.get_view('3'), dtype=uint8)

        for r in range(smooth_):
            s4_array = blur5x5_array24_c(s4_array)

        b4_blurred_surface = make_surface(asarray(s4_array))
        s4 = smoothscale(b4_blurred_surface, (w, h))
        surface_cp.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:
        s8 = scale(bpf_surface, (w8, h8))
        s8_array = asarray(s8.get_view('3'), dtype=uint8)

        for r in range(smooth_):
            s8_array = blur5x5_array24_c(s8_array)

        b8_blurred_surface = make_surface(asarray(s8_array))
        s8 = smoothscale(b8_blurred_surface, (w, h))
        surface_cp.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    # FOURTH SUBSURFACE DOWNSCALE x16
    # LESS SIGNIFICANT IN TERMS OF RENDERING AND PROCESSING TIME
    if x16:
        s16 = scale(bpf_surface, (w16, h16))
        s16_array = asarray(s16.get_view('3'), dtype=uint8)

        for r in range(smooth_):
            s16_array = blur5x5_array24_c(s16_array)

        b16_blurred_surface = make_surface(asarray(s16_array))
        s16 = smoothscale(b16_blurred_surface, (w, h))
        surface_cp.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    if mask_ is not None:
        # Multiply surface with mask values
        # RGB pixels = 0 when mask value = 0.0 otherwise modify RGB amplitude
        filtering24_inplace_c(surface_cp, mask_)

    return surface_cp


cpdef test_array32_rescale(rgb_array, w2, h2):
    """
    TEST ONLY - DO NOT USE   
    """
    return array32_rescale_c(rgb_array, w2, h2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, :] array32_rescale_c(unsigned char [:, :, :] rgb_array, int w2, int h2):
    """
    RESCALE A GIVEN 3D ARRAY SHAPE (W, H, 4)
    
    INPUT  : 
    rgb_array : Numpy.ndarray shape (w, h, 4) containing RGBA pixels (uint8) 
    w2        : integer; new array width 
    h2        : integer; new array height
    
    ____________
    
    OUTPUT : Rescaled array shape (h2, w2, 4) containing RGBA pixels (uint8). The final array is the original 
    array rescaled and transposed. The returned object is a MemoryViewSlice (contiguous)
    ____________  
    
    :param rgb_array: RGB numpy.ndarray, format (w, h, 4) numpy.uint8 with alpha channel
    :param w2       : integer; new width 
    :param h2       : integer; new height
    :return         : Return a MemoryViewSlice 3d numpy.ndarray format (w, h, 4) uint8
    """

    cdef int w1, h1, s
    try:
        w1, h1, s = (<object>rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char [:, :, :] new_array = zeros((h2, w2, 4), uint8)
        float fx = <float>w1 / <float>w2
        float fy = <float>h1 / <float>h2
        int x, y, xx, yy
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3
        unsigned char *c4

    with nogil:
        for x in prange(w2, schedule=METHOD, num_threads=THREADS):
            xx = <int>(x * fx)
            for y in range(h2):
                yy = <int>(y * fy)
                c1 = &rgb_array[xx, yy, 0]
                c2 = &rgb_array[xx, yy, 1]
                c3 = &rgb_array[xx, yy, 2]
                c4 = &rgb_array[xx, yy, 3]

                new_array[y, x, 0] = c1[0]
                new_array[y, x, 1] = c2[0]
                new_array[y, x, 2] = c3[0]
                new_array[y, x, 3] = c4[0]

    return new_array


cdef inline bloom_effect_array32_c(
        object surface_,
        unsigned char threshold_,
        unsigned short int smooth_ = 1,
        object mask_               = None,
        bint fast_                 = False):

    surface_cp = surface_.copy()

    cdef:
        int w, h, bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    with nogil:
        w2, h2   = w >> 1, h >> 1
        w4, h4   = w2 >> 1, h2 >> 1
        w8, h8   = w4 >> 1, h4 >> 1
        w16, h16 = w8 >> 1, h8 >> 1

        if w2 > 0 and h2 > 0:
            x2 = True
        else:
            x2 = False

        if w4 > 0 and h4 > 0:
            x4 = True
        else:
            x4 = False

        if w8 > 0 and h8 > 0:
            x8 = True
        else:
            x8 = False

        if w16 > 0 and h16 > 0:
            x16 = True
        else:
            x16 = False

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False

    bpf_array = bpf32_c(surface_, threshold_=threshold_)

    if x2:
        # downscale x 2 using fast scale pygame algorithm (no re-sampling)
        s2_array = array32_rescale_c(bpf_array, w2, h2)

        for r in range(smooth_):
            s2_array = blur5x5_array32_c(s2_array)

        b2_blurred = pygame.image.frombuffer(s2_array, (w2, h2), 'RGBA')
        s2 = smoothscale(b2_blurred, (w, h))
        surface_cp.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    if x4:
        # downscale x 4 using fast scale pygame algorithm (no re-sampling)
        s4_array = array32_rescale_c(bpf_array, w4, h4)

        for r in range(smooth_):
            s4_array = blur5x5_array32_c(s4_array)

        b4_blurred = pygame.image.frombuffer(s4_array, (w4, h4), 'RGBA')
        s4 = smoothscale(b4_blurred, (w, h))
        surface_cp.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    if x8:
        # downscale x 8 using fast scale pygame algorithm (no re-sampling)
        s8_array = array32_rescale_c(bpf_array, w8, h8)

        for r in range(smooth_):
            s8_array = blur5x5_array32_c(s8_array)

        b8_blurred = pygame.image.frombuffer(s8_array, (w8, h8), 'RGBA')
        s8 = smoothscale(b8_blurred, (w, h))
        surface_cp.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    if x16:
        # downscale x 16 using fast scale pygame algorithm (no re-sampling)
        s16_array = array32_rescale_c(bpf_array, w16, h16)

        for r in range(smooth_):
            s16_array = blur5x5_array32_c(s16_array)

        b16_blurred = pygame.image.frombuffer(s16_array, (w16, h16), 'RGBA')
        s16 = smoothscale(b16_blurred, (w, h))
        surface_cp.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    if mask_ is not None:
        # Multiply mask surface pixels with mask values.
        # RGB pixels = 0 when mask value = 0.0, otherwise
        # modify RGB amplitude
        filtering24_inplace_c(surface_cp, mask_)

    return surface_cp
