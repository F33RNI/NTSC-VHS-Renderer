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
    from numpy import asarray, uint8, float32, zeros, float64, empty

except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

# IndexMapping is require
try:
    import IndexMapping
except ImportError:
    raise ImportError(
        "\n<IndexMapping> library is missing on your system."
        "\nTry: \n   C:\\pip install IndexMapping on a window command prompt.")

try:
    from IndexMapping.mapping cimport xyz, to1d_c, vfb_rgb_c, vfb_c, vmap_buffer_c
except ImportError:
    raise ImportError("\n<IndexMapping> Cython library is missing on your system."
        "\nTry: \n   C:\\pip install IndexMapping on a window command prompt.")


from libc.math cimport fmin
from libc.stdio cimport printf
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


__version__ = "1.0.0"


cpdef scale_array24_mult(rgb_array):
    return scale_array24_mult_c(rgb_array)


# kernel 5x5 separable
cdef:
    float [5] KERNEL = \
        numpy.array(([ONE_SIXTEEN,
                      FOUR_SIXTEEN,
                      SIX_SIXTEEN,
                      FOUR_SIXTEEN,
                      ONE_SIXTEEN]), dtype=numpy.float32, copy=False)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline xyz to3d_c(unsigned int index, unsigned int width, unsigned short int depth)nogil:

    cdef:
        xyz v
        unsigned int ix = index // depth

    v.y = <int>(ix / width)
    v.x = <int>(ix % width)
    v.z = <int>(index % depth)
    return v


cpdef tuple blur5x5_buffer24(
        unsigned char [:] rgb_buffer,
        unsigned int width,
        unsigned int height,
        unsigned short int depth=3,
        unsigned char [:] mask=None):
    """
    APPLY A GAUSSIAN BLUR (5x5) TO A 1D ARRAY (containing tuple of RGB values) 

    Method using a C-buffer as input image (width * height * depth) uint8 data type
    5 x5 Gaussian kernel used:
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|

    It uses convolution property to process the image in two passes (horizontal and vertical passes).
    Pixels convoluted outside the image edges will be set to an adjacent pixel edge values

    :param depth        : integer; image depth (RGB = 3), default 3 (RGB)
    :param height       : integer; image height 
    :param width        : integer; image width
    :param rgb_buffer   : 1d buffer representing a 24bit format pygame.Surface  
    :param mask         : 1d numpy array for masking area of the image that will not be blurred, default None (NOT USED)
    :return             : 24-bit Pygame.Surface without per-pixel information and array and its equivalent array 
    """
    return blur5x5_buffer24_c(rgb_buffer, width, height, depth, mask)

cpdef blur5x5_buffer32(
        unsigned char [:] rgba_buffer,
        unsigned int width,
        unsigned int height,
        unsigned short int depth=4,
        unsigned char [:] mask=None):

    """
    APPLY A GAUSSIAN BLUR (5x5) TO A 1D ARRAY (containing tuple of RGBA values)

    Method using a C-buffer as input image (width * height * depth) uint8 data type
    5 x5 Gaussian kernel used:
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|

    It uses convolution property to process the image in two passes (horizontal and vertical passes).
    Pixels convoluted outside the image edges will be set to an adjacent pixel edge values

    :param depth        : integer; image depth (RGBA=4) , default 4
    :param height       : integer; image height
    :param width        : integer; image width
    :param rgba_buffer  : 1d buffer representing a 32bit format pygame.Surface  
    :param mask         : 1d numpy array for masking area of the image that will not be blurred, default None (NOT USED)
    :return             : 32-bit Pygame.Surface containing alpha values  
    """

    return blur5x5_buffer32_c(rgba_buffer, width, height, depth, mask)

cpdef bloom_effect_buffer24(
        object surface_,
        unsigned char threshold_,
        unsigned short int smooth_,
        object mask_=None,
        bint fast_=False):
    """
    CREATE A BLOOM EFFECT ON A PYGAME.SURFACE (COMPATIBLE 24 BIT SURFACE)
    THIS METHOD IS USING C-BUFFER STRUCTURE CONTAINING RGB PIXEL VALUES

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
    :param surface_  : pygame.surface 24 bit format surface
    :param threshold_: integer; threshold value used by the bright pass algorithm (default 128)
    :param smooth_   : integer; number of gaussian blur 5x5 to apply to downsized images.
    :return          : returns a pygame.surface with a bloom effect (24 bit surface)
    """
    return bloom_effect_buffer24_c(surface_, threshold_, smooth_, mask_=None, fast_=False)

cpdef bloom_effect_buffer32(
        object surface_,
        unsigned char threshold_,
        unsigned short int smooth_,
        object mask_=None,
        bint fast_=False):
    """
    CREATE A BLOOM EFFECT ON A PYGAME.SURFACE (COMPATIBLE 32 BIT SURFACE)
    THIS METHOD IS USING C-BUFFER STRUCTURE CONTAINING RGBA PIXEL VALUES

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
    :param surface_  : pygame.surface 32 bit format surface
    :param threshold_: integer; threshold value used by the bright pass algorithm (default 128)
    :param smooth_   : integer; number of gaussian blur 5x5 to apply to downsized images.
    :return          : returns a pygame.surface with a bloom effect (32 bit surface)
    """
    return bloom_effect_buffer32_c(surface_, threshold_, smooth_, mask_=None, fast_=False)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline blur5x5_buffer24_c(
        unsigned char [:] rgb_buffer,
        unsigned int width,
        unsigned int height,
        unsigned short int depth,
        object mask=None):

    cdef:
        int b_length = len(rgb_buffer)
        unsigned int length = width * height * depth

    # check if the buffer length equal theoretical length
    if b_length != length:
        raise ValueError("\nIncorrect 24-bit format image "
              "expecting %s bytes got %s " % (b_length, length))

    cdef:

        short int kernel_half = 2
        int xx, yy, index, i, ii
        float k, r, g, b
        char kernel_offset
        unsigned char red, green, blue
        xyz v;

        # convolve array contains pixels of the first pass(horizontal convolution)
        # convolved array contains pixels of the second pass.
        # buffer_ source pixels
        unsigned char [::1] convolve  = empty(length, numpy.uint8)
        unsigned char [::1] convolved = empty(length, numpy.uint8)
        unsigned char [:] buffer_     = rgb_buffer

    with nogil:
        # horizontal convolution
        # Goes through all RGB values and apply the horizontal convolution
        for i in prange(0, b_length, depth, schedule=METHOD, num_threads=THREADS):

            r, g, b = 0, 0, 0

            # v.x point to the row value of the equivalent 3d array
            # v.y point to the column value
            # v.z equal -> always point to the red value of the tuple RGB

            v = to3d_c(i, width, depth)

            # testing
            # index = to1d_c(v.x, v.y, v.z, width, 3)
            # print(v.x, v.y, v.z, i, index)

            for kernel_offset in range(-kernel_half, kernel_half + 1):

                k = KERNEL[kernel_offset + kernel_half]

                # Convert 1d indexing into a 3d indexing
                xx = v.x + kernel_offset

                # Avoid buffer overflow
                if xx < 0 or xx > (width - 1):
                    red, green, blue = 0, 0, 0

                else:
                    # Convert the 3d indexing into 1d buffer indexing
                    # The index value must always point to a red pixel
                    # v.z = 0
                    index = to1d_c(xx, v.y, v.z, width, depth)

                    # load the color value from the current pixel
                    red   = buffer_[index]
                    green = buffer_[index + 1]
                    blue  = buffer_[index + 2]

                r = r + red * k
                g = g + green * k
                b = b + blue * k

            # place the new RGB values into an empty array (convolve)
            convolve[i    ] = <unsigned char>r
            convolve[i + 1] = <unsigned char>g
            convolve[i + 2] = <unsigned char>b

        # Vertical convolution
        # In order to vertically convolve the kernel, we have to re-order the index value
        # to fetch data vertically with the vmap_buffer function.
        for i in prange(0, b_length, depth, schedule=METHOD, num_threads=THREADS):

                index = vmap_buffer_c(i, width, height, 3)

                r, g, b = 0, 0, 0

                v = to3d_c(index, width, depth)

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = KERNEL[kernel_offset + kernel_half]

                    yy = v.y + kernel_offset

                    if yy < 0 or yy > (height-1):

                        red, green, blue = 0, 0, 0
                    else:

                        ii = to1d_c(v.x, yy, v.z, width, depth)
                        red, green, blue = convolve[ii],\
                            convolve[ii+1], convolve[ii+2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[index    ] = <unsigned char>r
                convolved[index + 1] = <unsigned char>g
                convolved[index + 2] = <unsigned char>b

    return pygame.image.frombuffer(convolve, (width, height), "RGB"), convolve


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline blur5x5_buffer32_c(
        unsigned char [:] rgba_buffer,
        unsigned int width,
        unsigned int height,
        unsigned short int depth,
        object mask = None):

    cdef:
        int b_length = len(rgba_buffer)
        unsigned int length = width * height * depth
    # check if the buffer length equal theoretical length
    if b_length != length:
        raise ValueError(
            "\nIncorrect 32-bit format image, "
            "expecting %s got %s " % (length, b_length))

    cdef:
        short int kernel_half = 2
        int xx, yy, index, i, ii
        float k, r, g, b
        char kernel_offset
        unsigned char red, green, blue
        xyz v;

        # convolve array contains pixels of the first pass(horizontal convolution)
        # convolved array contains pixels of the second pass.
        # buffer_ source pixels
        unsigned char [:] convolve  = empty(length, numpy.uint8)
        unsigned char [:] convolved = empty(length, numpy.uint8)
        unsigned char [:] buffer_   = numpy.frombuffer(rgba_buffer, numpy.uint8)

    with nogil:
        # horizontal convolution
        # Goes through all RGB values of the buffer and apply the horizontal convolution
        for i in prange(0, b_length, depth, schedule=METHOD, num_threads=THREADS):

            r, g, b = 0, 0, 0

            # v.x point to the row value of the equivalent 3d array
            # v.y point to the column value
            # v.z = 0 always point to the red color of the tuple RGBA
            v = to3d_c(i, width, depth)

            # testing
            # index = to1d_c(v.x, v.y, v.z, width, 4)
            # print(v.x, v.y, v.z, i, index)

            for kernel_offset in range(-kernel_half, kernel_half + 1):

                k = KERNEL[kernel_offset + kernel_half]

                # Convert 1d indexing into a 3d indexing
                xx = v.x + kernel_offset

                # avoid buffer overflow
                if xx < 0 or xx > (width - 1):
                    red, green, blue = 0, 0, 0

                else:
                    # Convert the 3d indexing into 1d buffer indexing
                    # The index value must always point to a red pixel
                    # v.z = 0
                    index = to1d_c(xx, v.y, v.z, width, depth)

                    # load the color value from the current pixel
                    red   = buffer_[index    ]
                    green = buffer_[index + 1]
                    blue  = buffer_[index + 2]

                r = r + red * k
                g = g + green * k
                b = b + blue * k

            # place the new RGB values into an empty array (convolve)
            convolve[i    ] = <unsigned char>r
            convolve[i + 1] = <unsigned char>g
            convolve[i + 2] = <unsigned char>b
            convolve[i + 3] = buffer_[i + 3]

        # Vertical convolution
        # In order to vertically convolve the kernel, we have to re-order the index value
        # to fetch data vertically with the vmap_buffer function.
        for i in prange(0, b_length, depth, schedule=METHOD, num_threads=THREADS):

                index = vmap_buffer_c(i, width, height, depth)

                r, g, b = 0, 0, 0

                v = to3d_c(index, width, depth)

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = KERNEL[kernel_offset + kernel_half]

                    yy = v.y + kernel_offset

                    if yy < 0 or yy > (height-1):

                        red, green, blue = 0, 0, 0
                    else:

                        ii = to1d_c(v.x, yy, v.z, width, depth)
                        red, green, blue = convolve[ii],\
                            convolve[ii+1], convolve[ii+2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[index    ] = <unsigned char>r
                convolved[index + 1] = <unsigned char>g
                convolved[index + 2] = <unsigned char>b
                convolved[index + 3] = buffer_[index + 3]

    return pygame.image.frombuffer(convolved, (width, height), "RGBA"), convolved



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline bpf24_b_c(image, int threshold = 128, bint transpose=False):
    """
    Bright pass filter for 24bit image (method using c-buffer)

    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.

    :param transpose:boolean; True | False , transpose the final array / image. 
    :param image: pygame.Surface 24 bit format (RGB)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values,
    default is 128 range [0..255] unsigned char (python integer)
    :return: Return a 24 bit pygame.Surface filtered (only bright area of the image remains).
    """

    # Fallback to default threshold value if argument
    # threshold value is incorrect
    if 0 > threshold > 255:
        printf("\nArgument threshold must be in range [0...255], fallback to default value 128.")
        threshold = 128

    cdef:
        int w = image.get_width(), h = image.get_height()
        unsigned short int mode = 0;

    cdef int bitsize = image.get_bitsize()
    if image.get_flags() & pygame.SRCALPHA == pygame.SRCALPHA:
        raise ValueError('\nIncorrect image format, expecting 24-bit or 32-bit without per '
                         'pixel transparency got %s with per-pixel transparency' % bitsize)

    try:
        # BGR BUFFER
        # THIS IS THE FASTEST WAY TO GET THE BUFFER.
        # IF THE BUFFER IS NOT CONTIGUOUS, THIS WILL THROW
        # AN ERROR MESSAGE.
        #
        # buffer_ = numpy.asarray(im.get_view('2')).copy('C')
        # buffer_ = numpy.frombuffer(buffer_, dtype=numpy.uint8)

        buffer_ = image.get_view('2')
        buffer_ = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        mode = 1
    except:
        try:
            # RGB BUFFER
            # SLOWEST METHOD BUT WORKS EVEN IF THE BUFFER IS NOT
            # CONTIGUOUS
            buffer_ = tostring(image, 'RGB')
            buffer_ = numpy.frombuffer(buffer_, dtype=numpy.uint8).copy()   #.copy('C')
            mode = 0
        except:
            raise ValueError('\nInvalid surface.')

    # check sizes
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)

    cdef:
        # int b_length = buffer_.length
        int b_length = len(buffer_)
        unsigned char [:] c_buffer = buffer_
        unsigned char [:] out_buffer = numpy.empty(b_length, numpy.uint8)
        int i = 0, index =0, tmp
        float lum, c

    if transpose:
        # FINAL BUFFER IS TRANSPOSE USING METHOD vmap_buffer_c
        # IF ARRAY IS NOT SYMMETRIC ROWS AND COLUMNS ARE SWAPPED
        if w != h:
            tmp = w
            w = h
            h = tmp
        with nogil:
            for i in prange(0, b_length, 3, schedule=METHOD, num_threads=THREADS):
                # ITU-R BT.601 luma coefficients
                lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114

                index = vmap_buffer_c(i, w, h, depth=3)

                if lum > threshold:
                    c = (lum - threshold) / lum
                    if mode == 0:
                        # RGB
                        out_buffer[index    ] = <unsigned char>(c_buffer[i    ] * c)
                        out_buffer[index + 1] = <unsigned char>(c_buffer[i + 1] * c)
                        out_buffer[index + 2] = <unsigned char>(c_buffer[i + 2] * c)
                    else:
                        # BGR
                        out_buffer[index    ] = <unsigned char>(c_buffer[i + 2] * c)
                        out_buffer[index + 1] = <unsigned char>(c_buffer[i + 1] * c)
                        out_buffer[index + 2] = <unsigned char>(c_buffer[i    ] * c)
                else:
                    out_buffer[index], out_buffer[index + 1], out_buffer[index + 2] = 0, 0, 0

        return pygame.image.frombuffer(out_buffer, (w, h), 'RGB'), out_buffer

    else:
        with nogil:
            for i in prange(0, b_length, 3, schedule=METHOD, num_threads=THREADS):
                # ITU-R BT.601 luma coefficients
                lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114
                if lum > threshold:
                    c = (lum - threshold) / lum
                    if mode == 0:
                        # RGB
                        out_buffer[i    ] = <unsigned char>(c_buffer[i    ] * c)
                        out_buffer[i + 1] = <unsigned char>(c_buffer[i + 1] * c)
                        out_buffer[i + 2] = <unsigned char>(c_buffer[i + 2] * c)
                    else:
                        # BGR
                        out_buffer[i    ] = <unsigned char>(c_buffer[i + 2] * c)
                        out_buffer[i + 1] = <unsigned char>(c_buffer[i + 1] * c)
                        out_buffer[i + 2] = <unsigned char>(c_buffer[i    ] * c)
                else:
                    out_buffer[i], out_buffer[i + 1], out_buffer[i + 2] = 0, 0, 0

        return pygame.image.frombuffer(out_buffer,(w, h), 'RGB'), out_buffer


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline bpf32_b_c(image, int threshold = 128):
    """
    Bright pass filter for 32-bit image (method using c-buffer)

    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = cbuffer[i] * 0.299 + cbuffer[i+1] * 0.587 + cbuffer[i+2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.

    :param image: pygame.Surface 32 bit format (RGBA)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values
    :return: Return a 32-bit pygame.Surface filtered (only bright area of the image remains).
    """

    # Fallback to default threshold value if arguement
    # threshold value is incorrect
    if 0 > threshold > 255:
        printf("\nArgument threshold must be in range [0...255], fallback to default value 128.")
        threshold = 128

    assert isinstance(image, pygame.Surface), \
           "\nExpecting pygame surface for arguement image, got %s " % type(image)

    cdef:
        int w, h
    w, h = image.get_size()

    # make sure the surface is 32-bit format RGBA
    if not image.get_bitsize() == 32:
        raise ValueError('Surface is not 32-bit format.')

    try:
        # BGRA buffer
        buffer_ = image.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('\nInvalid surface.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, numpy.uint8)
        unsigned char [::1] out_buffer = numpy.empty(b_length, numpy.uint8)
        int i = 0
        float lum, c

    with nogil:
        for i in prange(0, b_length, 4, schedule=METHOD, num_threads=THREADS):
            # ITU-R BT.601 luma coefficients
            lum = cbuffer[i] * 0.299 + cbuffer[i+1] * 0.587 + cbuffer[i+2] * 0.114
            if lum > threshold:
                c = (lum - threshold) / lum
                # BGRA to RGBA
                out_buffer[i    ] = <unsigned char>(cbuffer[i + 2  ] * c)
                out_buffer[i + 1] = <unsigned char>(cbuffer[i + 1  ] * c)
                out_buffer[i + 2] = <unsigned char>(cbuffer[i      ] * c)
                out_buffer[i + 3] = 255
            else:
                out_buffer[i], out_buffer[i+1], \
                out_buffer[i+2], out_buffer[i+3] = 0, 0, 0, 0

    return pygame.image.frombuffer(out_buffer, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline filtering24_c(object surface_, mask_):
    """
    Multiply mask values with an array representing the surface pixels (Compatible 24 bit only).
    Mask values are floats in range (0 ... 1.0)

    :param surface_: pygame.Surface compatible 24-bit
    :param mask_: 2d array (MemoryViewSlice) containing alpha values (float).
    The mask_ output image is monochromatic (values range [0 ... 1.0] and R=B=G.
    :return: Return a pygame.Surface 24 bit
    """
    cdef int w, h, w_, h_
    w, h = surface_.get_size()
    try:
        w_, h_ = mask_.shape[:2]
    except (ValueError, pygame.error):
       raise ValueError(
           '\nArgument mask_ type not understood, '
           'expecting numpy.ndarray type (w, h) got %s ' % type(mask_))


    assert w == w_ and h == h_, \
        '\nSurface and mask size does not match (w:%s, h:%s), ' \
        '(w_:%s, h_:%s) ' % (w, h, w_, h_)

    try:
        rgb_ = pixels3d(surface_)
    except (ValueError, pygame.error):
        try:
            rgb_ = array3d(surface_)
        except (ValueError, pygame.error):
            raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb = rgb_.transpose(1, 0, 2)
        unsigned char [:, :, ::1] rgb1 = numpy.empty((h, w, 3), numpy.uint8)
        float [:, :] mask = numpy.asarray(mask_, numpy.float32)
        int i, j
    with nogil:
        for i in prange(0, w, schedule=METHOD, num_threads=THREADS):
            for j in range(h):
                rgb1[j, i, 0] = <unsigned char>(rgb[j, i, 0] * mask[i, j])
                rgb1[j, i, 1] = <unsigned char>(rgb[j, i, 1] * mask[i, j])
                rgb1[j, i, 2] = <unsigned char>(rgb[j, i, 2] * mask[i, j])

    return pygame.image.frombuffer(rgb1, (w, h), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline filtering32_c(surface_, mask_):
    """
    Multiply mask values with an array representing the surface pixels (Compatible 32 bit only).
    Mask values are floats in range (0 ... 1.0)

    :param surface_: pygame.Surface compatible 32-bit
    :param mask_: 2d array (MemoryViewSlice) containing alpha values (float).
    The mask_ output image is monochromatic (values range [0 ... 1.0] and R=B=G.
    :return: Return a pygame.Surface 32-bit
    """

    cdef int w, h, w_, h_
    w, h = surface_.get_size()

    try:
        w_, h_ = (<object>mask_).shape[:2]
    except (ValueError, pygame.error):
        raise ValueError(
            '\nArgument mask_ type not understood, expecting numpy.ndarray got %s ' % type(mask_))

    assert w == w_ and h == h_, 'Surface and mask size does not match.'

    try:
        rgb_ = pixels3d(surface_)
    except (ValueError, pygame.error):
        try:
            rgb_ = array3d(surface_)
        except (ValueError, pygame.error):
            raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb = rgb_
        unsigned char [:, :, ::1] rgb1 = numpy.empty((h, w, 4), numpy.uint8)
        float [:, :] mask = numpy.asarray(mask_, numpy.float32)
        int i, j
    with nogil:
        for i in prange(0, w, schedule=METHOD, num_threads=THREADS):
            for j in range(h):
                rgb1[j, i, 0] = <unsigned char>(rgb[i, j, 0] * mask[i, j])
                rgb1[j, i, 1] = <unsigned char>(rgb[i, j, 1] * mask[i, j])
                rgb1[j, i, 2] = <unsigned char>(rgb[i, j, 2] * mask[i, j])
                rgb1[j, i, 3] = <unsigned char>(mask[i, j] * 255.0)

    return pygame.image.frombuffer(rgb1, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline bloom_effect_buffer24_c(
        object surface_,
        unsigned char threshold_,
        unsigned short int smooth_=1,
        object mask_=None,
        bint fast_=False):


    surface_cp = surface_

    assert smooth_ > 0, \
           "\nArgument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
           "\nArgument threshold_ must be in range [0...255] got %s " % threshold_

    cdef:
        int w, h, bitsize
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bitsize = surface_.get_bitsize()

    if surface_.get_flags() & pygame.SRCALPHA == pygame.SRCALPHA:
        raise ValueError('\nIncorrect image format, expecting 24-bit or 32-bit without per '
                         'pixel transparency got %s with per-pixel transparency' % bitsize)

    with nogil:
        w2, h2   = w >> 1, h >> 1
        w4, h4   = w2 >> 1, h2 >> 1
        w8, h8   = w4 >> 1, h4 >> 1
        w16, h16 = w8 >> 1, h8 >> 1

    with nogil:
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

    # check if the first reduction is > 0
    # if not we cannot blur that surface (too small)
    if not x2:
        return pygame.Surface((1, 1))

    if fast_:

        x2, x4, x8 = False, False, False

        s4 = scale(surface_, (w >> 2, h >> 2))
        bpf_surface, bpf_array = bpf24_b_c(surface_, threshold=threshold_, transpose=False)
        bpf_surface = scale(bpf_surface, (w, h))

    else:
        # BRIGHT PASS FILTER
        bpf_surface, bpf_array = bpf24_b_c(surface_, threshold=threshold_, transpose=False)

    if x2:

        s2 = scale(bpf_surface, (w2, h2))
        b2 = tostring(s2, 'RGB')
        b2 = numpy.frombuffer(b2, dtype=numpy.uint8).copy()
        if smooth_ > 1:
            for r in range(smooth_):
                b2_blurred, b2 = blur5x5_buffer24_c(b2, w2, h2, 3)
        else:
            b2_blurred, b2 = blur5x5_buffer24_c(b2, w2, h2, 3)
        s2 = smoothscale(b2_blurred, (w, h))
        surface_cp.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    if x4:
        s4 = scale(bpf_surface, (w4, h4))
        b4 = tostring(s4, 'RGB')
        b4 = numpy.frombuffer(b4, dtype=numpy.uint8).copy()
        if smooth_ > 1:
            for r in range(smooth_):
                b4_blurred, b4 = blur5x5_buffer24_c(b4, w4, h4, 3)
        else:
            b4_blurred, b4 = blur5x5_buffer24_c(b4, w4, h4, 3)
        s4 = smoothscale(b4_blurred, (w, h))
        surface_cp.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    if x8:
        s8 = scale(bpf_surface, (w8, h8))
        b8 = tostring(s8, 'RGB')
        b8 = numpy.frombuffer(b8, dtype=numpy.uint8).copy()
        if smooth_ > 1:
            for r in range(smooth_):
                b8_blurred, b8 = blur5x5_buffer24_c(b8, w8, h8, 3)
        else:
            b8_blurred, b8 = blur5x5_buffer24_c(b8, w8, h8, 3)
        s8 = smoothscale(b8_blurred, (w, h))
        surface_cp.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    if x16:
        s16 = scale(bpf_surface, (w16, h16))
        b16 = tostring(s16, 'RGB')
        b16 = numpy.frombuffer(b16, dtype=numpy.uint8).copy()
        if smooth_ > 1:
            for r in range(smooth_):
                b16_blurred, b16 = blur5x5_buffer24_c(b16, w16, h16, 3)
        else:
            b16_blurred, b16 = blur5x5_buffer24_c(b16, w16, h16, 3)
        s16 = smoothscale(b16_blurred, (w, h))
        surface_cp.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    if mask_ is not None:
        # Multiply mask surface pixels with mask values.
        # RGB pixels = 0 when mask value = 0.0, otherwise
        # modify RGB amplitude
        surface_cp = filtering24_c(surface_cp, mask_)
    return surface_cp



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline bloom_effect_buffer32_c(
        object surface_,
        unsigned char threshold_,
        unsigned short int smooth_=1,
        object mask_=None,
        bint fast_=False):

    surface_cp = surface_.copy()

    assert smooth_ > 0, \
           "\nArgument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
           "\nArgument threshold_ must be in range [0...255] got %s " % threshold_

    cdef:
        int w, h, bitsize
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bitsize = surface_.get_bitsize()

    if not (surface_.get_flags() & pygame.SRCALPHA == pygame.SRCALPHA):
        raise ValueError('\nIncorrect image format, expecting 32-bit got %s without per-pixel transparency' % bitsize)

    with nogil:
        w2, h2 = w >> 1, h >> 1
        w4, h4 = w2 >> 1, h2 >> 1
        w8, h8 = w4 >> 1, h4 >> 1
        w16, h16 = w8 >> 1, h8 >> 1

    with nogil:
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

    # check if the first reduction is > 0
    # if not we cannot blur that surface (too small)
    if not x2:
        return pygame.Surface((1, 1))

    if fast_:

        x2, x4, x8 = False, False, False

        s4 = scale(surface_, (w >> 2, h >> 2))
        bpf_surface =  bpf32_b_c(surface_, threshold=threshold_)
        bpf_surface = scale(bpf_surface, (w, h))

    else:
        # BRIGHT PASS FILTER
        bpf_surface =  bpf32_b_c(surface_, threshold=threshold_)


    if x2:
        s2 = scale(bpf_surface, (w2, h2))
        b2 = numpy.frombuffer(s2.get_view("2"), numpy.uint8)
        if smooth_ > 1:
            for r in range(smooth_):
                b2_blurred, b2 = blur5x5_buffer32_c(b2, w2, h2, 4)#, mask_)
        else:
            b2_blurred, b2 = blur5x5_buffer32_c(b2, w2, h2, 4)#, mask_)
        s2 = smoothscale(b2_blurred, (w, h))
        surface_cp.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    if x4:
        # downscale x 4 using fast scale pygame algorithm (no re-sampling)
        s4 = scale(bpf_surface, (w4, h4))
        b4 = numpy.frombuffer(s4.get_view("2"), numpy.uint8)
        if smooth_ > 1:
            for r in range(smooth_):
                b4_blurred, b4 = blur5x5_buffer32_c(b4, w4, h4, 4)#, mask_)
        else:
            b4_blurred, b4 = blur5x5_buffer32_c(b4, w4, h4, 4)#, mask_)
        s4 = smoothscale(b4_blurred, (w, h))
        surface_cp.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    if x8:
        # downscale x 8 using fast scale pygame algorithm (no re-sampling)
        s8 = scale(bpf_surface, (w8, h8))
        b8 = numpy.frombuffer(s8.get_view("2"), numpy.uint8)
        if smooth_ > 1:
            for r in range(smooth_):
                b8_blurred, b8 = blur5x5_buffer32_c(b8, w8, h8, 4)#, mask_)
        else:
            b8_blurred, b8 = blur5x5_buffer32_c(b8, w8, h8, 4)#, mask_)
        s8 = smoothscale(b8_blurred, (w, h))
        surface_cp.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    if x16:
        # downscale x 16 using fast scale pygame algorithm (no re-sampling)
        s16 = scale(bpf_surface, (w16, h16))
        b16 = numpy.frombuffer(s16.get_view("2"), numpy.uint8)
        if smooth_ > 1:
            for r in range(smooth_):
                b16_blurred, b16 = blur5x5_buffer32_c(b16, w16, h16, 4)#, mask_)
        else:
            b16_blurred, b16 = blur5x5_buffer32_c(b16, w16, h16, 4)#, mask_)
        s16 = smoothscale(b16_blurred, (w, h))
        surface_cp.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    if mask_ is not None:
        # Multiply mask surface pixels with mask values.
        # RGB pixels = 0 when mask value = 0.0, otherwise
        # modify RGB amplitude
        surface_cp = filtering32_c(surface_cp.convert_alpha(), mask_)

    return surface_cp




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scale_array24_mult_c(unsigned char [:, :, :] rgb_array):
    """
    MULTIPLE DOWNSCALING 
    DOWNSCALE/RESIZE SURFACE/ARRAY BY FACTOR 2, 4, 8, 16
    :param rgb_array: 3D array representing the surface type(width, height, 3) with unsigned char  
    :return: Return MEMORYVIEWSLICE; Returns input image downscale factor 2, 4, 8, 16 
    """

    cdef:
        int w1, h1, s

    try:
        w1, h1, s = (<object>rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        int w2  = <int>(w1 * ONE_HALF),   h2 = <int>(h1 * ONE_HALF)          # div 2
        int w4  = <int>(w1 * ONE_FOURTH), h4 = <int>(h1 * ONE_FOURTH)        # div 4
        int w8  = <int>(w1 * ONE_EIGHTH), h8 = <int>(h1 * ONE_EIGHTH)        # div 8
        int w16 = <int>(w1 * ONE_SIXTEENTH), h16 = <int>(h1 * ONE_SIXTEENTH) # div 16

    cdef:
        unsigned char [:, :, ::1] new_array_div2  = numpy.empty((w2, h2, 3), numpy.uint8)
        unsigned char [:, :, ::1] new_array_div4  = numpy.empty((w4, h4, 3), numpy.uint8)
        unsigned char [:, :, ::1] new_array_div8  = numpy.empty((w8, h8, 3), numpy.uint8)
        unsigned char [:, :, ::1] new_array_div16 = numpy.empty((w16, h16, 3), numpy.uint8)
        int x, y, xx2, xx4, xx8, xx16, yy2
        int r, g, b

    # TODO TEST WITH MAX AND MIN (INTEGER)
    with nogil:
        for x in prange(0, w1, schedule=METHOD, num_threads=THREADS):
            xx2  = <int>fmin(x * ONE_HALF,   w2-1)
            xx4  = <int>fmin(x * ONE_FOURTH, w4-1)
            xx8  = <int>fmin(x * ONE_EIGHTH, w8-1)
            xx16 = <int>fmin(x * ONE_SIXTEENTH, w16-1)
            for y in range(0, h1):

                yy2 = <int>fmin(y * ONE_HALF, h2-1)
                r, g, b = rgb_array[x, y, 0], rgb_array[x, y, 1], rgb_array[x, y, 2]
                new_array_div2[xx2, yy2, 0] = r
                new_array_div2[xx2, yy2, 1] = g
                new_array_div2[xx2, yy2, 2] = b

                yy2 = <int>fmin(y * ONE_FOURTH, h4-1)
                new_array_div4[xx4, yy2, 0] = r
                new_array_div4[xx4, yy2, 1] = g
                new_array_div4[xx4, yy2, 2] = b

                yy2 = <int>fmin(y * ONE_EIGHTH, h8-1)
                new_array_div8[xx8, yy2, 0] = r
                new_array_div8[xx8, yy2, 1] = g
                new_array_div8[xx8, yy2, 2] = b

                yy2 = <int>fmin(y * ONE_SIXTEENTH, h16-1)
                new_array_div16[xx16, yy2, 0] = r
                new_array_div16[xx16, yy2, 1] = g
                new_array_div16[xx16, yy2, 2] = b
    return new_array_div2, new_array_div4, new_array_div8, new_array_div16


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scale_alpha24_mult_c(unsigned char [:, :] alpha_array):
    """
    MULTIPLE DOWNSCALING 
    DOWNSCALE/RESIZE SURFACE/ARRAY BY FACTOR 2, 4, 8, 16
    :param alpha_array: 2D alpha array type(width, height) 
    :return: Return the input alpha array downscale factor 2, 4, 8, 16 (MEMORYVIEWSLICE)
    """
    cdef:
        int w1, h1

    try:
        w1, h1 = (<object>alpha_array).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        int w2  = <int>(w1 * ONE_HALF),   h2 = <int>(h1 * ONE_HALF)          # div 2
        int w4  = <int>(w1 * ONE_FOURTH), h4 = <int>(h1 * ONE_FOURTH)        # div 4
        int w8  = <int>(w1 * ONE_EIGHTH), h8 = <int>(h1 * ONE_EIGHTH)        # div 8
        int w16 = <int>(w1 * ONE_SIXTEENTH), h16 = <int>(h1 * ONE_SIXTEENTH) # div 16

    cdef:
        unsigned char [:, ::1] new_array_div2  = numpy.empty((w2, h2), numpy.uint8)
        unsigned char [:, ::1] new_array_div4  = numpy.empty((w4, h4), numpy.uint8)
        unsigned char [:, ::1] new_array_div8  = numpy.empty((w8, h8), numpy.uint8)
        unsigned char [:, ::1] new_array_div16 = numpy.empty((w16, h16), numpy.uint8)
        int x, y, xx2, xx4, xx8, xx16, yy2

    # TODO TEST WITH MAX AND MIN (INTEGER)
    with nogil:
        for x in prange(0, w1, schedule=METHOD, num_threads=THREADS):
            xx2  = <int>fmin(x * ONE_HALF,   w2-1)
            xx4  = <int>fmin(x * ONE_FOURTH, w4-1)
            xx8  = <int>fmin(x * ONE_EIGHTH, w8-1)
            xx16 = <int>fmin(x * ONE_SIXTEENTH, w16-1)
            for y in range(0, h1):

                yy2 = <int>fmin(y * ONE_HALF, h2-1)
                new_array_div2[xx2, yy2] = alpha_array[x, y]

                yy2 = <int>fmin(y * ONE_FOURTH, h4-1)
                new_array_div4[xx4, yy2] = alpha_array[x, y]

                yy2 = <int>fmin(y * ONE_EIGHTH, h8-1)
                new_array_div8[xx8, yy2] = alpha_array[x, y]

                yy2 = <int>fmin(y * ONE_SIXTEENTH, h16-1)
                new_array_div16[xx16, yy2] = alpha_array[x, y]

    return new_array_div2, new_array_div4, new_array_div8, new_array_div16



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scale_alpha24_single_c(unsigned char [:, :] alpha_array, int w2, int h2):
    """
    RESCALE ALPHA ARRAY TYPE (W, H) TO DIMENSIONS (W2, H2)
    :param alpha_array: 2D Alpha array to rescale to w2, h2 dimensions 
    :param w2: final width
    :param h2: final height
    :return: Alpha array rescale to (w2, h2)
    """
    cdef:
        int w1, h1

    try:
        w1, h1 = (<object>alpha_array).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')


    cdef:
        unsigned char [:, ::1] new_array_div2  = numpy.empty((w2, h2), numpy.uint8)
        int x, y, xx, yy
        float fx = <float>w1 / <float>w2
        float fy = <float>h1 / <float>h2
    with nogil:
        for x in prange(0, w2, schedule=METHOD, num_threads=THREADS):
            xx = <int>(x * fx)
            for y in range(0, h2):
                yy = <int>(y * fy)
                new_array_div2[x, y] = alpha_array[xx, yy]

    return new_array_div2



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] scale_array24_c(unsigned char [:, :, :] rgb_array, int w2, int h2):
    """
    Rescale a 24-bit format image from its given array 
    The final array is equivalent to the input array re-scale and transposed.

    :param rgb_array: RGB numpy.ndarray, format (w, h, 3) numpy.uint8
    :param w2: new width 
    :param h2: new height
    :return: Return a MemoryViewSlice 3d numpy.ndarray format (w, h, 3) uint8
    """

    cdef int w1, h1, s
    try:
        w1, h1, s = (<object>rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char [:, :, ::1] new_array = numpy.zeros((h2, w2, 3), numpy.uint8)
        float fx = <float>w1 / <float>w2
        float fy = <float>h1 / <float>h2
        int x, y, xx, yy
    with nogil:
        for x in prange(w2, schedule=METHOD, num_threads=THREADS):
            xx = <int>(x * fx)
            for y in range(h2):
                yy = <int>(y * fy)
                new_array[x, y, 0] = rgb_array[xx, yy, 0]
                new_array[x, y, 1] = rgb_array[xx, yy, 1]
                new_array[x, y, 2] = rgb_array[xx, yy, 2]

    return new_array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] scale_array32_c(unsigned char [:, :, :] rgb_array, int w2, int h2):
    """
    Rescale a 32-bit format image from its given array 
    The final array is equivalent to the input array re-scale and transposed.

    :param rgb_array: RGB numpy.ndarray, format (w, h, 4) numpy.uint8 with alpha channel
    :param w2: new width 
    :param h2: new height
    :return: Return a MemoryViewSlice 3d numpy.ndarray format (w, h, 4) uint8
    """

    cdef int w1, h1, s
    try:
        w1, h1, s = (<object>rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char [:, :, ::1] new_array = numpy.zeros((h2, w2, 4), numpy.uint8)
        float fx = <float>w1 / <float>w2
        float fy = <float>h1 / <float>h2
        int x, y, xx, yy
    with nogil:
        for x in prange(w2, schedule=METHOD, num_threads=THREADS):
            xx = <int>(x * fx)
            for y in range(h2):
                yy = <int>(y * fy)
                new_array[y, x, 0] = rgb_array[xx, yy, 0]
                new_array[y, x, 1] = rgb_array[xx, yy, 1]
                new_array[y, x, 2] = rgb_array[xx, yy, 2]
                new_array[y, x, 3] = rgb_array[xx, yy, 3]

    return new_array
