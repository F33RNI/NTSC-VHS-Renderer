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

cimport numpy as np

# GAUSSIAN BLUR KERNEL 5x5 COMPATIBLE 24-32 BIT SURFACE
cdef unsigned char [:, :, ::1] blur5x5_array24_c(unsigned char [:, :, :] rgb_array_, object mask=*)
cdef unsigned char [:, :, ::1] blur5x5_array32_c(unsigned char [:, :, :] rgba_array_, object mask=*)
cdef void blur5x5_array24_inplace_c(unsigned char [:, :, :] rgb_array_, object mask=*)
cdef void blur5x5_array32_inplace_c(unsigned char [:, :, :] rgba_array_, object mask=*)

# KERNEL
cdef np.ndarray[np.float64_t, ndim=2] kernel_deviation_c(double sigma, unsigned short int kernel_size)

# BRIGHT PASS FILTERS
cdef object bpf24_c(unsigned char [:, :, :] input_array_, unsigned char threshold = *)
cdef void bpf24_inplace_c(unsigned char [:, :, :] input_array_, unsigned char threshold = *)
cdef unsigned char [:, :, ::1] bpf32_c(image, unsigned char threshold_=*)
cdef void bpf32_inplace_c(object image_, unsigned char threshold_ = *)

# FILTERING
cdef void filtering24_inplace_c(object surface_, float [:, :] mask_)
cdef void filtering32_inplace_c(object surface_, float [:, :] mask_)

# RESCALING
cdef unsigned char [:, :, :] array32_rescale_c(unsigned char [:, :, :] rgb_array, int w2, int h2)

# BUILD MASK
cpdef np.ndarray[np.float32_t, ndim=2] build_mask_from_surface(object surface_, bint invert_mask = *)

# BLOOM EFFECT
cdef bloom_effect_array24_c(
        object surface_,
        unsigned char threshold_,
        unsigned short int smooth_  =*,
        object mask_                =*,
        bint fast_                  =*)

cdef bloom_effect_array32_c(
        object surface_,
        unsigned char threshold_,
        unsigned short int smooth_  =*,
        object mask_                =*,
        bint fast_                  =*)

cdef void bloom_effect_array24_inplace_c(
        object surface_,
        unsigned int threshold_,
        bint fast_ = *)

cdef void bloom_effect_array32_inplace_c(
        object surface_,
        unsigned int threshold_,
        bint fast_ = *)
