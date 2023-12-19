
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused_0 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15,
                       float* out_ptr16,
                       float* out_ptr17)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(3L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(49L); i2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(i2 + (49L*i1) + (147L*i0))];
                    out_ptr0[static_cast<long>(i1 + (3L*i2) + (147L*i0))] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (576L*i0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(i1 + (64L*i2) + (576L*i0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr1[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (576L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(i1 + (64L*i2) + (576L*i0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (576L*i0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(i1 + (64L*i2) + (576L*i0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr2[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (576L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(i1 + (64L*i2) + (576L*i0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (576L*i0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(i1 + (64L*i2) + (576L*i0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr3[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (576L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr3 + static_cast<long>(i1 + (64L*i2) + (576L*i0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (1152L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(i1 + (128L*i2) + (1152L*i0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr4[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (1152L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(i1 + (128L*i2) + (1152L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (1152L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(i1 + (128L*i2) + (1152L*i0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr5[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (1152L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr5 + static_cast<long>(i1 + (128L*i2) + (1152L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (1152L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(i1 + (128L*i2) + (1152L*i0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr6[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (1152L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(i1 + (128L*i2) + (1152L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (1152L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(i1 + (128L*i2) + (1152L*i0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr7[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (1152L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr7 + static_cast<long>(i1 + (128L*i2) + (1152L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr8[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr8 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr9 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr9[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr9 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr10 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr10[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr10 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr11 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr11[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr11 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr12 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr12[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr12 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr13 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr13[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (2304L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr13 + static_cast<long>(i1 + (256L*i2) + (2304L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (4608L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr14 + static_cast<long>(i1 + (512L*i2) + (4608L*i0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr14[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (4608L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr14 + static_cast<long>(i1 + (512L*i2) + (4608L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (4608L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr15 + static_cast<long>(i1 + (512L*i2) + (4608L*i0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr15[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (4608L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr15 + static_cast<long>(i1 + (512L*i2) + (4608L*i0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(8L); i2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long i1_inner = 0; i1_inner < 8; i1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (4608L*i0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*i1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr16 + static_cast<long>(i1 + (512L*i2) + (4608L*i0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(8L); i2<static_cast<long>(9L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long i1_inner = 0; i1_inner < 8; i1_inner++) tmpbuf[i1_inner] = in_ptr16[static_cast<long>(i2 + (9L*i1) + (9L*i1_inner) + (4608L*i0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr16 + static_cast<long>(i1 + (512L*i2) + (4608L*i0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(3L); i0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long i1=static_cast<long>(0L); i1<static_cast<long>(4096L); i1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr17[static_cast<long>(i1 + (4096L*i0))];
                        out_ptr17[static_cast<long>(i0 + (3L*i1))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       long* out_ptr7)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long i1=static_cast<long>(0L); i1<static_cast<long>(1024L); i1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (64L*i1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(i0)];
                    auto tmp10 = in_ptr1[static_cast<long>(i0)];
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0009775171065494);
                    auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr2[static_cast<long>(i0)] = tmp5;
                    out_ptr3[static_cast<long>(i0)] = tmp13;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = tmp2 + tmp5;
                    tmp6.store(out_ptr4 + static_cast<long>(i0));
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(i1 + (64L*i0))];
                    auto tmp1 = out_ptr0[static_cast<long>(i1)];
                    auto tmp3 = out_ptr1[static_cast<long>(i1)];
                    auto tmp10 = in_ptr3[static_cast<long>(i1)];
                    auto tmp12 = in_ptr4[static_cast<long>(i1)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1024.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[static_cast<long>(i1 + (64L*i0))] = tmp14;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long i2=static_cast<long>(0L); i2<static_cast<long>(64L); i2+=static_cast<long>(1L))
                        {
                            auto tmp0 = static_cast<long>((-1L) + (2L*i0));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = static_cast<long>((-1L) + (2L*i1));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr5[static_cast<long>((-2112L) + i2 + (128L*i1) + (4096L*i0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : -std::numeric_limits<decltype(tmp11())>::infinity();
                            auto tmp14 = static_cast<long>(2L*i1);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = out_ptr5[static_cast<long>((-2048L) + i2 + (128L*i1) + (4096L*i0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : -std::numeric_limits<decltype(tmp19())>::infinity();
                            auto tmp22 = max_propagate_nan(tmp21, tmp13);
                            auto tmp23 = static_cast<long>(1L + (2L*i1));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = out_ptr5[static_cast<long>((-1984L) + i2 + (128L*i1) + (4096L*i0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : -std::numeric_limits<decltype(tmp28())>::infinity();
                            auto tmp31 = max_propagate_nan(tmp30, tmp22);
                            auto tmp32 = static_cast<long>(2L*i0);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = out_ptr5[static_cast<long>((-64L) + i2 + (128L*i1) + (4096L*i0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                            auto tmp40 = max_propagate_nan(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = out_ptr5[static_cast<long>(i2 + (128L*i1) + (4096L*i0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = out_ptr5[static_cast<long>(64L + i2 + (128L*i1) + (4096L*i0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = max_propagate_nan(tmp49, tmp45);
                            auto tmp51 = static_cast<long>(1L + (2L*i0));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = out_ptr5[static_cast<long>(1984L + i2 + (128L*i1) + (4096L*i0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                            auto tmp59 = max_propagate_nan(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = out_ptr5[static_cast<long>(2048L + i2 + (128L*i1) + (4096L*i0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                            auto tmp64 = max_propagate_nan(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = out_ptr5[static_cast<long>(2112L + i2 + (128L*i1) + (4096L*i0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                            auto tmp69 = max_propagate_nan(tmp68, tmp64);
                            auto tmp70 = [&]
                            {
                                auto tmp71 = out_ptr5[static_cast<long>((-2112L) + i2 + (128L*i1) + (4096L*i0))];
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp10 ? tmp70() : -std::numeric_limits<decltype(tmp70())>::infinity();
                            auto tmp73 = [&]
                            {
                                auto tmp74 = out_ptr5[static_cast<long>((-2048L) + i2 + (128L*i1) + (4096L*i0))];
                                return tmp74;
                            }
                            ;
                            auto tmp75 = tmp18 ? tmp73() : -std::numeric_limits<decltype(tmp73())>::infinity();
                            auto tmp76 = tmp75 > tmp72;
                            auto tmp77 = static_cast<long>((-32L) + (2L*i1) + (64L*i0));
                            auto tmp78 = static_cast<long>((-33L) + (2L*i1) + (64L*i0));
                            auto tmp79 = tmp76 ? tmp77 : tmp78;
                            auto tmp80 = max_propagate_nan(tmp75, tmp72);
                            auto tmp81 = [&]
                            {
                                auto tmp82 = out_ptr5[static_cast<long>((-1984L) + i2 + (128L*i1) + (4096L*i0))];
                                return tmp82;
                            }
                            ;
                            auto tmp83 = tmp27 ? tmp81() : -std::numeric_limits<decltype(tmp81())>::infinity();
                            auto tmp84 = tmp83 > tmp80;
                            auto tmp85 = static_cast<long>((-31L) + (2L*i1) + (64L*i0));
                            auto tmp86 = tmp84 ? tmp85 : tmp79;
                            auto tmp87 = max_propagate_nan(tmp83, tmp80);
                            auto tmp88 = [&]
                            {
                                auto tmp89 = out_ptr5[static_cast<long>((-64L) + i2 + (128L*i1) + (4096L*i0))];
                                return tmp89;
                            }
                            ;
                            auto tmp90 = tmp36 ? tmp88() : -std::numeric_limits<decltype(tmp88())>::infinity();
                            auto tmp91 = tmp90 > tmp87;
                            auto tmp92 = static_cast<long>((-1L) + (2L*i1) + (64L*i0));
                            auto tmp93 = tmp91 ? tmp92 : tmp86;
                            auto tmp94 = max_propagate_nan(tmp90, tmp87);
                            auto tmp95 = [&]
                            {
                                auto tmp96 = out_ptr5[static_cast<long>(i2 + (128L*i1) + (4096L*i0))];
                                return tmp96;
                            }
                            ;
                            auto tmp97 = tmp41 ? tmp95() : -std::numeric_limits<decltype(tmp95())>::infinity();
                            auto tmp98 = tmp97 > tmp94;
                            auto tmp99 = static_cast<long>((2L*i1) + (64L*i0));
                            auto tmp100 = tmp98 ? tmp99 : tmp93;
                            auto tmp101 = max_propagate_nan(tmp97, tmp94);
                            auto tmp102 = [&]
                            {
                                auto tmp103 = out_ptr5[static_cast<long>(64L + i2 + (128L*i1) + (4096L*i0))];
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp46 ? tmp102() : -std::numeric_limits<decltype(tmp102())>::infinity();
                            auto tmp105 = tmp104 > tmp101;
                            auto tmp106 = static_cast<long>(1L + (2L*i1) + (64L*i0));
                            auto tmp107 = tmp105 ? tmp106 : tmp100;
                            auto tmp108 = max_propagate_nan(tmp104, tmp101);
                            auto tmp109 = [&]
                            {
                                auto tmp110 = out_ptr5[static_cast<long>(1984L + i2 + (128L*i1) + (4096L*i0))];
                                return tmp110;
                            }
                            ;
                            auto tmp111 = tmp55 ? tmp109() : -std::numeric_limits<decltype(tmp109())>::infinity();
                            auto tmp112 = tmp111 > tmp108;
                            auto tmp113 = static_cast<long>(31L + (2L*i1) + (64L*i0));
                            auto tmp114 = tmp112 ? tmp113 : tmp107;
                            auto tmp115 = max_propagate_nan(tmp111, tmp108);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = out_ptr5[static_cast<long>(2048L + i2 + (128L*i1) + (4096L*i0))];
                                return tmp117;
                            }
                            ;
                            auto tmp118 = tmp60 ? tmp116() : -std::numeric_limits<decltype(tmp116())>::infinity();
                            auto tmp119 = tmp118 > tmp115;
                            auto tmp120 = static_cast<long>(32L + (2L*i1) + (64L*i0));
                            auto tmp121 = tmp119 ? tmp120 : tmp114;
                            auto tmp122 = max_propagate_nan(tmp118, tmp115);
                            auto tmp123 = [&]
                            {
                                auto tmp124 = out_ptr5[static_cast<long>(2112L + i2 + (128L*i1) + (4096L*i0))];
                                return tmp124;
                            }
                            ;
                            auto tmp125 = tmp65 ? tmp123() : -std::numeric_limits<decltype(tmp123())>::infinity();
                            auto tmp126 = tmp125 > tmp122;
                            auto tmp127 = static_cast<long>(33L + (2L*i1) + (64L*i0));
                            auto tmp128 = tmp126 ? tmp127 : tmp121;
                            auto tmp129 = max_propagate_nan(tmp125, tmp122);
                            out_ptr6[static_cast<long>(i2 + (64L*i1) + (1024L*i0))] = tmp69;
                            out_ptr7[static_cast<long>(i2 + (64L*i1) + (1024L*i0))] = tmp128;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (64L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.003921568627451);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (64L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(256.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (64L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (64L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.003921568627451);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (64L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(256.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (64L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_4 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(i0)];
                    auto tmp10 = in_ptr1[static_cast<long>(i0)];
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.003921568627451);
                    auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr2[static_cast<long>(i0)] = tmp5;
                    out_ptr3[static_cast<long>(i0)] = tmp13;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = tmp2 + tmp5;
                    tmp6.store(out_ptr4 + static_cast<long>(i0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(i0)];
                    auto tmp10 = in_ptr1[static_cast<long>(i0)];
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.003921568627451);
                    auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr2[static_cast<long>(i0)] = tmp5;
                    out_ptr3[static_cast<long>(i0)] = tmp13;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = tmp2 + tmp5;
                    tmp6.store(out_ptr4 + static_cast<long>(i0));
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(i1 + (256L*i0))];
                    auto tmp1 = in_ptr4[static_cast<long>(i1)];
                    auto tmp3 = in_ptr5[static_cast<long>(i1)];
                    auto tmp10 = in_ptr6[static_cast<long>(i1)];
                    auto tmp12 = in_ptr7[static_cast<long>(i1)];
                    auto tmp14 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                    auto tmp15 = out_ptr0[static_cast<long>(i1)];
                    auto tmp17 = out_ptr1[static_cast<long>(i1)];
                    auto tmp22 = in_ptr8[static_cast<long>(i1)];
                    auto tmp24 = in_ptr9[static_cast<long>(i1)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp16 = tmp14 - tmp15;
                    auto tmp18 = tmp17 / tmp4;
                    auto tmp19 = tmp18 + tmp6;
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = decltype(tmp16)(tmp16 * tmp20);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp25 = tmp23 + tmp24;
                    auto tmp26 = tmp13 + tmp25;
                    out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp26;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(65536L); i0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(i0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(i0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (64L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.003921568627451);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (64L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(256.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (64L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (64L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.003921568627451);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (64L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(256.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (64L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(i0)];
                    auto tmp10 = in_ptr1[static_cast<long>(i0)];
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.003921568627451);
                    auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr2[static_cast<long>(i0)] = tmp5;
                    out_ptr3[static_cast<long>(i0)] = tmp13;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = tmp2 + tmp5;
                    tmp6.store(out_ptr4 + static_cast<long>(i0));
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                    auto tmp1 = out_ptr0[static_cast<long>(i1)];
                    auto tmp3 = out_ptr1[static_cast<long>(i1)];
                    auto tmp10 = in_ptr3[static_cast<long>(i1)];
                    auto tmp12 = in_ptr4[static_cast<long>(i1)];
                    auto tmp14 = in_ptr5[static_cast<long>(i1 + (256L*i0))];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = tmp15 * (tmp15>0);
                    out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (64L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.003921568627451);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (64L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(256.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (64L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (64L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.003921568627451);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (64L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(256.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (64L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(i0)];
                    auto tmp10 = in_ptr1[static_cast<long>(i0)];
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.003921568627451);
                    auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr2[static_cast<long>(i0)] = tmp5;
                    out_ptr3[static_cast<long>(i0)] = tmp13;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = tmp2 + tmp5;
                    tmp6.store(out_ptr4 + static_cast<long>(i0));
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                    auto tmp1 = out_ptr0[static_cast<long>(i1)];
                    auto tmp3 = out_ptr1[static_cast<long>(i1)];
                    auto tmp10 = in_ptr3[static_cast<long>(i1)];
                    auto tmp12 = in_ptr4[static_cast<long>(i1)];
                    auto tmp14 = in_ptr5[static_cast<long>(i1 + (256L*i0))];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = tmp15 * (tmp15>0);
                    out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (128L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.003921568627451);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (128L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(256.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (128L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (128L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (128L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (128L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_14 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (512L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (512L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr3[static_cast<long>(i1 + (512L*i0))];
                auto tmp1 = in_ptr4[static_cast<long>(i1)];
                auto tmp3 = in_ptr5[static_cast<long>(i1)];
                auto tmp10 = in_ptr6[static_cast<long>(i1)];
                auto tmp12 = in_ptr7[static_cast<long>(i1)];
                auto tmp14 = in_ptr0[static_cast<long>(i1 + (512L*i0))];
                auto tmp15 = out_ptr0[static_cast<long>(i1)];
                auto tmp17 = out_ptr1[static_cast<long>(i1)];
                auto tmp22 = in_ptr8[static_cast<long>(i1)];
                auto tmp24 = in_ptr9[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp16 = tmp14 - tmp15;
                auto tmp18 = tmp17 / tmp4;
                auto tmp19 = tmp18 + tmp6;
                auto tmp20 = 1 / std::sqrt(tmp19);
                auto tmp21 = decltype(tmp16)(tmp16 * tmp20);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp25 = tmp23 + tmp24;
                auto tmp26 = tmp13 + tmp25;
                out_ptr5[static_cast<long>(i1 + (512L*i0))] = tmp26;
            }
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(32768L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(i0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(i0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (128L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (128L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (128L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (128L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (128L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (128L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (512L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (512L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp14 = in_ptr5[static_cast<long>(i1 + (512L*i0))];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr5[static_cast<long>(i1 + (512L*i0))] = tmp16;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (128L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (128L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (128L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (128L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (128L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (128L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (512L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (512L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp14 = in_ptr5[static_cast<long>(i1 + (512L*i0))];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr5[static_cast<long>(i1 + (512L*i0))] = tmp16;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (128L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (128L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (128L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (128L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(128L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(128L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (128L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (128L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (512L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (512L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp14 = in_ptr5[static_cast<long>(i1 + (512L*i0))];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr5[static_cast<long>(i1 + (512L*i0))] = tmp16;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(64L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(64.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0158730158730158);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(64L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(64.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_27 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (1024L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (1024L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(1024L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr3[static_cast<long>(i1 + (1024L*i0))];
                auto tmp1 = in_ptr4[static_cast<long>(i1)];
                auto tmp3 = in_ptr5[static_cast<long>(i1)];
                auto tmp10 = in_ptr6[static_cast<long>(i1)];
                auto tmp12 = in_ptr7[static_cast<long>(i1)];
                auto tmp14 = in_ptr0[static_cast<long>(i1 + (1024L*i0))];
                auto tmp15 = out_ptr0[static_cast<long>(i1)];
                auto tmp17 = out_ptr1[static_cast<long>(i1)];
                auto tmp22 = in_ptr8[static_cast<long>(i1)];
                auto tmp24 = in_ptr9[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp16 = tmp14 - tmp15;
                auto tmp18 = tmp17 / tmp4;
                auto tmp19 = tmp18 + tmp6;
                auto tmp20 = 1 / std::sqrt(tmp19);
                auto tmp21 = decltype(tmp16)(tmp16 * tmp20);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp25 = tmp23 + tmp24;
                auto tmp26 = tmp13 + tmp25;
                out_ptr5[static_cast<long>(i1 + (1024L*i0))] = tmp26;
            }
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16384L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(i0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(i0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (1024L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(1024L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (1024L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp14 = in_ptr5[static_cast<long>(i1 + (1024L*i0))];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr5[static_cast<long>(i1 + (1024L*i0))] = tmp16;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (1024L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(1024L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (1024L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp14 = in_ptr5[static_cast<long>(i1 + (1024L*i0))];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr5[static_cast<long>(i1 + (1024L*i0))] = tmp16;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (1024L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(1024L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (1024L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp14 = in_ptr5[static_cast<long>(i1 + (1024L*i0))];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr5[static_cast<long>(i1 + (1024L*i0))] = tmp16;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (1024L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(1024L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (1024L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp14 = in_ptr5[static_cast<long>(i1 + (1024L*i0))];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr5[static_cast<long>(i1 + (1024L*i0))] = tmp16;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (256L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(256L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(256L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (256L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (256L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (1024L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1024L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(1024L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (1024L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp14 = in_ptr5[static_cast<long>(i1 + (1024L*i0))];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr5[static_cast<long>(i1 + (1024L*i0))] = tmp16;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(16L); i1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (512L*i1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(i0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(i0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(i0)];
            auto tmp10 = in_ptr1[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0666666666666667);
            auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = tmp9 + tmp12;
            out_ptr2[static_cast<long>(i0)] = tmp5;
            out_ptr3[static_cast<long>(i0)] = tmp13;
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp5 = tmp3 * tmp4;
            auto tmp6 = tmp2 + tmp5;
            tmp6.store(out_ptr4 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (512L*i0))];
                auto tmp1 = out_ptr0[static_cast<long>(i1)];
                auto tmp3 = out_ptr1[static_cast<long>(i1)];
                auto tmp10 = in_ptr3[static_cast<long>(i1)];
                auto tmp12 = in_ptr4[static_cast<long>(i1)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr5[static_cast<long>(i1 + (512L*i0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + i0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            auto tmp9 = tmp0 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            auto tmp11 = tmp1 - tmp8;
            auto tmp12 = tmp11 * tmp11;
            auto tmp13 = tmp10 + tmp12;
            auto tmp14 = tmp3 - tmp8;
            auto tmp15 = tmp14 * tmp14;
            auto tmp16 = tmp13 + tmp15;
            auto tmp17 = tmp5 - tmp8;
            auto tmp18 = tmp17 * tmp17;
            auto tmp19 = tmp16 + tmp18;
            auto tmp20 = tmp19 / tmp7;
            auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp22 = tmp8 * tmp21;
            auto tmp24 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp25 = tmp23 * tmp24;
            auto tmp26 = tmp22 + tmp25;
            auto tmp27 = at::vec::Vectorized<float>(static_cast<float>(1.3333333333333333));
            auto tmp28 = tmp20 * tmp27;
            auto tmp29 = tmp28 * tmp21;
            auto tmp31 = tmp30 * tmp24;
            auto tmp32 = tmp29 + tmp31;
            tmp8.store(out_ptr0 + static_cast<long>(i0));
            tmp20.store(out_ptr1 + static_cast<long>(i0));
            tmp26.store(out_ptr2 + static_cast<long>(i0));
            tmp32.store(out_ptr3 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
        {
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (512L*i0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(i1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(i1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp5.rsqrt();
                auto tmp7 = tmp2 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = at::vec::clamp_min(tmp11, decltype(tmp11)(0));
                tmp12.store(out_ptr4 + static_cast<long>(i1 + (512L*i0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_46 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(2048L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2048L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(4096L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(6144L + i0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            auto tmp9 = tmp0 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            auto tmp11 = tmp1 - tmp8;
            auto tmp12 = tmp11 * tmp11;
            auto tmp13 = tmp10 + tmp12;
            auto tmp14 = tmp3 - tmp8;
            auto tmp15 = tmp14 * tmp14;
            auto tmp16 = tmp13 + tmp15;
            auto tmp17 = tmp5 - tmp8;
            auto tmp18 = tmp17 * tmp17;
            auto tmp19 = tmp16 + tmp18;
            auto tmp20 = tmp19 / tmp7;
            auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp22 = tmp8 * tmp21;
            auto tmp24 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp25 = tmp23 * tmp24;
            auto tmp26 = tmp22 + tmp25;
            auto tmp27 = at::vec::Vectorized<float>(static_cast<float>(1.3333333333333333));
            auto tmp28 = tmp20 * tmp27;
            auto tmp29 = tmp28 * tmp21;
            auto tmp31 = tmp30 * tmp24;
            auto tmp32 = tmp29 + tmp31;
            tmp8.store(out_ptr0 + static_cast<long>(i0));
            tmp20.store(out_ptr1 + static_cast<long>(i0));
            tmp26.store(out_ptr2 + static_cast<long>(i0));
            tmp32.store(out_ptr3 + static_cast<long>(i0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(2048L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2048L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(4096L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(6144L + i0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            auto tmp9 = tmp0 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            auto tmp11 = tmp1 - tmp8;
            auto tmp12 = tmp11 * tmp11;
            auto tmp13 = tmp10 + tmp12;
            auto tmp14 = tmp3 - tmp8;
            auto tmp15 = tmp14 * tmp14;
            auto tmp16 = tmp13 + tmp15;
            auto tmp17 = tmp5 - tmp8;
            auto tmp18 = tmp17 * tmp17;
            auto tmp19 = tmp16 + tmp18;
            auto tmp20 = tmp19 / tmp7;
            auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp22 = tmp8 * tmp21;
            auto tmp24 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp25 = tmp23 * tmp24;
            auto tmp26 = tmp22 + tmp25;
            auto tmp27 = at::vec::Vectorized<float>(static_cast<float>(1.3333333333333333));
            auto tmp28 = tmp20 * tmp27;
            auto tmp29 = tmp28 * tmp21;
            auto tmp31 = tmp30 * tmp24;
            auto tmp32 = tmp29 + tmp31;
            tmp8.store(out_ptr0 + static_cast<long>(i0));
            tmp20.store(out_ptr1 + static_cast<long>(i0));
            tmp26.store(out_ptr2 + static_cast<long>(i0));
            tmp32.store(out_ptr3 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
        {
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(2048L); i1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1 + (2048L*i0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(i1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(i1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(i1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(i1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (2048L*i0)));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(i1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(i1));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(i1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp5.rsqrt();
                auto tmp7 = tmp2 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp14 = tmp12 - tmp13;
                auto tmp16 = tmp15 + tmp4;
                auto tmp17 = tmp16.rsqrt();
                auto tmp18 = tmp14 * tmp17;
                auto tmp20 = tmp18 * tmp19;
                auto tmp22 = tmp20 + tmp21;
                auto tmp23 = tmp11 + tmp22;
                tmp23.store(out_ptr4 + static_cast<long>(i1 + (2048L*i0)));
            }
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(8192L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(i0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(i0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + i0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            auto tmp9 = tmp0 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            auto tmp11 = tmp1 - tmp8;
            auto tmp12 = tmp11 * tmp11;
            auto tmp13 = tmp10 + tmp12;
            auto tmp14 = tmp3 - tmp8;
            auto tmp15 = tmp14 * tmp14;
            auto tmp16 = tmp13 + tmp15;
            auto tmp17 = tmp5 - tmp8;
            auto tmp18 = tmp17 * tmp17;
            auto tmp19 = tmp16 + tmp18;
            auto tmp20 = tmp19 / tmp7;
            auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp22 = tmp8 * tmp21;
            auto tmp24 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp25 = tmp23 * tmp24;
            auto tmp26 = tmp22 + tmp25;
            auto tmp27 = at::vec::Vectorized<float>(static_cast<float>(1.3333333333333333));
            auto tmp28 = tmp20 * tmp27;
            auto tmp29 = tmp28 * tmp21;
            auto tmp31 = tmp30 * tmp24;
            auto tmp32 = tmp29 + tmp31;
            tmp8.store(out_ptr0 + static_cast<long>(i0));
            tmp20.store(out_ptr1 + static_cast<long>(i0));
            tmp26.store(out_ptr2 + static_cast<long>(i0));
            tmp32.store(out_ptr3 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
        {
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (512L*i0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(i1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(i1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp5.rsqrt();
                auto tmp7 = tmp2 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = at::vec::clamp_min(tmp11, decltype(tmp11)(0));
                tmp12.store(out_ptr4 + static_cast<long>(i1 + (512L*i0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + i0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            auto tmp9 = tmp0 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            auto tmp11 = tmp1 - tmp8;
            auto tmp12 = tmp11 * tmp11;
            auto tmp13 = tmp10 + tmp12;
            auto tmp14 = tmp3 - tmp8;
            auto tmp15 = tmp14 * tmp14;
            auto tmp16 = tmp13 + tmp15;
            auto tmp17 = tmp5 - tmp8;
            auto tmp18 = tmp17 * tmp17;
            auto tmp19 = tmp16 + tmp18;
            auto tmp20 = tmp19 / tmp7;
            auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp22 = tmp8 * tmp21;
            auto tmp24 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp25 = tmp23 * tmp24;
            auto tmp26 = tmp22 + tmp25;
            auto tmp27 = at::vec::Vectorized<float>(static_cast<float>(1.3333333333333333));
            auto tmp28 = tmp20 * tmp27;
            auto tmp29 = tmp28 * tmp21;
            auto tmp31 = tmp30 * tmp24;
            auto tmp32 = tmp29 + tmp31;
            tmp8.store(out_ptr0 + static_cast<long>(i0));
            tmp20.store(out_ptr1 + static_cast<long>(i0));
            tmp26.store(out_ptr2 + static_cast<long>(i0));
            tmp32.store(out_ptr3 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
        {
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (512L*i0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(i1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(i1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp5.rsqrt();
                auto tmp7 = tmp2 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = at::vec::clamp_min(tmp11, decltype(tmp11)(0));
                tmp12.store(out_ptr4 + static_cast<long>(i1 + (512L*i0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(2048L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2048L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(4096L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(6144L + i0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            auto tmp9 = tmp0 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            auto tmp11 = tmp1 - tmp8;
            auto tmp12 = tmp11 * tmp11;
            auto tmp13 = tmp10 + tmp12;
            auto tmp14 = tmp3 - tmp8;
            auto tmp15 = tmp14 * tmp14;
            auto tmp16 = tmp13 + tmp15;
            auto tmp17 = tmp5 - tmp8;
            auto tmp18 = tmp17 * tmp17;
            auto tmp19 = tmp16 + tmp18;
            auto tmp20 = tmp19 / tmp7;
            auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp22 = tmp8 * tmp21;
            auto tmp24 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp25 = tmp23 * tmp24;
            auto tmp26 = tmp22 + tmp25;
            auto tmp27 = at::vec::Vectorized<float>(static_cast<float>(1.3333333333333333));
            auto tmp28 = tmp20 * tmp27;
            auto tmp29 = tmp28 * tmp21;
            auto tmp31 = tmp30 * tmp24;
            auto tmp32 = tmp29 + tmp31;
            tmp8.store(out_ptr0 + static_cast<long>(i0));
            tmp20.store(out_ptr1 + static_cast<long>(i0));
            tmp26.store(out_ptr2 + static_cast<long>(i0));
            tmp32.store(out_ptr3 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
        {
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(2048L); i1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (2048L*i0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(i1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(i1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(i1 + (2048L*i0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp5.rsqrt();
                auto tmp7 = tmp2 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(out_ptr4 + static_cast<long>(i1 + (2048L*i0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_51 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + i0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            auto tmp9 = tmp0 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            auto tmp11 = tmp1 - tmp8;
            auto tmp12 = tmp11 * tmp11;
            auto tmp13 = tmp10 + tmp12;
            auto tmp14 = tmp3 - tmp8;
            auto tmp15 = tmp14 * tmp14;
            auto tmp16 = tmp13 + tmp15;
            auto tmp17 = tmp5 - tmp8;
            auto tmp18 = tmp17 * tmp17;
            auto tmp19 = tmp16 + tmp18;
            auto tmp20 = tmp19 / tmp7;
            auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp22 = tmp8 * tmp21;
            auto tmp24 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp25 = tmp23 * tmp24;
            auto tmp26 = tmp22 + tmp25;
            auto tmp27 = at::vec::Vectorized<float>(static_cast<float>(1.3333333333333333));
            auto tmp28 = tmp20 * tmp27;
            auto tmp29 = tmp28 * tmp21;
            auto tmp31 = tmp30 * tmp24;
            auto tmp32 = tmp29 + tmp31;
            tmp8.store(out_ptr0 + static_cast<long>(i0));
            tmp20.store(out_ptr1 + static_cast<long>(i0));
            tmp26.store(out_ptr2 + static_cast<long>(i0));
            tmp32.store(out_ptr3 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
        {
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (512L*i0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(i1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(i1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp5.rsqrt();
                auto tmp7 = tmp2 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = at::vec::clamp_min(tmp11, decltype(tmp11)(0));
                tmp12.store(out_ptr4 + static_cast<long>(i1 + (512L*i0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(512L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + i0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            auto tmp9 = tmp0 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            auto tmp11 = tmp1 - tmp8;
            auto tmp12 = tmp11 * tmp11;
            auto tmp13 = tmp10 + tmp12;
            auto tmp14 = tmp3 - tmp8;
            auto tmp15 = tmp14 * tmp14;
            auto tmp16 = tmp13 + tmp15;
            auto tmp17 = tmp5 - tmp8;
            auto tmp18 = tmp17 * tmp17;
            auto tmp19 = tmp16 + tmp18;
            auto tmp20 = tmp19 / tmp7;
            auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp22 = tmp8 * tmp21;
            auto tmp24 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp25 = tmp23 * tmp24;
            auto tmp26 = tmp22 + tmp25;
            auto tmp27 = at::vec::Vectorized<float>(static_cast<float>(1.3333333333333333));
            auto tmp28 = tmp20 * tmp27;
            auto tmp29 = tmp28 * tmp21;
            auto tmp31 = tmp30 * tmp24;
            auto tmp32 = tmp29 + tmp31;
            tmp8.store(out_ptr0 + static_cast<long>(i0));
            tmp20.store(out_ptr1 + static_cast<long>(i0));
            tmp26.store(out_ptr2 + static_cast<long>(i0));
            tmp32.store(out_ptr3 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
        {
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (512L*i0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(i1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(i1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp5.rsqrt();
                auto tmp7 = tmp2 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = at::vec::clamp_min(tmp11, decltype(tmp11)(0));
                tmp12.store(out_ptr4 + static_cast<long>(i1 + (512L*i0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_mean_relu_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(2048L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2048L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(4096L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(6144L + i0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            auto tmp9 = tmp0 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            auto tmp11 = tmp1 - tmp8;
            auto tmp12 = tmp11 * tmp11;
            auto tmp13 = tmp10 + tmp12;
            auto tmp14 = tmp3 - tmp8;
            auto tmp15 = tmp14 * tmp14;
            auto tmp16 = tmp13 + tmp15;
            auto tmp17 = tmp5 - tmp8;
            auto tmp18 = tmp17 * tmp17;
            auto tmp19 = tmp16 + tmp18;
            auto tmp20 = tmp19 / tmp7;
            auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp22 = tmp8 * tmp21;
            auto tmp24 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp25 = tmp23 * tmp24;
            auto tmp26 = tmp22 + tmp25;
            auto tmp27 = at::vec::Vectorized<float>(static_cast<float>(1.3333333333333333));
            auto tmp28 = tmp20 * tmp27;
            auto tmp29 = tmp28 * tmp21;
            auto tmp31 = tmp30 * tmp24;
            auto tmp32 = tmp29 + tmp31;
            tmp8.store(out_ptr0 + static_cast<long>(i0));
            tmp20.store(out_ptr1 + static_cast<long>(i0));
            tmp26.store(out_ptr2 + static_cast<long>(i0));
            tmp32.store(out_ptr3 + static_cast<long>(i0));
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
        {
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(2048L); i1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (2048L*i0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(i1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(i1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(i1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(i1 + (2048L*i0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp5.rsqrt();
                auto tmp7 = tmp2 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(out_ptr4 + static_cast<long>(i1 + (2048L*i0)));
            }
        }
    }
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(2048L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(2048L + i0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(4096L + i0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(6144L + i0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(4.0));
            auto tmp8 = tmp6 / tmp7;
            tmp8.store(out_ptr5 + static_cast<long>(i0));
        }
    }
}
''')


cpp_fused_add_threshold_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_fjr38/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const long* in_ptr1,
                       const long* in_ptr2,
                       const long* in_ptr3,
                       const long* in_ptr4,
                       const long* in_ptr5,
                       const long* in_ptr6,
                       const long* in_ptr7,
                       const long* in_ptr8,
                       const long* in_ptr9,
                       const long* in_ptr10,
                       const long* in_ptr11,
                       const long* in_ptr12,
                       const long* in_ptr13,
                       const long* in_ptr14,
                       const long* in_ptr15,
                       const long* in_ptr16,
                       const long* in_ptr17,
                       const long* in_ptr18,
                       const long* in_ptr19,
                       const long* in_ptr20,
                       const long* in_ptr21,
                       const long* in_ptr22,
                       const long* in_ptr23,
                       const long* in_ptr24,
                       const long* in_ptr25,
                       const long* in_ptr26,
                       const long* in_ptr27,
                       const long* in_ptr28,
                       const long* in_ptr29,
                       const long* in_ptr30,
                       const long* in_ptr31,
                       const long* in_ptr32,
                       const long* in_ptr33,
                       const long* in_ptr34,
                       const long* in_ptr35,
                       const long* in_ptr36,
                       const long* in_ptr37,
                       const long* in_ptr38,
                       const long* in_ptr39,
                       const long* in_ptr40,
                       const long* in_ptr41,
                       const long* in_ptr42,
                       const long* in_ptr43,
                       const long* in_ptr44,
                       const long* in_ptr45,
                       const long* in_ptr46,
                       const long* in_ptr47,
                       const long* in_ptr48,
                       const long* in_ptr49,
                       const long* in_ptr50,
                       const long* in_ptr51,
                       const long* in_ptr52,
                       const long* in_ptr53,
                       bool* out_ptr0,
                       long* out_ptr1,
                       long* out_ptr2,
                       long* out_ptr3,
                       long* out_ptr4,
                       long* out_ptr5,
                       long* out_ptr6,
                       long* out_ptr7,
                       long* out_ptr8,
                       long* out_ptr9,
                       long* out_ptr10,
                       long* out_ptr11,
                       long* out_ptr12,
                       long* out_ptr13,
                       long* out_ptr14,
                       long* out_ptr15,
                       long* out_ptr16,
                       long* out_ptr17,
                       long* out_ptr18,
                       long* out_ptr19,
                       long* out_ptr20,
                       long* out_ptr21,
                       long* out_ptr22,
                       long* out_ptr23,
                       long* out_ptr24,
                       long* out_ptr25,
                       long* out_ptr26,
                       long* out_ptr27,
                       long* out_ptr28,
                       long* out_ptr29,
                       long* out_ptr30,
                       long* out_ptr31,
                       long* out_ptr32,
                       long* out_ptr33,
                       long* out_ptr34,
                       long* out_ptr35,
                       long* out_ptr36,
                       long* out_ptr37,
                       long* out_ptr38,
                       long* out_ptr39,
                       long* out_ptr40,
                       long* out_ptr41,
                       long* out_ptr42,
                       long* out_ptr43,
                       long* out_ptr44,
                       long* out_ptr45,
                       long* out_ptr46,
                       long* out_ptr47,
                       long* out_ptr48,
                       long* out_ptr49,
                       long* out_ptr50,
                       long* out_ptr51,
                       long* out_ptr52,
                       long* out_ptr53)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(8192L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = tmp0 <= tmp1;
            out_ptr0[static_cast<long>(i0)] = tmp2;
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr1[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr2[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr2[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr3[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr3[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr4[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr4[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr5[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr5[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr6[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr6[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr7[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr7[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr8[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr8[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr9[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr9[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr10[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr10[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr11[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr11[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr12[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr12[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr13[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr13[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr14[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr14[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr15[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr15[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr16[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr16[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr17[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr17[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr18[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr18[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr19[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr19[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr20[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr20[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr21[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr21[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr22[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr22[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr23[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr23[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr24[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr24[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr25[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr25[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr26[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr26[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr27[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr27[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr28[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr28[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr29[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr29[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr30[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr30[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr31[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr31[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr32[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr32[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr33[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr33[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr34[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr34[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr35[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr35[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr36[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr36[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr37[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr37[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr38[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr38[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr39[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr39[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr40[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr40[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr41[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr41[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr42[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr42[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr43[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr43[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr44[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr44[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr45[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr45[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr46[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr46[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr47[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr47[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr48[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr48[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr49[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr49[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr50[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr50[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr51[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr51[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr52[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr52[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr53[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr53[static_cast<long>(0L)] = tmp2;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_81, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_85, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1024, ), (1, ))
    assert_size_stride(primals_94, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_102, (1024, ), (1, ))
    assert_size_stride(primals_103, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_120, (1024, ), (1, ))
    assert_size_stride(primals_121, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (1024, ), (1, ))
    assert_size_stride(primals_130, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (512, ), (1, ))
    assert_size_stride(primals_136, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_138, (2048, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (2048, ), (1, ))
    assert_size_stride(primals_142, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (512, ), (1, ))
    assert_size_stride(primals_148, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_150, (2048, ), (1, ))
    assert_size_stride(primals_151, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_153, (512, ), (1, ))
    assert_size_stride(primals_154, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (512, ), (1, ))
    assert_size_stride(primals_157, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_159, (2048, ), (1, ))
    assert_size_stride(primals_160, (1000, 2048), (2048, 1))
    assert_size_stride(primals_161, (1000, ), (1, ))
    assert_size_stride(primals_162, (64, ), (1, ))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (), ())
    assert_size_stride(primals_165, (64, ), (1, ))
    assert_size_stride(primals_166, (64, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (64, ), (1, ))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (64, ), (1, ))
    assert_size_stride(primals_178, (64, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (64, ), (1, ))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (64, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (128, ), (1, ))
    assert_size_stride(primals_199, (128, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (512, ), (1, ))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (128, ), (1, ))
    assert_size_stride(primals_208, (128, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (128, ), (1, ))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (128, ), (1, ))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (128, ), (1, ))
    assert_size_stride(primals_220, (128, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (128, ), (1, ))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (256, ), (1, ))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (256, ), (1, ))
    assert_size_stride(primals_247, (256, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (256, ), (1, ))
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (1024, ), (1, ))
    assert_size_stride(primals_253, (1024, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (256, ), (1, ))
    assert_size_stride(primals_256, (256, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (256, ), (1, ))
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (1024, ), (1, ))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_263, (), ())
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (256, ), (1, ))
    assert_size_stride(primals_266, (), ())
    assert_size_stride(primals_267, (256, ), (1, ))
    assert_size_stride(primals_268, (256, ), (1, ))
    assert_size_stride(primals_269, (), ())
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_272, (), ())
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_274, (256, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (256, ), (1, ))
    assert_size_stride(primals_277, (256, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (256, ), (1, ))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (512, ), (1, ))
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (2048, ), (1, ))
    assert_size_stride(primals_298, (2048, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (2048, ), (1, ))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (512, ), (1, ))
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (512, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (2048, ), (1, ))
    assert_size_stride(primals_310, (2048, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (512, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (512, ), (1, ))
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (2048, ), (1, ))
    assert_size_stride(primals_319, (2048, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (1, 3, 64, 64), (12288, 4096, 64, 1))
    buf0 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((1, 3, 64, 64), (12288, 1, 192, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del primals_1
    del primals_106
    del primals_115
    del primals_124
    del primals_133
    del primals_145
    del primals_154
    del primals_19
    del primals_28
    del primals_321
    del primals_37
    del primals_49
    del primals_58
    del primals_67
    del primals_7
    del primals_76
    del primals_88
    del primals_97
    # Source Nodes: [l__self___conv1], Original ATen: [aten.convolution]
    buf18 = extern_kernels.convolution(buf17, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf18, (1, 64, 32, 32), (65536, 1, 2048, 64))
    buf19 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((1, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    buf26 = empty_strided((1, 64, 16, 16), (16384, 1, 1024, 64), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((1, 64, 16, 16), (16384, 1, 1024, 64), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_functional_max_pool2d_with_indices_relu_1(c_void_p(buf18.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    del primals_162
    del primals_163
    del primals_3
    # Source Nodes: [getattr_l__self___layer1___0___conv1], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(buf26, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf28, (1, 64, 16, 16), (16384, 1, 1024, 64))
    buf29 = buf20; del buf20  # reuse
    buf30 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf32 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf35 = empty_strided((1, 64, 16, 16), (16384, 1, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_2(c_void_p(buf28.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf35.data_ptr()))
    del primals_165
    del primals_166
    del primals_6
    # Source Nodes: [getattr_l__self___layer1___0___conv2], Original ATen: [aten.convolution]
    buf36 = extern_kernels.convolution(buf35, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf36, (1, 64, 16, 16), (16384, 1, 1024, 64))
    buf37 = buf30; del buf30  # reuse
    buf38 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf40 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf42 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((1, 64, 16, 16), (16384, 1, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_3(c_void_p(buf36.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()))
    del primals_168
    del primals_169
    del primals_9
    # Source Nodes: [getattr_l__self___layer1___0___conv3], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (1, 256, 16, 16), (65536, 1, 4096, 256))
    buf45 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf46 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf50 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf49 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_4(c_void_p(buf44.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf49.data_ptr()))
    del primals_171
    del primals_172
    # Source Nodes: [getattr_l__self___layer1___0___downsample_0], Original ATen: [aten.convolution]
    buf51 = extern_kernels.convolution(buf26, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf51, (1, 256, 16, 16), (65536, 1, 4096, 256))
    buf52 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf53 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf57 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf56 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((1, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    buf59 = buf58; del buf58  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_relu_5(c_void_p(buf59.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf56.data_ptr()))
    del primals_12
    del primals_15
    del primals_174
    del primals_175
    # Source Nodes: [getattr_l__self___layer1___1___conv1], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf59, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (1, 64, 16, 16), (16384, 1, 1024, 64))
    buf61 = buf38; del buf38  # reuse
    buf62 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf64 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf66 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf65 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((1, 64, 16, 16), (16384, 1, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_6(c_void_p(buf60.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()))
    del primals_177
    del primals_178
    del primals_18
    # Source Nodes: [getattr_l__self___layer1___1___conv2], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (1, 64, 16, 16), (16384, 1, 1024, 64))
    buf69 = buf62; del buf62  # reuse
    buf70 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf72 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf74 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf73 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((1, 64, 16, 16), (16384, 1, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_7(c_void_p(buf68.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()))
    del primals_180
    del primals_181
    del primals_21
    # Source Nodes: [getattr_l__self___layer1___1___conv3], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf76, (1, 256, 16, 16), (65536, 1, 4096, 256))
    buf77 = buf53; del buf53  # reuse
    buf78 = buf46; del buf46  # reuse
    buf80 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf82 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf81 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf83 = empty_strided((1, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_8(c_void_p(buf76.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()))
    del primals_183
    del primals_184
    del primals_24
    # Source Nodes: [getattr_l__self___layer1___2___conv1], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(buf83, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (1, 64, 16, 16), (16384, 1, 1024, 64))
    buf85 = buf70; del buf70  # reuse
    buf86 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf88 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf90 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf89 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf91 = empty_strided((1, 64, 16, 16), (16384, 1, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_9(c_void_p(buf84.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()))
    del primals_186
    del primals_187
    del primals_27
    # Source Nodes: [getattr_l__self___layer1___2___conv2], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf92, (1, 64, 16, 16), (16384, 1, 1024, 64))
    buf93 = buf86; del buf86  # reuse
    buf94 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf96 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf98 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf99 = empty_strided((1, 64, 16, 16), (16384, 1, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_10(c_void_p(buf92.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()))
    del buf94
    del primals_189
    del primals_190
    del primals_30
    # Source Nodes: [getattr_l__self___layer1___2___conv3], Original ATen: [aten.convolution]
    buf100 = extern_kernels.convolution(buf99, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf100, (1, 256, 16, 16), (65536, 1, 4096, 256))
    buf101 = buf78; del buf78  # reuse
    buf102 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf104 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf106 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf105 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf107 = empty_strided((1, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_11(c_void_p(buf100.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()))
    del primals_192
    del primals_193
    del primals_33
    # Source Nodes: [getattr_l__self___layer2___0___conv1], Original ATen: [aten.convolution]
    buf108 = extern_kernels.convolution(buf107, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf108, (1, 128, 16, 16), (32768, 1, 2048, 128))
    buf109 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf110 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf112 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf114 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf113 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf115 = empty_strided((1, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_12(c_void_p(buf108.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()))
    del primals_195
    del primals_196
    del primals_36
    # Source Nodes: [getattr_l__self___layer2___0___conv2], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(buf115, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf116, (1, 128, 8, 8), (8192, 1, 1024, 128))
    buf117 = buf110; del buf110  # reuse
    buf118 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf120 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf122 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf121 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf123 = empty_strided((1, 128, 8, 8), (8192, 1, 1024, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_13(c_void_p(buf116.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf123.data_ptr()))
    del primals_198
    del primals_199
    del primals_39
    # Source Nodes: [getattr_l__self___layer2___0___conv3], Original ATen: [aten.convolution]
    buf124 = extern_kernels.convolution(buf123, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (1, 512, 8, 8), (32768, 1, 4096, 512))
    buf125 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf126 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf128 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf130 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf129 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_14(c_void_p(buf124.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf129.data_ptr()))
    del primals_201
    del primals_202
    # Source Nodes: [getattr_l__self___layer2___0___downsample_0], Original ATen: [aten.convolution]
    buf131 = extern_kernels.convolution(buf107, primals_43, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf131, (1, 512, 8, 8), (32768, 1, 4096, 512))
    buf132 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf135 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf137 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf136 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf138 = empty_strided((1, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    buf139 = buf138; del buf138  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_relu_15(c_void_p(buf139.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf136.data_ptr()))
    del primals_204
    del primals_205
    del primals_42
    del primals_45
    # Source Nodes: [getattr_l__self___layer2___1___conv1], Original ATen: [aten.convolution]
    buf140 = extern_kernels.convolution(buf139, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf140, (1, 128, 8, 8), (8192, 1, 1024, 128))
    buf141 = buf118; del buf118  # reuse
    buf142 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf144 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf146 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf147 = empty_strided((1, 128, 8, 8), (8192, 1, 1024, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_16(c_void_p(buf140.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()))
    del primals_207
    del primals_208
    del primals_48
    # Source Nodes: [getattr_l__self___layer2___1___conv2], Original ATen: [aten.convolution]
    buf148 = extern_kernels.convolution(buf147, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf148, (1, 128, 8, 8), (8192, 1, 1024, 128))
    buf149 = buf142; del buf142  # reuse
    buf150 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf152 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf153 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf155 = empty_strided((1, 128, 8, 8), (8192, 1, 1024, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_17(c_void_p(buf148.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf155.data_ptr()))
    del primals_210
    del primals_211
    del primals_51
    # Source Nodes: [getattr_l__self___layer2___1___conv3], Original ATen: [aten.convolution]
    buf156 = extern_kernels.convolution(buf155, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf156, (1, 512, 8, 8), (32768, 1, 4096, 512))
    buf157 = buf133; del buf133  # reuse
    buf158 = buf126; del buf126  # reuse
    buf160 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf162 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf161 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf163 = empty_strided((1, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_18(c_void_p(buf156.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()))
    del primals_213
    del primals_214
    del primals_54
    # Source Nodes: [getattr_l__self___layer2___2___conv1], Original ATen: [aten.convolution]
    buf164 = extern_kernels.convolution(buf163, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf164, (1, 128, 8, 8), (8192, 1, 1024, 128))
    buf165 = buf150; del buf150  # reuse
    buf166 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf168 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf170 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf171 = empty_strided((1, 128, 8, 8), (8192, 1, 1024, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_19(c_void_p(buf164.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()))
    del primals_216
    del primals_217
    del primals_57
    # Source Nodes: [getattr_l__self___layer2___2___conv2], Original ATen: [aten.convolution]
    buf172 = extern_kernels.convolution(buf171, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf172, (1, 128, 8, 8), (8192, 1, 1024, 128))
    buf173 = buf166; del buf166  # reuse
    buf174 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf176 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf178 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf177 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf179 = empty_strided((1, 128, 8, 8), (8192, 1, 1024, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_20(c_void_p(buf172.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()))
    del primals_219
    del primals_220
    del primals_60
    # Source Nodes: [getattr_l__self___layer2___2___conv3], Original ATen: [aten.convolution]
    buf180 = extern_kernels.convolution(buf179, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf180, (1, 512, 8, 8), (32768, 1, 4096, 512))
    buf181 = buf158; del buf158  # reuse
    buf182 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf184 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf186 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf185 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf187 = empty_strided((1, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_21(c_void_p(buf180.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()))
    del primals_222
    del primals_223
    del primals_63
    # Source Nodes: [getattr_l__self___layer2___3___conv1], Original ATen: [aten.convolution]
    buf188 = extern_kernels.convolution(buf187, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf188, (1, 128, 8, 8), (8192, 1, 1024, 128))
    buf189 = buf174; del buf174  # reuse
    buf190 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf192 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf194 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf193 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf195 = empty_strided((1, 128, 8, 8), (8192, 1, 1024, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_22(c_void_p(buf188.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()))
    del primals_225
    del primals_226
    del primals_66
    # Source Nodes: [getattr_l__self___layer2___3___conv2], Original ATen: [aten.convolution]
    buf196 = extern_kernels.convolution(buf195, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf196, (1, 128, 8, 8), (8192, 1, 1024, 128))
    buf197 = buf190; del buf190  # reuse
    buf198 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf200 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf202 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf201 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf203 = empty_strided((1, 128, 8, 8), (8192, 1, 1024, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_23(c_void_p(buf196.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()))
    del buf198
    del primals_228
    del primals_229
    del primals_69
    # Source Nodes: [getattr_l__self___layer2___3___conv3], Original ATen: [aten.convolution]
    buf204 = extern_kernels.convolution(buf203, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf204, (1, 512, 8, 8), (32768, 1, 4096, 512))
    buf205 = buf182; del buf182  # reuse
    buf206 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf208 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf210 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf209 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((1, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_24(c_void_p(buf204.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()))
    del primals_231
    del primals_232
    del primals_72
    # Source Nodes: [getattr_l__self___layer3___0___conv1], Original ATen: [aten.convolution]
    buf212 = extern_kernels.convolution(buf211, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf212, (1, 256, 8, 8), (16384, 1, 2048, 256))
    buf213 = buf102; del buf102  # reuse
    buf214 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf216 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf218 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf217 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf219 = empty_strided((1, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_25(c_void_p(buf212.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()))
    del primals_234
    del primals_235
    del primals_75
    # Source Nodes: [getattr_l__self___layer3___0___conv2], Original ATen: [aten.convolution]
    buf220 = extern_kernels.convolution(buf219, buf8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf220, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf221 = buf214; del buf214  # reuse
    buf222 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf224 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf226 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf225 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf227 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_26(c_void_p(buf220.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()))
    del primals_237
    del primals_238
    del primals_78
    # Source Nodes: [getattr_l__self___layer3___0___conv3], Original ATen: [aten.convolution]
    buf228 = extern_kernels.convolution(buf227, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf228, (1, 1024, 4, 4), (16384, 1, 4096, 1024))
    buf229 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf230 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf232 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf234 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf233 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_27(c_void_p(buf228.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf233.data_ptr()))
    del primals_240
    del primals_241
    # Source Nodes: [getattr_l__self___layer3___0___downsample_0], Original ATen: [aten.convolution]
    buf235 = extern_kernels.convolution(buf211, primals_82, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf235, (1, 1024, 4, 4), (16384, 1, 4096, 1024))
    buf236 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf237 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf239 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf241 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf240 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf242 = empty_strided((1, 1024, 4, 4), (16384, 1, 4096, 1024), device='cpu', dtype=torch.float32)
    buf243 = buf242; del buf242  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_relu_28(c_void_p(buf243.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf240.data_ptr()))
    del primals_243
    del primals_244
    del primals_81
    del primals_84
    # Source Nodes: [getattr_l__self___layer3___1___conv1], Original ATen: [aten.convolution]
    buf244 = extern_kernels.convolution(buf243, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf244, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf245 = buf222; del buf222  # reuse
    buf246 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf248 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf250 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf249 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf251 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_29(c_void_p(buf244.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()))
    del primals_246
    del primals_247
    del primals_87
    # Source Nodes: [getattr_l__self___layer3___1___conv2], Original ATen: [aten.convolution]
    buf252 = extern_kernels.convolution(buf251, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf252, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf253 = buf246; del buf246  # reuse
    buf254 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf256 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf258 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf257 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf259 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_30(c_void_p(buf252.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()))
    del primals_249
    del primals_250
    del primals_90
    # Source Nodes: [getattr_l__self___layer3___1___conv3], Original ATen: [aten.convolution]
    buf260 = extern_kernels.convolution(buf259, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf260, (1, 1024, 4, 4), (16384, 1, 4096, 1024))
    buf261 = buf237; del buf237  # reuse
    buf262 = buf230; del buf230  # reuse
    buf264 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf266 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf265 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf267 = empty_strided((1, 1024, 4, 4), (16384, 1, 4096, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_31(c_void_p(buf260.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf267.data_ptr()))
    del primals_252
    del primals_253
    del primals_93
    # Source Nodes: [getattr_l__self___layer3___2___conv1], Original ATen: [aten.convolution]
    buf268 = extern_kernels.convolution(buf267, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf268, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf269 = buf254; del buf254  # reuse
    buf270 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf272 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf274 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf273 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf275 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_32(c_void_p(buf268.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf275.data_ptr()))
    del primals_255
    del primals_256
    del primals_96
    # Source Nodes: [getattr_l__self___layer3___2___conv2], Original ATen: [aten.convolution]
    buf276 = extern_kernels.convolution(buf275, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf276, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf277 = buf270; del buf270  # reuse
    buf278 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf280 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf282 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf281 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf283 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_33(c_void_p(buf276.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf283.data_ptr()))
    del primals_258
    del primals_259
    del primals_99
    # Source Nodes: [getattr_l__self___layer3___2___conv3], Original ATen: [aten.convolution]
    buf284 = extern_kernels.convolution(buf283, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf284, (1, 1024, 4, 4), (16384, 1, 4096, 1024))
    buf285 = buf262; del buf262  # reuse
    buf286 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf288 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf290 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf289 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf291 = empty_strided((1, 1024, 4, 4), (16384, 1, 4096, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_34(c_void_p(buf284.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    del primals_102
    del primals_261
    del primals_262
    # Source Nodes: [getattr_l__self___layer3___3___conv1], Original ATen: [aten.convolution]
    buf292 = extern_kernels.convolution(buf291, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf292, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf293 = buf278; del buf278  # reuse
    buf294 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf296 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf298 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf297 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf299 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_35(c_void_p(buf292.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()))
    del primals_105
    del primals_264
    del primals_265
    # Source Nodes: [getattr_l__self___layer3___3___conv2], Original ATen: [aten.convolution]
    buf300 = extern_kernels.convolution(buf299, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf300, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf301 = buf294; del buf294  # reuse
    buf302 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf304 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf306 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf305 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf307 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_36(c_void_p(buf300.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf307.data_ptr()))
    del primals_108
    del primals_267
    del primals_268
    # Source Nodes: [getattr_l__self___layer3___3___conv3], Original ATen: [aten.convolution]
    buf308 = extern_kernels.convolution(buf307, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf308, (1, 1024, 4, 4), (16384, 1, 4096, 1024))
    buf309 = buf286; del buf286  # reuse
    buf310 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf312 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf314 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf313 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf315 = empty_strided((1, 1024, 4, 4), (16384, 1, 4096, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_37(c_void_p(buf308.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf315.data_ptr()))
    del primals_111
    del primals_270
    del primals_271
    # Source Nodes: [getattr_l__self___layer3___4___conv1], Original ATen: [aten.convolution]
    buf316 = extern_kernels.convolution(buf315, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf316, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf317 = buf302; del buf302  # reuse
    buf318 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf320 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf322 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf321 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf323 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_38(c_void_p(buf316.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf323.data_ptr()))
    del primals_114
    del primals_273
    del primals_274
    # Source Nodes: [getattr_l__self___layer3___4___conv2], Original ATen: [aten.convolution]
    buf324 = extern_kernels.convolution(buf323, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf324, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf325 = buf318; del buf318  # reuse
    buf326 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf328 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf330 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf329 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf331 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_39(c_void_p(buf324.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()))
    del primals_117
    del primals_276
    del primals_277
    # Source Nodes: [getattr_l__self___layer3___4___conv3], Original ATen: [aten.convolution]
    buf332 = extern_kernels.convolution(buf331, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf332, (1, 1024, 4, 4), (16384, 1, 4096, 1024))
    buf333 = buf310; del buf310  # reuse
    buf334 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf336 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf338 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf337 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf339 = empty_strided((1, 1024, 4, 4), (16384, 1, 4096, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_40(c_void_p(buf332.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()))
    del primals_120
    del primals_279
    del primals_280
    # Source Nodes: [getattr_l__self___layer3___5___conv1], Original ATen: [aten.convolution]
    buf340 = extern_kernels.convolution(buf339, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf340, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf341 = buf326; del buf326  # reuse
    buf342 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf344 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf346 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf345 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf347 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_41(c_void_p(buf340.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()))
    del primals_123
    del primals_282
    del primals_283
    # Source Nodes: [getattr_l__self___layer3___5___conv2], Original ATen: [aten.convolution]
    buf348 = extern_kernels.convolution(buf347, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf348, (1, 256, 4, 4), (4096, 1, 1024, 256))
    buf349 = buf342; del buf342  # reuse
    buf350 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf352 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf354 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf353 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf355 = empty_strided((1, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_42(c_void_p(buf348.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf355.data_ptr()))
    del buf350
    del primals_126
    del primals_285
    del primals_286
    # Source Nodes: [getattr_l__self___layer3___5___conv3], Original ATen: [aten.convolution]
    buf356 = extern_kernels.convolution(buf355, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf356, (1, 1024, 4, 4), (16384, 1, 4096, 1024))
    buf357 = buf334; del buf334  # reuse
    buf358 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf360 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf362 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf361 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf363 = empty_strided((1, 1024, 4, 4), (16384, 1, 4096, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_43(c_void_p(buf356.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()))
    del buf358
    del primals_129
    del primals_288
    del primals_289
    # Source Nodes: [getattr_l__self___layer4___0___conv1], Original ATen: [aten.convolution]
    buf364 = extern_kernels.convolution(buf363, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf364, (1, 512, 4, 4), (8192, 1, 2048, 512))
    buf365 = buf206; del buf206  # reuse
    buf366 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf368 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf370 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf369 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf371 = empty_strided((1, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_44(c_void_p(buf364.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf371.data_ptr()))
    del primals_132
    del primals_291
    del primals_292
    # Source Nodes: [getattr_l__self___layer4___0___conv2], Original ATen: [aten.convolution]
    buf372 = extern_kernels.convolution(buf371, buf14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf372, (1, 512, 2, 2), (2048, 1, 1024, 512))
    buf373 = buf366; del buf366  # reuse
    buf374 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf375 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf376 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf377 = empty_strided((1, 512, 2, 2), (2048, 1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_45(c_void_p(buf372.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del primals_135
    del primals_294
    del primals_295
    # Source Nodes: [getattr_l__self___layer4___0___conv3], Original ATen: [aten.convolution]
    buf378 = extern_kernels.convolution(buf377, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf378, (1, 2048, 2, 2), (8192, 1, 4096, 2048))
    buf379 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf380 = empty_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf381 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf382 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_46(c_void_p(buf378.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()))
    del primals_297
    del primals_298
    # Source Nodes: [getattr_l__self___layer4___0___downsample_0], Original ATen: [aten.convolution]
    buf383 = extern_kernels.convolution(buf363, primals_139, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf383, (1, 2048, 2, 2), (8192, 1, 4096, 2048))
    buf384 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf385 = empty_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf386 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf387 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf388 = empty_strided((1, 2048, 2, 2), (8192, 1, 4096, 2048), device='cpu', dtype=torch.float32)
    buf389 = buf388; del buf388  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_relu_47(c_void_p(buf389.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()))
    del primals_138
    del primals_141
    del primals_300
    del primals_301
    # Source Nodes: [getattr_l__self___layer4___1___conv1], Original ATen: [aten.convolution]
    buf390 = extern_kernels.convolution(buf389, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf390, (1, 512, 2, 2), (2048, 1, 1024, 512))
    buf391 = reinterpret_tensor(buf374, (1, 512, 1, 1), (512, 1, 512, 512)); del buf374  # reuse
    buf392 = reinterpret_tensor(buf373, (1, 512, 1, 1), (512, 1, 1, 1)); del buf373  # reuse
    buf393 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf394 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf395 = reinterpret_tensor(buf385, (1, 512, 2, 2), (2048, 1, 1024, 512)); del buf385  # reuse
    cpp_fused__native_batch_norm_legit_functional_relu_48(c_void_p(buf390.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()))
    del primals_144
    del primals_303
    del primals_304
    # Source Nodes: [getattr_l__self___layer4___1___conv2], Original ATen: [aten.convolution]
    buf396 = extern_kernels.convolution(buf395, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf396, (1, 512, 2, 2), (2048, 1, 1024, 512))
    buf397 = reinterpret_tensor(buf392, (1, 512, 1, 1), (512, 1, 512, 512)); del buf392  # reuse
    buf398 = reinterpret_tensor(buf391, (1, 512, 1, 1), (512, 1, 1, 1)); del buf391  # reuse
    buf399 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf400 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf401 = reinterpret_tensor(buf384, (1, 512, 2, 2), (2048, 1, 1024, 512)); del buf384  # reuse
    cpp_fused__native_batch_norm_legit_functional_relu_49(c_void_p(buf396.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()))
    del primals_147
    del primals_306
    del primals_307
    # Source Nodes: [getattr_l__self___layer4___1___conv3], Original ATen: [aten.convolution]
    buf402 = extern_kernels.convolution(buf401, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf402, (1, 2048, 2, 2), (8192, 1, 4096, 2048))
    buf403 = reinterpret_tensor(buf380, (1, 2048, 1, 1), (2048, 1, 2048, 2048)); del buf380  # reuse
    buf404 = reinterpret_tensor(buf379, (1, 2048, 1, 1), (2048, 1, 1, 1)); del buf379  # reuse
    buf405 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf406 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf407 = empty_strided((1, 2048, 2, 2), (8192, 1, 4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_50(c_void_p(buf402.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    del primals_150
    del primals_309
    del primals_310
    # Source Nodes: [getattr_l__self___layer4___2___conv1], Original ATen: [aten.convolution]
    buf408 = extern_kernels.convolution(buf407, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf408, (1, 512, 2, 2), (2048, 1, 1024, 512))
    buf409 = reinterpret_tensor(buf398, (1, 512, 1, 1), (512, 1, 512, 512)); del buf398  # reuse
    buf410 = reinterpret_tensor(buf397, (1, 512, 1, 1), (512, 1, 1, 1)); del buf397  # reuse
    buf411 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf412 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf413 = reinterpret_tensor(buf404, (1, 512, 2, 2), (2048, 1, 1024, 512)); del buf404  # reuse
    cpp_fused__native_batch_norm_legit_functional_relu_51(c_void_p(buf408.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()))
    del primals_153
    del primals_312
    del primals_313
    # Source Nodes: [getattr_l__self___layer4___2___conv2], Original ATen: [aten.convolution]
    buf414 = extern_kernels.convolution(buf413, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf414, (1, 512, 2, 2), (2048, 1, 1024, 512))
    buf415 = reinterpret_tensor(buf410, (1, 512, 1, 1), (512, 1, 512, 512)); del buf410  # reuse
    buf416 = reinterpret_tensor(buf409, (1, 512, 1, 1), (512, 1, 1, 1)); del buf409  # reuse
    buf417 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf418 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf419 = reinterpret_tensor(buf403, (1, 512, 2, 2), (2048, 1, 1024, 512)); del buf403  # reuse
    cpp_fused__native_batch_norm_legit_functional_relu_52(c_void_p(buf414.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()))
    del buf415
    del buf416
    del primals_156
    del primals_315
    del primals_316
    # Source Nodes: [getattr_l__self___layer4___2___conv3], Original ATen: [aten.convolution]
    buf420 = extern_kernels.convolution(buf419, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf420, (1, 2048, 2, 2), (8192, 1, 4096, 2048))
    buf421 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf422 = empty_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf423 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf424 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf425 = empty_strided((1, 2048, 2, 2), (8192, 1, 4096, 2048), device='cpu', dtype=torch.float32)
    buf426 = empty_strided((1, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_mean_relu_view_53(c_void_p(buf420.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()))
    del buf421
    del buf422
    del primals_159
    del primals_318
    del primals_319
    buf427 = empty_strided((1, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___fc], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_161, buf426, reinterpret_tensor(primals_160, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf427)
    del primals_161
    buf428 = empty_strided((1, 2048, 2, 2), (8192, 1, 4096, 2048), device='cpu', dtype=torch.bool)
    buf429 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf430 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf431 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf432 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf433 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf434 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf435 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf436 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf437 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf438 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf439 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf440 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf441 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf442 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf443 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf444 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf445 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf446 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf447 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf448 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf449 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf450 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf451 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf452 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf453 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf454 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf455 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf456 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf457 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf458 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf459 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf460 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf461 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf462 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf463 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf464 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf465 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf466 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf467 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf468 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf469 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf470 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf471 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf472 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf473 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf474 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf475 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf476 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf477 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf478 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf479 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf480 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf481 = empty_strided((), (), device='cpu', dtype=torch.int64)
    cpp_fused_add_threshold_backward_54(c_void_p(buf425.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()))
    del buf425
    del primals_164
    del primals_167
    del primals_170
    del primals_173
    del primals_176
    del primals_179
    del primals_182
    del primals_185
    del primals_188
    del primals_191
    del primals_194
    del primals_197
    del primals_200
    del primals_203
    del primals_206
    del primals_209
    del primals_212
    del primals_215
    del primals_218
    del primals_221
    del primals_224
    del primals_227
    del primals_230
    del primals_233
    del primals_236
    del primals_239
    del primals_242
    del primals_245
    del primals_248
    del primals_251
    del primals_254
    del primals_257
    del primals_260
    del primals_263
    del primals_266
    del primals_269
    del primals_272
    del primals_275
    del primals_278
    del primals_281
    del primals_284
    del primals_287
    del primals_290
    del primals_293
    del primals_296
    del primals_299
    del primals_302
    del primals_305
    del primals_308
    del primals_311
    del primals_314
    del primals_317
    del primals_320
    return (buf23, buf24, buf429, buf33, buf34, buf430, buf41, buf42, buf431, buf49, buf50, buf432, buf56, buf57, buf433, buf65, buf66, buf434, buf73, buf74, buf435, buf81, buf82, buf436, buf89, buf90, buf437, buf97, buf98, buf438, buf105, buf106, buf439, buf113, buf114, buf440, buf121, buf122, buf441, buf129, buf130, buf442, buf136, buf137, buf443, buf145, buf146, buf444, buf153, buf154, buf445, buf161, buf162, buf446, buf169, buf170, buf447, buf177, buf178, buf448, buf185, buf186, buf449, buf193, buf194, buf450, buf201, buf202, buf451, buf209, buf210, buf452, buf217, buf218, buf453, buf225, buf226, buf454, buf233, buf234, buf455, buf240, buf241, buf456, buf249, buf250, buf457, buf257, buf258, buf458, buf265, buf266, buf459, buf273, buf274, buf460, buf281, buf282, buf461, buf289, buf290, buf462, buf297, buf298, buf463, buf305, buf306, buf464, buf313, buf314, buf465, buf321, buf322, buf466, buf329, buf330, buf467, buf337, buf338, buf468, buf345, buf346, buf469, buf353, buf354, buf470, buf361, buf362, buf471, buf369, buf370, buf472, buf375, buf376, buf473, buf381, buf382, buf474, buf386, buf387, buf475, buf393, buf394, buf476, buf399, buf400, buf477, buf405, buf406, buf478, buf411, buf412, buf479, buf417, buf418, buf480, buf423, buf424, buf481, buf427, buf0, primals_2, primals_4, primals_5, buf1, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, buf2, primals_20, primals_22, primals_23, primals_25, primals_26, buf3, primals_29, primals_31, primals_32, primals_34, primals_35, buf4, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, buf5, primals_50, primals_52, primals_53, primals_55, primals_56, buf6, primals_59, primals_61, primals_62, primals_64, primals_65, buf7, primals_68, primals_70, primals_71, primals_73, primals_74, buf8, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, buf9, primals_89, primals_91, primals_92, primals_94, primals_95, buf10, primals_98, primals_100, primals_101, primals_103, primals_104, buf11, primals_107, primals_109, primals_110, primals_112, primals_113, buf12, primals_116, primals_118, primals_119, primals_121, primals_122, buf13, primals_125, primals_127, primals_128, primals_130, primals_131, buf14, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, buf15, primals_146, primals_148, primals_149, primals_151, primals_152, buf16, primals_155, primals_157, primals_158, buf17, buf18, buf22, buf25, buf26, buf27, buf28, buf32, buf35, buf36, buf40, buf43, buf44, buf48, buf51, buf55, buf59, buf60, buf64, buf67, buf68, buf72, buf75, buf76, buf80, buf83, buf84, buf88, buf91, buf92, buf96, buf99, buf100, buf104, buf107, buf108, buf112, buf115, buf116, buf120, buf123, buf124, buf128, buf131, buf135, buf139, buf140, buf144, buf147, buf148, buf152, buf155, buf156, buf160, buf163, buf164, buf168, buf171, buf172, buf176, buf179, buf180, buf184, buf187, buf188, buf192, buf195, buf196, buf200, buf203, buf204, buf208, buf211, buf212, buf216, buf219, buf220, buf224, buf227, buf228, buf232, buf235, buf239, buf243, buf244, buf248, buf251, buf252, buf256, buf259, buf260, buf264, buf267, buf268, buf272, buf275, buf276, buf280, buf283, buf284, buf288, buf291, buf292, buf296, buf299, buf300, buf304, buf307, buf308, buf312, buf315, buf316, buf320, buf323, buf324, buf328, buf331, buf332, buf336, buf339, buf340, buf344, buf347, buf348, buf352, buf355, buf356, buf360, buf363, buf364, buf368, buf371, buf372, buf377, buf378, buf383, buf389, buf390, buf395, buf396, buf401, buf402, buf407, buf408, buf413, buf414, buf419, buf420, buf426, reinterpret_tensor(primals_160, (1000, 2048), (2048, 1), 0), buf428, reinterpret_tensor(buf365, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf357, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf349, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf341, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf333, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf325, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf317, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf309, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf301, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf293, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf285, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf277, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf269, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf261, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf253, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf245, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf236, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf229, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf221, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf213, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf205, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf197, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf189, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf181, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf173, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf165, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf157, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf149, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf141, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf132, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf125, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf117, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf109, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf101, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf93, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf85, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf77, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf69, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf61, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf52, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf45, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf37, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf29, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf19, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_165 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_168 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_171 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_174 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_177 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_180 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_183 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_186 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_189 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_192 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_195 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_198 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_201 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_204 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_207 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_210 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_213 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_216 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_219 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_222 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_225 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_228 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_231 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_234 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_237 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_240 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_243 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_246 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_249 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_252 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_255 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_258 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_261 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_264 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_267 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_270 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_273 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_276 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_279 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_282 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_285 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_288 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_291 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_294 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_297 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_300 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_303 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_306 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_309 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_312 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_315 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_318 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_321 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
