/*
 * @file config.hpp
 *
 * @copyright Copyright (C) 2024 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
 * 
 * MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions: 
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "internals/checkError.hpp"

unsigned int get_arch()
{
    int device_id = -1;
    check_cuda( cudaGetDevice(&device_id) );
    cudaDeviceProp prop;
    check_cuda( cudaGetDeviceProperties ( &prop, device_id ) );
    unsigned int cc = prop.major * 10 + prop.minor;
    return cc;
}

template<class Config>
auto dispatch_target_arch(const unsigned int target_arch)
{
    switch(target_arch)
    {
        case 80:
            return Config::template architecture_config<80>::params;
        case 89:
            return Config::template architecture_config<89>::params;
    }
    return Config::template architecture_config<0>::params;
}


template<class Config>
__device__
constexpr auto device_params()
{
#ifdef __CUDA_ARCH__
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 890
    return Config::template architecture_config<89>::params;
#elif __CUDA_ARCH__ == 800
    return Config::template architecture_config<80>::params;
#else
#warning "missing cuda arch"
#endif
#endif
#else
    return Config::template architecture_config<0>::params;
#endif
}

#endif