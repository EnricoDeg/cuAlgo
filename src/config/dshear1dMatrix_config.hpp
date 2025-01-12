/*
 * @file histogram_config.hpp
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

#ifndef DSHEAR1DMATRIX_HPP
#define DSHEAR1DMATRIX_HPP

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

struct kernel_config_params
{
    unsigned int block_sizeX = 32;
    unsigned int block_sizeY = 32;
    unsigned int items_per_thread = 1;
};

struct dshear_config_params
{
    kernel_config_params dshear_kernel_config;
};

template<unsigned int BlockSizeX,
         unsigned int BlockSizeY,
         unsigned int ItemsPerThread>
struct dshear_config : public dshear_config_params
{
    static constexpr unsigned int block_sizeX      = BlockSizeX;
    static constexpr unsigned int block_sizeY      = BlockSizeY;
    static constexpr unsigned int items_per_thread = ItemsPerThread;

    constexpr dshear_config()
        : dshear_config_params{
            {BlockSizeX, BlockSizeY, ItemsPerThread}} {};
};

template<class Value>
struct default_dshear_config_base
{
    using type = dshear_config<
        32u,
        32u,
        1u>;
};

template<unsigned int arch, class value_type, class enable = void>
struct default_dshear_config
    : default_dshear_config_base<value_type>::type
{};

// Based on value_type = double
template<class value_type>
struct default_dshear_config<
    89,
    value_type,
    std::enable_if_t<(bool(std::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : dshear_config<32, 32, 1>
{};

// Based on value_type = float
template<class value_type>
struct default_dshear_config<
    89,
    value_type,
    std::enable_if_t<(bool(std::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : dshear_config<32, 32, 2>
{};

template<typename DshearConfig, typename>
struct wrapped_dshear_config
{
    template<unsigned int Arch>
    struct architecture_config
    {
        static constexpr dshear_config_params params = DshearConfig{};
    };
};

struct default_config
{ };

template<typename Value>
struct wrapped_dshear_config<default_config, Value>
{
    template<unsigned int Arch>
    struct architecture_config
    {
        static constexpr dshear_config_params params
            = default_dshear_config<static_cast<unsigned int>(Arch), Value>{};
    };
};

template<typename Type>
template<unsigned int Arch>
constexpr dshear_config_params
    wrapped_dshear_config<default_config, Type>::architecture_config<Arch>::params;

template<class DshearConfig, class Value>
template<unsigned int Arch>
constexpr dshear_config_params
    wrapped_dshear_config<DshearConfig, Value>::
        architecture_config<Arch>::params;
#endif
