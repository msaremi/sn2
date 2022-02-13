#include <torch/extension.h>
#include "device_data.h"
#include "kernel_setup.h"
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <utility>

template <typename scalar_t>
struct indexed_scalar_t {
    int index;
    scalar_t value;
};

template <typename scalar_t>
using parent_weight_t = indexed_scalar_t<scalar_t>;

template <typename scalar_t>
using child_weight_t = indexed_scalar_t<scalar_t>;

template <typename scalar_t>
using neighbour_covariance_t = indexed_scalar_t<scalar_t>;

class array_chunk {
    private:
    int32_t arr_size;
    int32_t chnk_size;

    public:
    __device__ __forceinline__ array_chunk(const int32_t arr_size, const int32_t chnk_size) :
            arr_size(arr_size), chnk_size(chnk_size) { }

    public:
    __device__ __forceinline__ int32_t chunk_size(const int32_t chnk_idx) const {
        int32_t size = arr_size - chnk_idx * chnk_size;
        return (size <= chnk_size) ? size : chnk_size;
    }

    public:
    __device__ __forceinline__ int32_t chunk_base(const int32_t chnk_idx) const {
        return chnk_idx * chnk_size;
    }

    public:
    __device__ __forceinline__ int32_t num_chunks() const {
        return (arr_size + chnk_size - 1) / chnk_size;
    }
};

template <typename scalar_t>
__global__ void semnan_cuda_forward_kernel(
        SEMNANDeviceData<scalar_t> data,
        SEMNANLayerData layer
) {
    /*
     * Compute covariance at [i, j].
     * This requires to get the Pa(i)×Pa(j) covariance sub-matrix.
     * Note however the only parent of the nodes previously met (alias nodes) is themselves.
     * `base` is given to the following methods to check whether the node has already been met.
     */
    __shared__ unsigned char shared_memory[SHARED_MEMORY_SIZE_FORWARD];
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x + layer.base;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t i_num_parents = data.get_num_parents(i, layer);
    int32_t j, j_num_parents;

    if (y < layer.get_num_vars()) {
        j = data.get_layer_var(y, layer);
        j_num_parents = data.get_num_parents(j, layer);
    }

    auto i_data = reinterpret_cast<parent_weight_t<scalar_t> *>(shared_memory);
    const int32_t shared_size = SHARED_MEMORY_SIZE_FORWARD / sizeof(parent_weight_t<scalar_t>);
    const array_chunk chunker(i_num_parents, shared_size);

    for (int32_t shared_round = 0; shared_round < chunker.num_chunks(); shared_round++) {
        const int32_t k_max = chunker.chunk_size(shared_round);
        const int32_t shared_base = chunker.chunk_base(shared_round);

        for (int32_t k = threadIdx.y; k < k_max; k += blockDim.y) {
            const int32_t i_parent = data.get_parent(i, shared_base + k, layer);
            i_data[k].index = i_parent;
            i_data[k].value = data.get_weight(i_parent, i, layer);
        }

    __syncthreads();

        if (y < layer.get_num_vars() && j <= i && j_num_parents > 0) {
            scalar_t covariance_ij = 0.0;

            for (int32_t l = 0; l < j_num_parents; l++) {
                const int32_t j_parent = data.get_parent(j, l, layer);
                const scalar_t j_parent_weight = data.get_weight(j_parent, j, layer);
                scalar_t lambda_il = 0.0;

                for (int32_t k = 0; k < k_max; k++) {
                    const scalar_t lambda_ikl = i_data[k].value * data.get_covariance(i_data[k].index, j_parent);
                    lambda_il += lambda_ikl;
                    covariance_ij += lambda_ikl * j_parent_weight;
                }

                data.set_lambda(j_parent, i, lambda_il + (shared_round > 0 ? data.get_lambda(j_parent, i) : 0.0));
            }

            data.set_covariance(i, j, covariance_ij + (shared_round > 0 ? data.get_covariance(i, j) : 0.0));
        }

        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void semnan_cuda_backward_covariance_kernel(
        SEMNANDeviceData<scalar_t> data,
        SEMNANLayerData layer
) {
    /*
     * Compute covariance_grad at [i, j].
     * TODO: The covariance_grad between two latent variables is currently ignored. Correct it.
     */
    __shared__ unsigned char shared_memory[SHARED_MEMORY_SIZE_BACKWARD];
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t i = data.get_layer_var(x, layer);
    int32_t i_begin, i_end;
    data.get_children_range(i, i_begin, i_end, layer);
    const int32_t i_num_children = i_end - i_begin;
    int32_t j, j_begin, j_end, j_num_children;

    if (y < layer.get_num_vars()) {
        j = data.get_layer_var(y, layer);
        data.get_children_range(j, j_begin, j_end, layer);
        j_num_children = j_end - j_begin;
    }

    auto i_data = reinterpret_cast<child_weight_t<scalar_t> *>(shared_memory);
    const int32_t shared_size = SHARED_MEMORY_SIZE_BACKWARD / sizeof(child_weight_t<scalar_t>);
    const array_chunk chunker(i_num_children, shared_size);

    for (int32_t shared_round = 0; shared_round < chunker.num_chunks(); shared_round++) {
        const int32_t k_max = chunker.chunk_size(shared_round);
        const int32_t shared_base = chunker.chunk_base(shared_round);

        for (int32_t k = threadIdx.y; k < k_max; k += blockDim.y) {
            const int32_t i_child = data.get_child(i, i_begin + shared_base + k, layer);
            i_data[k].index = i_child;
            i_data[k].value = data.get_weight(i, i_child, layer);
        }

        __syncthreads();

        if (y < layer.get_num_vars() && j <= i && j_num_children > 0) {
            scalar_t covariance_grad_ij = 0.0;

            for (int32_t l = 0; l < j_num_children; l++) {
                const int32_t j_child = data.get_child(j, j_begin + l, layer);
                const scalar_t j_child_weight = data.get_weight(j, j_child, layer);

                for (int32_t k = 0; k < k_max; k++) {
                    covariance_grad_ij += i_data[k].value
                                        * data.get_covariance_grad(i_data[k].index, j_child, (layer.idx + 1) % 2)
                                        * j_child_weight;
                }
            }

            data.set_covariance_grad(
                    i, j,
                    covariance_grad_ij + (shared_round > 0 ? data.get_covariance_grad(i, j, layer.idx % 2) : 0.0),
                    layer.idx % 2
            );
        }

        __syncthreads();
    }
}


template <typename scalar_t>
__global__ void semnan_cuda_backward_weights_kernel(
        SEMNANDeviceData<scalar_t> data,
        SEMNANLayerData layer
) {
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t i = data.get_layer_var(x, layer);
    int32_t i_begin, i_end;
    data.get_children_range(i, i_begin, i_end, layer);

    if (y < i_end) {
        const int32_t j = data.get_child(i, y, layer);
        scalar_t weight_grad_ij = 0.0;

        for (int32_t k = 0; k < data.get_vis_len(); k++) {
            weight_grad_ij += data.get_lambda(i, k) * data.get_covariance_grad(k, j, (layer.idx + 1) % 2);
        }

        data.set_weight_grad(i, j, weight_grad_ij);
    }
}

std::pair<dim3, dim3> get_blocks_and_threads(const int32_t width, const int32_t height) {
    const dim3 threads(1, min(height, THREADS_PER_BLOCK));
    const dim3 blocks(width, (height + threads.y - 1) / threads.y);
    return std::make_pair(blocks, threads);
}

template <typename scalar_t>
void semnan_cuda_forward(const std::vector<SEMNANLayerData>& layers_vec, SEMNANDeviceData<scalar_t>& data) {
    dim3 threads, blocks;

    for (int32_t l = 1; l < layers_vec.size(); l++) {
        const auto& layer = layers_vec[l];
        std::tie(blocks, threads) = get_blocks_and_threads(layer.get_num_new_vars(), layer.get_num_vars());
        semnan_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(data, layer);
    }
}

template <typename scalar_t>
void semnan_cuda_backward(const std::vector<SEMNANLayerData>& layers_vec, SEMNANDeviceData<scalar_t>& data) {
    dim3 threads, blocks;
    cudaStream_t covariance_stream, weights_stream;
    cudaStreamCreate(&covariance_stream);
    cudaStreamCreate(&weights_stream);

    for (int32_t l = layers_vec.size() - 2; l >= 0; l--) {
        const auto& layer = layers_vec[l];
        const auto& next_layer = layers_vec[l + 1];

        if (l > 0) {
            std::tie(blocks, threads) = get_blocks_and_threads(layer.get_num_vars(), layer.get_num_vars());
            semnan_cuda_backward_covariance_kernel<scalar_t><<<blocks, threads, 0, covariance_stream>>>(data, layer);
        }

        std::tie(blocks, threads) = get_blocks_and_threads(layer.get_num_vars(), next_layer.get_num_new_vars());
        semnan_cuda_backward_weights_kernel<scalar_t><<<blocks, threads, 0, weights_stream>>>(data, layer);
        cudaDeviceSynchronize();
    }

    cudaStreamDestroy(covariance_stream);
    cudaStreamDestroy(weights_stream);
}


// ==============
// Concrete types
template class SEMNANDeviceData<float>;
template class SEMNANDeviceData<double>;

template void semnan_cuda_forward<float>(const std::vector<SEMNANLayerData>&, SEMNANDeviceData<float>&);
template void semnan_cuda_forward<double>(const std::vector<SEMNANLayerData>&, SEMNANDeviceData<double>&);

template void semnan_cuda_backward<float>(const std::vector<SEMNANLayerData>&, SEMNANDeviceData<float>&);
template void semnan_cuda_backward<double>(const std::vector<SEMNANLayerData>&, SEMNANDeviceData<double>&);