#include <torch/extension.h>
#include "device_data.h"
#include "kernel_config.h"
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <utility>

namespace semnan_cuda {
    // A structure for holding a value and its index
    template <typename scalar_t>
    struct indexed_scalar_t {
        int index;
        scalar_t value;
    };

    template <typename scalar_t>
    using parent_weight_t = indexed_scalar_t<scalar_t>;

    template <typename scalar_t>
    using child_weight_t = indexed_scalar_t<scalar_t>;

    // A structure for helping chunk arrays, lists, etc.
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

    std::pair<dim3, dim3> get_blocks_and_threads(const int32_t width, const int32_t height) {
        const dim3 threads(1, min(height, THREADS_PER_BLOCK));
        const dim3 blocks(width, (height + threads.y - 1) / threads.y);
        return std::make_pair(blocks, threads);
    }

    // Concrete types
    template class DeviceData<float>;
    template class DeviceData<double>;

    namespace covar {
        template <typename scalar_t>
        __global__ void forward_kernel(
                DeviceData<scalar_t> data,
                LayerData layer
        ) {
            /*
             * Compute covariance at [i, j].
             * This requires to get the Pa(i)×Pa(j) covariance sub-matrix.
             * The only parent of the nodes previously met (alias nodes) is themselves.
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

                // Store data to shared memory
                for (int32_t k = threadIdx.y; k < k_max; k += blockDim.y) {
                    const int32_t i_parent = data.get_parent(i, shared_base + k, layer);
                    i_data[k].index = i_parent;
                    i_data[k].value = data.get_weight(i_parent, i, layer);
                }

                __syncthreads();

                if (y < layer.get_num_vars() && j <= i && j_num_parents > 0) {
                    scalar_t covariance_ij = shared_round > 0 ? data.get_covariance(i, j) : 0.0;

                    for (int32_t l = 0; l < j_num_parents; l++) {
                        const int32_t j_parent = data.get_parent(j, l, layer);
                        const scalar_t j_parent_weight = data.get_weight(j_parent, j, layer);
                        scalar_t lambda_il = 0.0;

                        for (int32_t k = 0; k < k_max; k++) {
                            const scalar_t lambda_ikl = i_data[k].value * data.get_covariance(i_data[k].index, j_parent);
                            lambda_il += lambda_ikl;
                            covariance_ij += lambda_ikl * j_parent_weight;
                        }

                        // lambda is computed and stored along with covariance
                        data.set_lambda(j_parent, i, lambda_il + (shared_round > 0 ? data.get_lambda(j_parent, i) : 0.0));
                    }

                    data.set_covariance(i, j, covariance_ij);
                }

                __syncthreads();
            }
        }

        template <typename scalar_t>
        __global__ void backward_covariance_kernel(
                DeviceData<scalar_t> data,
                LayerData layer
        ) {
            /*
             * Compute covariance_grad at [i, j].
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
                    scalar_t covariance_grad_ij = shared_round > 0 ? data.get_covariance_grad(i, j, layer.idx % 2) : 0.0;

                    for (int32_t l = 0; l < j_num_children; l++) {
                        const int32_t j_child = data.get_child(j, j_begin + l, layer);
                        const scalar_t j_child_weight = data.get_weight(j, j_child, layer);

                        for (int32_t k = 0; k < k_max; k++) {
                            covariance_grad_ij += i_data[k].value
                                                * data.get_covariance_grad(i_data[k].index, j_child, (layer.idx + 1) % 2)
                                                * j_child_weight;
                        }
                    }

                    data.set_covariance_grad(i, j, covariance_grad_ij, layer.idx % 2);
                }

                __syncthreads();
            }
        }

    //     template <typename scalar_t>
    //     __global__ void backward_weights_kernel_noshare(
    //             DeviceData<scalar_t> data,
    //             LayerData layer
    //     ) {
    //         /*
    //          * Compute weight_grad at [i, j].
    //          */
    //         const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    //         const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    //         const int32_t i = data.get_layer_var(x, layer);
    //         int32_t i_begin, i_end;
    //         data.get_children_range(i, i_begin, i_end, layer);
    //
    //         if (y < i_end) {
    //             const int32_t j = data.get_child(i, y, layer);
    //             scalar_t weight_grad_ij = 0.0;
    //
    //             for (int32_t k = 0; k < data.get_vis_len(); k++) {
    //                 weight_grad_ij += data.get_lambda(i, k) * data.get_covariance_grad(k, j, (layer.idx + 1) % 2);
    //             }
    //
    //             data.set_weight_grad(i, j, weight_grad_ij);
    //         }
    //     }

        template <typename scalar_t> /* * */
        __global__ void backward_weights_kernel(
                DeviceData<scalar_t> data,
                LayerData layer
        ) {
            /*
             * Compute weight_grad at [i, j].
             */
            __shared__ unsigned char shared_memory[SHARED_MEMORY_SIZE_BACKWARD];
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            const int32_t i = data.get_layer_var(x, layer);
            int32_t i_begin, i_end;
            data.get_children_range(i, i_begin, i_end, layer);
            const int32_t i_num_children = i_end - i_begin;

            auto i_data = reinterpret_cast<scalar_t*>(shared_memory);
            const int32_t shared_size = SHARED_MEMORY_SIZE_BACKWARD / sizeof(scalar_t);
            const array_chunk chunker(data.get_vis_len(), shared_size);

            for (int32_t shared_round = 0; shared_round < chunker.num_chunks(); shared_round++) {
                const int32_t k_max = chunker.chunk_size(shared_round);
                const int32_t shared_base = chunker.chunk_base(shared_round);

                for (int32_t k = threadIdx.y; k < k_max; k += blockDim.y) {
                    i_data[k] = data.get_lambda(i, shared_base + k);
                }

                __syncthreads();

                if (y < i_end) {
                    const int32_t j = data.get_child(i, y, layer);
                    scalar_t weight_grad_ij = shared_round > 0 ? data.get_weight_grad(i, j, layer) : 0.0;

                    for (int32_t k = 0; k < k_max; k++) {
                        weight_grad_ij += i_data[k] * data.get_covariance_grad(shared_base + k, j, (layer.idx + 1) % 2);
                    }

                    data.set_weight_grad(i, j, weight_grad_ij);
                }

                __syncthreads();
            }
        }

        // template <typename scalar_t>
        // void forward_accum(const std::vector<LayerData>& layers_vec, DeviceData<scalar_t>& data) {
        //     dim3 threads, blocks;
        //
        //     for (int32_t l = 1; l < layers_vec.size(); l++) {
        //         const auto& layer = layers_vec[l];
        //         const dim3 data_size(layer.get_num_new_vars(), data.get_lat_len());
        //         const dim3 grid_size = get_grid_size(data_size, cfg::w_accumulation_block_size);
        //         const size_t mem_size = get_mem_size<scalar_t>(cfg::w_accumulation_block_size);
        //         forward_accum_kernel<scalar_t><<<grid_size, cfg::w_accumulation_block_size, mem_size>>>(data, layer);
        //     }
        // }

        // The forward CUDA function. Calls the cuda forward kernel layer by layer.
        template <typename scalar_t>
        void forward(const std::vector<LayerData>& layers_vec, DeviceData<scalar_t>& data) {
            dim3 threads, blocks;

            for (int32_t l = 1; l < layers_vec.size(); l++) {
                const auto& layer = layers_vec[l];
                std::tie(blocks, threads) = get_blocks_and_threads(layer.get_num_new_vars(), layer.get_num_vars());
                forward_kernel<scalar_t><<<blocks, threads>>>(data, layer);
            }
        }

        // The backward CUDA function. Calls the two cuda backward kernels concurrently layer by layer.
        // These kernels compute the weights gradient and the temporary covariance gradient.
        template <typename scalar_t>
        void backward(const std::vector<LayerData>& layers_vec, DeviceData<scalar_t>& data) {
            dim3 threads, blocks;
            cudaStream_t covariance_stream, weights_stream;
            cudaStreamCreate(&covariance_stream);
            cudaStreamCreate(&weights_stream);

            for (int32_t l = layers_vec.size() - 2; l >= 0; l--) {
                const auto& layer = layers_vec[l];
                const auto& next_layer = layers_vec[l + 1];

                if (l > 0) {
                    std::tie(blocks, threads) = get_blocks_and_threads(layer.get_num_vars(), layer.get_num_vars());
                    backward_covariance_kernel<scalar_t><<<blocks, threads, 0, covariance_stream>>>(data, layer);
                }

                std::tie(blocks, threads) = get_blocks_and_threads(layer.get_num_vars(), next_layer.get_num_new_vars());
                backward_weights_kernel<scalar_t><<<blocks, threads, 0, weights_stream>>>(data, layer);
                cudaDeviceSynchronize();
            }

            cudaStreamDestroy(covariance_stream);
            cudaStreamDestroy(weights_stream);
        }

        // ==============
        // Concrete types
        template void forward<float>(const std::vector<LayerData>&, DeviceData<float>&);
        template void forward<double>(const std::vector<LayerData>&, DeviceData<double>&);
        template void backward<float>(const std::vector<LayerData>&, DeviceData<float>&);
        template void backward<double>(const std::vector<LayerData>&, DeviceData<double>&);
    }

    namespace accum {
        template <typename scalar_t>
        __global__ void forward_kernel(
                DeviceData<scalar_t> data,
                LayerData layer
        ) {
            /*
             * Compute W^acc[:, i].
             */
            __shared__ unsigned char shared_memory[SHARED_MEMORY_SIZE_FORWARD];
            const int32_t i = blockIdx.x * blockDim.x + threadIdx.x + layer.base;
            const int32_t j = blockIdx.y * blockDim.y + threadIdx.y - data.get_lat_len();
            const int32_t i_num_parents = data.get_num_parents(i, layer);

            auto i_data = reinterpret_cast<parent_weight_t<scalar_t> *>(shared_memory);
            const int32_t shared_size = SHARED_MEMORY_SIZE_FORWARD / sizeof(parent_weight_t<scalar_t>);
            const array_chunk chunker(i_num_parents, shared_size);

            for (int32_t shared_round = 0; shared_round < chunker.num_chunks(); shared_round++) {
                const int32_t k_max = chunker.chunk_size(shared_round);
                const int32_t shared_base = chunker.chunk_base(shared_round);

                // Store data to shared memory
                for (int32_t k = threadIdx.y; k < k_max; k += blockDim.y) {
                    const int32_t i_parent = data.get_parent(i, shared_base + k, layer);
                    i_data[k].index = i_parent;
                    i_data[k].value = data.get_weight(i_parent, i, layer);
                }

                __syncthreads();

                if (j < 0) {
                    scalar_t w_accum = shared_round > 0 ? data.get_w_accum(j, i) : 0.0;

                    for (int32_t k = 0; k < k_max; k++) {
                        w_accum += i_data[k].value * data.get_w_accum(j, i_data[k].index);
                    }

                    data.set_w_accum(j, i, w_accum);
                }

                __syncthreads();
            }
        }

//         template <typename scalar_t>
//         __global__ void backward_omega_kernel_noshare(DeviceData<scalar_t> data, LayerData layer) {
//            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
//            const int32_t i = data.get_layer_var(x, layer);
//            const int32_t j = blockIdx.y * blockDim.y + threadIdx.y - data.get_lat_len();
//            int32_t i_begin, i_end;
//            data.get_children_range(i, i_begin, i_end, layer);
//
//             if (j < 0) {
//                 scalar_t omega_ji = 0.0;
//
//                 for (int32_t k = i_begin; k < i_end; k++) {
//                     const int32_t i_child = data.get_child(i, k, layer);
//                    omega_ji += data.get_weight(i, i_child, layer) * data.get_omega(j, i_child, (layer.idx + 1) % 2);
//                 }
//
//                 data.set_omega(j, i, omega_ji, layer.idx % 2);
//             }
//         }

        template <typename scalar_t>
        __global__ void backward_omega_kernel(
                DeviceData<scalar_t> data,
                LayerData layer
        ) {
            __shared__ unsigned char shared_memory[SHARED_MEMORY_SIZE_BACKWARD];
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t i = data.get_layer_var(x, layer);
            const int32_t j = blockIdx.y * blockDim.y + threadIdx.y - data.get_lat_len();
            int32_t i_begin, i_end;
            data.get_children_range(i, i_begin, i_end, layer);
            const int32_t i_num_children = i_end - i_begin;

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

                if (j < 0) {
                    scalar_t omega_ji = shared_round > 0 ? data.get_omega(j, i, layer.idx % 2) : 0.0;

                    for (int32_t k = 0; k < k_max; k++) {
                        omega_ji += i_data[k].value
                                  * data.get_omega(j, i_data[k].index, (layer.idx + 1) % 2);
                    }

                    data.set_omega(j, i, omega_ji, layer.idx % 2);
                }

                __syncthreads();
            }
        }

        template <typename scalar_t>
        __global__ void backward_weights_kernel(
                DeviceData<scalar_t> data,
                LayerData layer
        ) {
            /*
             * Compute weight_grad at [i, j].
             */
            __shared__ unsigned char shared_memory[SHARED_MEMORY_SIZE_BACKWARD];
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            const int32_t i = data.get_layer_var(x, layer);
            int32_t i_begin, i_end;
            data.get_children_range(i, i_begin, i_end, layer);
            const int32_t i_num_children = i_end - i_begin;
            const int32_t lat_len = data.get_lat_len();

            auto i_data = reinterpret_cast<scalar_t*>(shared_memory);
            const int32_t shared_size = SHARED_MEMORY_SIZE_BACKWARD / sizeof(scalar_t);
            const array_chunk chunker(lat_len, shared_size);

            for (int32_t shared_round = 0; shared_round < chunker.num_chunks(); shared_round++) {
                const int32_t k_max = chunker.chunk_size(shared_round);
                const int32_t shared_base = chunker.chunk_base(shared_round);

                for (int32_t k = threadIdx.y; k < k_max; k += blockDim.y) {
                    i_data[k] = data.get_w_accum(shared_base + k - lat_len, i);
                }

                __syncthreads();

                if (y < i_end) {
                    const int32_t j = data.get_child(i, y, layer);
                    scalar_t weight_grad_ij = shared_round > 0 ? data.get_weight_grad(i, j, layer) : 0.0;

                    for (int32_t k = 0; k < k_max; k++) {
                        weight_grad_ij += i_data[k] * data.get_omega(shared_base + k - lat_len, j, (layer.idx + 1) % 2);
                    }

                    data.set_weight_grad(i, j, weight_grad_ij);
                }

                __syncthreads();
            }
        }

        template <typename scalar_t>
        void forward(const std::vector<LayerData>& layers_vec, DeviceData<scalar_t>& data) {
            dim3 threads, blocks;

            for (int32_t l = 1; l < layers_vec.size(); l++) {
                const auto& layer = layers_vec[l];
                std::tie(blocks, threads) = get_blocks_and_threads(layer.get_num_new_vars(), data.get_lat_len());
                forward_kernel<scalar_t><<<blocks, threads>>>(data, layer);
            }
        }

        template <typename scalar_t>
        void backward(const std::vector<LayerData>& layers_vec, DeviceData<scalar_t>& data) {
            dim3 threads, blocks;
            cudaStream_t omega_stream, weights_stream;
            cudaStreamCreate(&omega_stream);
            cudaStreamCreate(&weights_stream);

            for (int32_t l = layers_vec.size() - 2; l >= 0; l--) {
                const auto& layer = layers_vec[l];
                const auto& next_layer = layers_vec[l + 1];

                if (l > 0) {
                    std::tie(blocks, threads) = get_blocks_and_threads(layer.get_num_vars(), data.get_lat_len());
                    backward_omega_kernel<scalar_t><<<blocks, threads, 0, omega_stream>>>(data, layer);
                }

                std::tie(blocks, threads) = get_blocks_and_threads(layer.get_num_vars(), next_layer.get_num_new_vars());
                backward_weights_kernel<scalar_t><<<blocks, threads, 0, weights_stream>>>(data, layer);
                cudaDeviceSynchronize();
            }

            cudaStreamDestroy(omega_stream);
            cudaStreamDestroy(weights_stream);
        }

        // Concrete types
        template void forward<float>(const std::vector<LayerData>&, DeviceData<float>&);
        template void forward<double>(const std::vector<LayerData>&, DeviceData<double>&);
        template void backward<float>(const std::vector<LayerData>&, DeviceData<float>&);
        template void backward<double>(const std::vector<LayerData>&, DeviceData<double>&);
    }
}