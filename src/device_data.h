#ifndef DEVICE_DATA_H
#define DEVICE_DATA_H
#include <torch/extension.h>
#include <stdio.h>

namespace sn2_cuda {
    /*
     * A declaration for the cpp compiler and a definition for the nvcc compiler.
     */

    // Stores the structural data for a layer in sn2
    class LayerData {
        public:
        int32_t idx;
        int32_t base;
        int32_t num;
        int32_t lat_width;

#ifdef __CUDACC__
        public:
        __device__ __host__ __forceinline__ int32_t get_num_vars() const {
            return base + num + lat_width;
        }

        public:
        __device__ __host__ __forceinline__ int32_t get_num_new_vars() const {
            return num;
        }

#else
        public:
        LayerData(
            const int32_t idx,
            const int32_t base
        ):  idx(idx),
            base(base),
            num(0),
            lat_width(0)
        { }

        public:
        LayerData()
        :   idx(0),
            base(0),
            num(0),
            lat_width(0)
        { }
#endif
    };

    /**
     * Stores the reference to all the matrices that are passed to kernel functions
     * facilitates access to matrix entries by providing a model-view architecture
     * @tparam scalar_t `float` or `double`
     */
    template <typename scalar_t>
    class DeviceData {
        private:
        const bool* structure;
        scalar_t* lambda;
        scalar_t* weights;
        scalar_t* covariance;
        scalar_t* weights_grad;
        scalar_t* w_accum;
        scalar_t* covariance_grads[2];
        scalar_t* omegas[2];
        const int32_t* parents;
        const int32_t* parents_bases;
        const int32_t* children;
        const int32_t* children_bases;
        const int32_t* lat_neighbors;
        const int32_t* lat_neighbors_bases;
        const int32_t* lat_range;
        int32_t vis_len;
        int32_t lat_len;
        int32_t num_layers;

#ifdef __CUDACC__
        public:
        __device__ __forceinline__ scalar_t get_w_accum(int32_t a, int32_t d) const {
            if (a == d)
                return 1.0;
            else if (a < 0 && d < 0)
                return 0.0;
            else
                return (a < d) ? w_accum[(a + lat_len) * vis_len + d] : 0.0;
        }

        public:
        __device__ __forceinline__ void set_w_accum(int32_t a, int32_t d, scalar_t val) {
            if (a <= d)
                w_accum[(a + lat_len) * vis_len + d] = val;
        }

        public:
        __device__ __forceinline__ scalar_t get_weight(int32_t p, int32_t c, const LayerData& layer) const {
            if (p == c)
                return 1.0;
            else if (c >= layer.base) // if the variable is visible and appearing on this layer forward
                return weights[(p + lat_len) * vis_len + c];
            else
                return 0.0;
        }

        public:
        __device__ __forceinline__ void set_weight(int32_t p, int32_t c, scalar_t val) {
            if (p <= c)
                weights[(p + lat_len) * vis_len + c] = val;
        }

        public:
        __device__ __forceinline__ scalar_t get_weight_grad(int32_t p, int32_t c, const LayerData& layer) const {
            if (p == c)
                return 0.0;
            else if (c >= layer.base)
                return weights_grad[(p + lat_len) * vis_len + c];
            else
                return 0.0;
        }

        public:
        __device__ __forceinline__ void set_weight_grad(int32_t p, int32_t c, scalar_t val) {
            if (p <= c && structure[(p + lat_len) * vis_len + c])
                weights_grad[(p + lat_len) * vis_len + c] = val;
        }

        public:
        __device__ __forceinline__ scalar_t get_covariance(int32_t u, int32_t v) const {
            if (u >= 0 || v >= 0)
                return covariance[(min(u, v) + lat_len) * vis_len + max(u, v)];
            else
                return u == v ? 1.0 : 0.0;
        }

        public:
        __device__ __forceinline__ void set_covariance(int32_t u, int32_t v, scalar_t val) {
            if (u >= 0 && v >= 0)
                covariance[(u + lat_len) * vis_len + v] = covariance[(v + lat_len) * vis_len + u] = val;
            else if (u >= 0 || v >= 0)
                covariance[(min(u, v) + lat_len) * vis_len + max(u, v)] = val;
        }

        public:
        __device__ __forceinline__ scalar_t get_omega(int32_t a, int32_t d, int8_t buff) const {
            if (a < 0) {
                return omegas[buff][(a + lat_len) * (lat_len + vis_len) + (d + lat_len)];
            } else
                return 0.0;
        }

        public:
        __device__ __forceinline__ void set_omega(int32_t a, int32_t d, scalar_t val, int8_t buff) {
            if (a < 0)
                omegas[buff][(a + lat_len) * (lat_len + vis_len) + (d + lat_len)] = val;
        }

        public:
        __device__ __forceinline__ scalar_t get_covariance_grad(int32_t u, int32_t v, int8_t buff) const {
            if (u >= 0 || v >= 0) {
                return covariance_grads[buff][(min(u, v) + lat_len) * vis_len + max(u, v)];
            } else
                return 0.0;
        }

        public:
        __device__ __forceinline__ void set_covariance_grad(int32_t u, int32_t v, scalar_t val, int8_t buff) {
            if (u >= 0 || v >= 0)
                covariance_grads[buff][(min(u, v) + lat_len) * vis_len + max(u, v)] = val;
        }

        public:
        __device__ __forceinline__ scalar_t get_lambda(int32_t np, int32_t nc) const {
            if (np == nc)
                return get_covariance(np, nc);
            else if (nc >= 0)
                return lambda[(np + lat_len) * vis_len + nc];
            else
                return 0.0;
        }

        public:
        __device__ __forceinline__ scalar_t set_lambda(int32_t np, int32_t nc, scalar_t val) {
            lambda[(np + lat_len) * vis_len + nc] = val;
        }

        public:
        __device__ __forceinline__ int32_t get_num_parents(int32_t v, const LayerData& layer) const {
            if (v >= layer.base)
                return parents_bases[v + 1] - parents_bases[v];
            else if (v >= 0)
                return 1;
            else {
                const int32_t base = (lat_len + v) * 2;
                return (lat_range[base] < layer.idx && layer.idx <= lat_range[base + 1]) ? 1 : 0;
                // ^ only after the presence range the latent variable has parents until after its presence
            }
        }

        public:
        __device__ __forceinline__ int32_t get_parent(int32_t v, int32_t idx, const LayerData& layer) const {
            if (v >= layer.base)
                return parents[parents_bases[v] + idx];
            else
                return v;
        }

        public:
        /**
         * @param v the node in layer `layer`
         * @param begin the index of the first child is stored in it. Either -1 (the node parets itself) or 0 (otherwise)
         * @param end the index after the last child. Number of children in the current layer (except for itself)
         */
        __device__ __forceinline__ void get_children_range(int32_t v, int32_t& begin, int32_t& end, const LayerData& layer) const {
            const int32_t idx = (v + lat_len) * num_layers + layer.idx;
            end = children_bases[idx + 1] - children_bases[idx];

            if (v >= 0)
                begin = -1;
            else {
                const int32_t base = (v + lat_len) * 2;
                begin = (lat_range[base] <= layer.idx && layer.idx < lat_range[base + 1]) ? -1 : 0;
            }
        }

        public:
        /**
         * Get the `idx`-th child of `v` in layer `layer`; also get `v` if `idx == -1`.
         */
        __device__ __forceinline__ int32_t get_child(int32_t v, int32_t idx, const LayerData& layer) const {
            if (idx == -1)
                return v;
            else {
                const int32_t base_idx = (v + lat_len) * num_layers + layer.idx;
                return children[children_bases[base_idx] + idx];
            }
        }

        public:
        __device__ __forceinline__ int32_t get_layer_var(int32_t idx, const LayerData& layer) const {
            if (idx >= layer.lat_width)
                return idx - layer.lat_width;
            else
                return lat_neighbors[lat_neighbors_bases[layer.idx] + idx];
        }

        public:
        __host__ __device__ __forceinline__ int32_t get_vis_len() const {
            return vis_len;
        }

        public:
        __host__ __device__ __forceinline__ int32_t get_lat_len() const {
            return lat_len;
        }

#else
        public:
        DeviceData(
                const bool* const structure,
                scalar_t* const lambda,
                scalar_t* const weights,
                scalar_t* const covariance,
                scalar_t* const weights_grad,
                scalar_t* const w_accum,
                scalar_t* const covariance_grads,
                scalar_t* const omegas,
                const int32_t* const parents,
                const int32_t* const parents_bases,
                const int32_t* const children,
                const int32_t* const children_bases,
                const int32_t* const lat_neighbors,
                const int32_t* const lat_neighbors_bases,
                const int32_t* const lat_range,
                const int32_t vis_len,
                const int32_t lat_len,
                const int32_t num_layers
        ):  structure(structure),
            lambda(lambda),
            weights(weights),
            covariance(covariance),
            weights_grad(weights_grad),
            w_accum(w_accum),
            covariance_grads{
                covariance_grads,
                covariance_grads ? covariance_grads + ((lat_len + vis_len) * vis_len) : nullptr
            },
            omegas{
                omegas,
                omegas ? omegas + (lat_len * (lat_len + vis_len)) : nullptr
            },
            parents(parents),
            parents_bases(parents_bases),
            children(children),
            children_bases(children_bases),
            lat_neighbors(lat_neighbors),
            lat_neighbors_bases(lat_neighbors_bases),
            lat_range(lat_range),
            vis_len(vis_len),
            lat_len(lat_len),
            num_layers(num_layers)
        { }
#endif
    };
}

#endif