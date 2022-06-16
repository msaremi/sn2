#ifndef DEVICE_DATA_H
#define DEVICE_DATA_H
#include <torch/extension.h>
#include <stdio.h>

/*
 * A declaration for the cpp compiler and a definition for the nvcc compiler.
 */

// Stores the structural data for a layer in semnan
class SEMNANLayerData {
    public:
    int32_t idx;
    int32_t base;
    int32_t num;
    int32_t lat_width;

#ifdef __CUDACC__
    public:
    SEMNANLayerData(
        const int32_t idx,
        const int32_t base
    );

    public:
    SEMNANLayerData();

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
    SEMNANLayerData(
        const int32_t idx,
        const int32_t base
    ):  idx(idx),
        base(base),
        num(0),
        lat_width(0)
    { }

    public:
    SEMNANLayerData()
    :   idx(0),
        base(0),
        num(0),
        lat_width(0)
    { }
#endif
};

// Stores the lambda, weights, covariance matrices along with
// the weights_grad and the two covariance_grads buffers.
// It provides a view to get/set elements in the matrices.
template <typename scalar_t>
class SEMNANDeviceData {
    private:
    const bool* structure;
    scalar_t* lambda;
    scalar_t* weights;
    scalar_t* covariance;
    scalar_t* weights_grad;
    scalar_t* covariance_grads[2];
    scalar_t* lv_transformation;
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
    __device__ __forceinline__ scalar_t get_lv_transformation(int32_t a, int32_t d) const {
        if (a == d)
            return 1.0;
        else
            return (a < d) ? lv_transformation[(a + lat_len) * vis_len + d] : 0.0;
    }

    public:
    __device__ __forceinline__ void set_lv_transformation(int32_t a, int32_t d, scalar_t val) {
        if (a <= d)
            lv_transformation[(a + lat_len) * vis_len + d] = val;
    }

    public:
    __device__ __forceinline__ scalar_t get_weight(int32_t p, int32_t c, const SEMNANLayerData& layer) const {
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
    __device__ __forceinline__ scalar_t get_covariance_grad(int32_t u, int32_t v, int8_t buff) const {
        if (u >= 0 || v >= 0) {
            return covariance_grads[buff][(min(u, v) + lat_len) * vis_len + max(u, v)];
        } else
            return 0.0;
    }

    public:
    __device__ __forceinline__ void set_covariance_grad(int32_t u, int32_t v, scalar_t val, int8_t buff) {
        if (u >= 0 || v >= 0) {
            covariance_grads[buff][(min(u, v) + lat_len) * vis_len + max(u, v)] = val;
        }
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
    __device__ __forceinline__ int32_t get_num_parents(int32_t v, const SEMNANLayerData& layer) const {
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
    __device__ __forceinline__ int32_t get_parent(int32_t v, int32_t idx, const SEMNANLayerData& layer) const {
        if (v >= layer.base)
            return parents[parents_bases[v] + idx];
        else
            return v;
    }

    public:
    __device__ __forceinline__ int32_t get_children_range(int32_t v, int32_t& begin, int32_t& end, const SEMNANLayerData& layer) const {
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
    __device__ __forceinline__ int32_t get_child(int32_t v, int32_t idx, const SEMNANLayerData& layer) const {
        if (idx == -1)
            return v;
        else {
            const int32_t base_idx = (v + lat_len) * num_layers + layer.idx;
            return children[children_bases[base_idx] + idx];
        }
    }

    public:
    __device__ __forceinline__ int32_t get_layer_var(int32_t idx, const SEMNANLayerData& layer) const {
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
    SEMNANDeviceData(
            const bool* const structure,
            scalar_t* const lambda,
            scalar_t* const weights,
            scalar_t* const covariance,
            scalar_t* const weights_grad,
            scalar_t* const covariance_grads[2],
            scalar_t* const lv_transformation,
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
        covariance_grads{covariance_grads[0], covariance_grads[1]},
        lv_transformation(lv_transformation),
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

    void set_lv_transformation(scalar_t* const value) { lv_transformation = value; }
    scalar_t* get_lv_transformation() const { return lv_transformation; }
#endif
};

#endif