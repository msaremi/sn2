#ifndef SEMNAN_SOLVER_H
#define SEMNAN_SOLVER_H

#include <torch/extension.h>
#include "device_data.h"
#include "declarations.h"
#include "semnan_solver_loss.h"
#include <math_constants.h>
#include <stddef.h>
#include <vector>
#include <set>
#include <variant>
#include <iostream>
#include <optional>
#include <utility>

// Declarations
template <typename scalar_t>
void semnan_cuda_forward(
        const std::vector<SEMNANLayerData>& layers_vec,
        SEMNANDeviceData<scalar_t>& data
);

template <typename scalar_t>
void semnan_cuda_backward(
        const std::vector<SEMNANLayerData>& layers_vec,
        SEMNANDeviceData<scalar_t>& data
);

template <typename scalar_t>
void semnan_cuda_lv_transformation(
        const std::vector<SEMNANLayerData>& layers_vec,
        SEMNANDeviceData<scalar_t>& data
);

extern template class SEMNANDeviceData<float>;
extern template class SEMNANDeviceData<double>;

namespace semnan_cuda {
    using namespace torch::indexing;
    using namespace semnan_cuda::loss;

    // Class SEMNANSolver
    class SEMNANSolver {
        torch::Tensor structure;                // We keep all the tensors alive for the lifetime of SEMNANSolver
        torch::Tensor lambda;
        torch::Tensor weights;
        torch::Tensor covariance;
        torch::Tensor visible_covariance;
        torch::Tensor parents;                  // Information regarding the SEMNAN structure
        torch::Tensor parents_bases;            // ^
        torch::Tensor children;                 // ^
        torch::Tensor children_bases;           // ^
        torch::Tensor latent_neighbors;         // ^
        torch::Tensor latent_neighbors_bases;   // ^
        torch::Tensor latent_presence_range;    // ^
        std::vector<SEMNANLayerData> layers_vec;// ^
        std::variant<
                std::monostate,
                SEMNANDeviceData<float>,
                SEMNANDeviceData<double>
        > data;

        int32_t visible_size;                   // Number of visible variables (|V|)
        int32_t latent_size;                    // Number of latent variables (|V| + |L|)
        c10::DeviceIndex cuda_device_number;
        torch::Dtype dtype;
        bool check;

        std::shared_ptr<LossBase> loss_function;

        private:
        inline int32_t num_layers() {
            return this->layers_vec.size();
        }

        private:
        void init_parameters(
                const torch::Tensor& structure,
                const torch::Tensor& parameters,
                torch::Dtype dtype,
                std::shared_ptr<LossBase> loss_function,
                bool check
        ) {
            this->visible_size = structure.size(1);
            this->latent_size = structure.size(0) - visible_size;
            this->structure = structure;
            this->dtype = dtype;
            this->check = check;
            const int32_t total_size = latent_size + visible_size;

            TORCH_CHECK(structure.is_cuda(), "`structure` must be a CUDA tensor.");
            TORCH_CHECK(structure.dim() == 2, "`structure` must be 2-dimensional; it is ", structure.dim(), "-dimensional.");
            TORCH_CHECK(structure.numel() > 0, "`structure` needs at least one element.");
            TORCH_CHECK(latent_size >= 0, "`structure` must be a vertical-rectangular matrix.");
            TORCH_CHECK(dtype == torch::kFloat || dtype == torch::kDouble, "`dtype` must be either float or double.");
            TORCH_CHECK(!parameters.defined() || parameters.sizes() == structure.sizes(), "`parameters` must have the same size as `structure`.");

            if (check) {
                auto latent_structure = structure.index({Slice(None, this->latent_size), Slice()});
                auto visible_structure = structure.index({Slice(this->latent_size, None), Slice()});

                TORCH_CHECK(
                    latent_structure.any(0).all(0).item<bool>(),
                    "All visible variables must be connected to at least one latent variable."
                );

                TORCH_CHECK(
                    !torch::tril(visible_structure).any().item<bool>(),
                    "Visible space must be an upper-triangular matrix."
                );

                if (latent_structure.any(1).logical_not().any(0).item<bool>())
                    TORCH_WARN_ONCE("There are loose variables in the latent space.");
            }

            const bool parameters_exist = parameters.defined();
            const auto& base = parameters_exist ? parameters : structure;

            this->cuda_device_number = base.device().is_cuda() ? base.device().index() : -1;
            torch::TensorOptions options = torch::TensorOptions()
                                                  .dtype(dtype)
                                                  .device(torch::kCUDA, this->cuda_device_number)
                                                  .requires_grad(false);

            this->weights = base.to(options);
            this->weights *= parameters_exist ?
                                 this->structure :
                                 torch::randn_like(this->weights, options);

            this->covariance = torch::zeros_like(this->weights, options);
            this->lambda = torch::zeros_like(this->weights, options);
            this->visible_covariance = this->covariance.index({
                Slice(this->latent_size, None),
                Slice()
            });

            this->weights.mutable_grad() = torch::zeros_like(this->weights, options);
            this->covariance.mutable_grad() = torch::zeros({2, this->covariance.size(0), this->covariance.size(1)}, options);
            this->visible_covariance.mutable_grad() = this->covariance.mutable_grad().index({
                Slice(None),
                Slice(this->latent_size, None),
                Slice(None)
            });

            this->loss_function = loss_function ? loss_function : std::make_shared<KullbackLeibler>();
        }

        private:
        void make_structures() {
            const int32_t total_size = latent_size + visible_size;
            torch::TensorOptions options = torch::TensorOptions()
                                                  .dtype(torch::kInt32)
                                                  .device(torch::kCUDA, this->cuda_device_number)
                                                  .requires_grad(false);

            int32_t edge_count = 0, rev_edge_count = 0;
            std::vector<std::vector<int32_t>> parents_vec(this->visible_size);
            std::vector<std::vector<std::vector<int32_t>>> children_vec({std::vector<std::vector<int32_t>>(total_size)});
            this->latent_presence_range = torch::full({latent_size, 2}, -1, options);
            this->layers_vec.resize(1);
            this->layers_vec.push_back(SEMNANLayerData(1, 0));

            for (int32_t c = 0, layer_max = 0; c < visible_size; c++) {
                for (int32_t p = -latent_size; p < visible_size; p++)
                    if (structure[p + latent_size][c].item<bool>()) {
                        parents_vec[c].push_back(p); // Add to the parents of the current child

                        // If parent does not belong to previous layer, make a new layer
                        if (p >= layer_max) {
                            layer_max = c;
                            this->layers_vec.push_back(SEMNANLayerData(layers_vec.size(), c));
                            children_vec.push_back(std::vector<std::vector<int32_t>>(total_size));
                        }

                        edge_count++;
                    }

                for (int32_t p = -latent_size; p < visible_size; p++) {
                    if (structure[p + latent_size][c].item<bool>()) {
                        if (p < 0) {
                            auto& this_latent = this->latent_presence_range[p + latent_size];
                            this_latent.index_put_(
                                {Slice(this_latent[0].item() == -1 ? 0 : 1, 2)},
                                layers_vec.back().idx - 1
                            );
                        }

                        children_vec.back()[p + latent_size].push_back(c); // Add to the children of the current parent
                    }
                }

                this->layers_vec.back().num++;
            }

            // Create the parents data
            this->parents = torch::zeros(edge_count, options);
            this->parents_bases = torch::zeros(this->visible_size + 1, options);

            for (int32_t c = 0, idx = 0; c < this->visible_size; ) {
                for (int32_t p : parents_vec[c])
                    this->parents[idx++] = p;

                this->parents_bases[++c] = idx;
            }

            // Create the children data
            this->children = torch::zeros(edge_count, options);
            this->children_bases = torch::zeros({total_size, this->num_layers()}, options);

            for (int32_t p = -latent_size, idx = 0; p < visible_size; p++) {
                this->children_bases.index_put_({p + latent_size, 0}, idx);

                for (int32_t l = 0; l < children_vec.size(); ) {
                    for (int32_t c : children_vec[l][p + latent_size])
                        this->children[idx++] = c;

                    this->children_bases.index_put_({p + latent_size, ++l}, idx);
                }
            }

            int32_t latent_neighbors_count = 0;
            std::vector<std::vector<int32_t>> latent_neighbors_vec(this->num_layers());

            for (int32_t v = -this->latent_size; v < 0; v++) {
                auto& this_latent = this->latent_presence_range[v + latent_size];

                // `l >= 0` takes care of "loose" latent variables (those with no children)
                for (int32_t l = this_latent[0].item<int32_t>(); l <= this_latent[1].item<int32_t>() && l >= 0; l++) {
                    latent_neighbors_vec[l].push_back(v);
                    latent_neighbors_count++;
                }
            }

            // Create the latent neighbors data
            this->latent_neighbors = torch::zeros(latent_neighbors_count, options);
            this->latent_neighbors_bases = torch::zeros(latent_neighbors_vec.size() + 1, options);

            for (int32_t l = 0, idx = 0; l < latent_neighbors_vec.size(); ) {
                for (int32_t i : latent_neighbors_vec[l])
                    this->latent_neighbors[idx++] = i;

                this->layers_vec[l].lat_width = latent_neighbors_vec[l].size();
                this->latent_neighbors_bases[++l] = idx;
            }
        }

        public:
        SEMNANSolver(
                torch::Tensor& structure,
                torch::Tensor& parameters = torch::Tensor(),
                torch::Dtype dtype = torch::kFloat,
                std::shared_ptr<LossBase> loss_function = nullptr,
                bool check = true
        ) {
            this->init_parameters(structure, parameters, dtype, loss_function, check);
            this->make_structures();

            AT_DISPATCH_FLOATING_TYPES(dtype, "SEMNANDeviceData::init", ([&] {
                scalar_t* const covariance_grads[2] = {
                    covariance.mutable_grad()[0].data_ptr<scalar_t>(),
                    covariance.mutable_grad()[1].data_ptr<scalar_t>()
                };

                this->data = SEMNANDeviceData<scalar_t>(
                    structure.data_ptr<bool>(),
                    lambda.data_ptr<scalar_t>(),
                    weights.data_ptr<scalar_t>(),
                    covariance.data_ptr<scalar_t>(),
                    weights.mutable_grad().data_ptr<scalar_t>(),
                    covariance_grads,
                    nullptr,

                    parents.data_ptr<int32_t>(),
                    parents_bases.data_ptr<int32_t>(),
                    children.data_ptr<int32_t>(),
                    children_bases.data_ptr<int32_t>(),

                    latent_neighbors.data_ptr<int32_t>(),
                    latent_neighbors_bases.data_ptr<int32_t>(),
                    latent_presence_range.data_ptr<int32_t>(),
                    visible_size,
                    latent_size,
                    num_layers()
                );
            }));
        }

        public:
        torch::Tensor loss() {
            return loss_function->loss(visible_covariance);
        }

        public:
        torch::Tensor loss_proxy() {
            return loss_function->loss_proxy(visible_covariance);
        }

        private:
        void loss_backward(torch::Tensor& visible_covariance_grad) {
            loss_function->loss_backward(visible_covariance, visible_covariance_grad);
        }

        public:
        torch::Tensor get_lv_transformation() {
            torch::Tensor lv_transformation = torch::zeros({latent_size, visible_size}, weights.options());
            AT_DISPATCH_FLOATING_TYPES(dtype, "SEMNANDeviceData::lv_transformation", ([&] {
                std::get<SEMNANDeviceData<scalar_t>>(this->data).set_lv_transformation(lv_transformation.data_ptr<scalar_t>());
                semnan_cuda_lv_transformation<scalar_t>(
                    this->layers_vec, std::get<SEMNANDeviceData<scalar_t>>(this->data)
                );
                std::get<SEMNANDeviceData<scalar_t>>(this->data).set_lv_transformation(nullptr);
            }));
            return lv_transformation;
        }

        public:
        void set_sample_covariance(const torch::Tensor& sample_covariance) {
            TORCH_CHECK(sample_covariance.size(0) == visible_size, "`sample_covariance` must be a ", visible_size, "×", visible_size, " matrix.");
            this->loss_function->set_sample_covariance(sample_covariance.to(this->weights.options()));
        }

        public:
        torch::Tensor& get_sample_covariance() {
           return this->loss_function->get_sample_covariance();
        }

        public:
        void forward() {
            AT_DISPATCH_FLOATING_TYPES(dtype, "SEMNANDeviceData::forward", ([&] {
                semnan_cuda_forward<scalar_t>(
                        this->layers_vec, std::get<SEMNANDeviceData<scalar_t>>(this->data)
                );
            }));
        }

        public:
        void backward() {
            AT_DISPATCH_FLOATING_TYPES(dtype, "SEMNANDeviceData::backward", ([&] {
                loss_backward(visible_covariance.mutable_grad()[(this->num_layers() + 1) % 2]);
                semnan_cuda_backward<scalar_t>(
                        this->layers_vec, std::get<SEMNANDeviceData<scalar_t>>(this->data)
                );
            }));
        }

        public:
        torch::Tensor& get_weights() {
            return this->weights;
        }

        public:
        void set_weights(torch::Tensor& weights) {
            this->weights.copy_(weights);
        }

        public:
        torch::Tensor& get_covariance() {
            return this->covariance;
        }

        public:
        torch::Tensor& get_visible_covariance() {
            return this->visible_covariance;
        }

        public:
        torch::Tensor& get_lambda() {
            return this->lambda;
        }
    };
}

#endif