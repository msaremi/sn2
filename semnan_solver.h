#ifndef SEMNAN_SOLVER_H
#define SEMNAN_SOLVER_H

#include <torch/extension.h>
#include "stringify.h"
#include "device_data.h"
#include "declarations.h"
#include "semnan_solver_loss.h"
#include <stddef.h>
#include <vector>
#include <set>
#include <variant>
#include <iostream>
#include <optional>
#include <utility>

namespace semnan_cuda {
    using namespace torch::indexing;
    using namespace semnan_cuda::loss;

    // Declarations
    namespace covar {
        template <typename scalar_t>
        void forward(
                const std::vector<LayerData>& layers_vec,
                DeviceData<scalar_t>& data
        );

        template <typename scalar_t>
        void backward(
                const std::vector<LayerData>& layers_vec,
                DeviceData<scalar_t>& data
        );
    }

    namespace accum {
        template <typename scalar_t>
        void forward(
                const std::vector<LayerData>& layers_vec,
                DeviceData<scalar_t>& data
        );

        template <typename scalar_t>
        void backward(
                const std::vector<LayerData>& layers_vec,
                DeviceData<scalar_t>& data
        );
    }

    extern template class DeviceData<float>;
    extern template class DeviceData<double>;

    // Class SEMNANSolver
    class SEMNANSolver {
        public:
        enum struct METHOD {
            COVAR = 0,
            ACCUM
        };

        private:
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
        std::vector<LayerData> layers_vec;// ^
        torch::Tensor weights_accum;
        torch::Tensor omegas;
        std::variant<
                std::monostate,
                DeviceData<float>,
                DeviceData<double>
        > data;

        int32_t visible_size;                   // Number of visible variables (|V|)
        int32_t latent_size;                    // Number of latent variables (|V| + |L|)
        c10::DeviceIndex cuda_device_number;
        torch::Dtype dtype;
        bool validate;
        METHOD method;
        void (SEMNANSolver::*forward_method)(void);
        void (SEMNANSolver::*backward_method)(void);

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
                METHOD method,
                bool validate
        ) {
            this->visible_size = structure.size(1);
            this->latent_size = structure.size(0) - visible_size;
            this->structure = structure;
            this->dtype = dtype;
            this->method = method;
            this->validate = validate;
            const int32_t total_size = latent_size + visible_size;

            TORCH_CHECK(structure.is_cuda(), STRINGIFY(structure) " must be a CUDA tensor.");
            TORCH_CHECK(structure.dim() == 2, STRINGIFY(structure) " must be 2-dimensional; it is ", structure.dim(), "-dimensional.");
            TORCH_CHECK(structure.numel() > 0, STRINGIFY(structure) " needs at least one element.");
            TORCH_CHECK(latent_size >= 0, STRINGIFY(structure) " must be a vertical-rectangular matrix.");
            TORCH_CHECK(dtype == torch::kFloat || dtype == torch::kDouble, STRINGIFY(dtype) " must be either " STRINGIFY(torch::kFloat) " or " STRINGIFY(torch::kDouble) ".");
            TORCH_CHECK(!parameters.defined() || parameters.sizes() == structure.sizes(), STRINGIFY(parameters) " must be of the same size as " STRINGIFY(structure) ".");

            if (validate) {
                auto latent_structure = structure.index({Slice(None, this->latent_size), Slice()});
                auto visible_structure = structure.index({Slice(this->latent_size, None), Slice()});

                // TODO: They need to be connected to at least one variable (latent or visible)
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
            this->weights *= parameters_exist ? this->structure : torch::randn_like(this->weights, options);
            this->weights.mutable_grad() = torch::zeros_like(this->weights, options);

            switch (method) {
                case METHOD::COVAR:
                    this->lambda = torch::zeros_like(this->weights, options);
                    this->covariance = torch::zeros_like(this->weights, options);
                    this->covariance.mutable_grad() = torch::zeros({2, total_size, visible_size}, options);
                    this->visible_covariance = this->covariance.index({ Slice(latent_size, None), Slice() });
                    this->visible_covariance.mutable_grad() = this->covariance.mutable_grad().index({ Slice(), Slice(latent_size, None), Slice() });

                    this->forward_method = &SEMNANSolver::forward_covar;
                    this->backward_method = &SEMNANSolver::backward_covar;
                    break;

                case METHOD::ACCUM:
                    this->visible_covariance = torch::zeros({visible_size, visible_size}, options);
                    this->visible_covariance.mutable_grad() = torch::zeros_like(this->visible_covariance, options);
                    this->weights_accum = torch::zeros({latent_size, visible_size}, options);
                    this->omegas = torch::zeros({2, total_size, visible_size}, options);

                    this->forward_method = &SEMNANSolver::forward_accum;
                    this->backward_method = &SEMNANSolver::backward_accum;
                    break;
            }

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
            this->layers_vec.push_back(LayerData(1, 0));

            for (int32_t c = 0, layer_max = 0; c < visible_size; c++) {
                for (int32_t p = -latent_size; p < visible_size; p++)
                    if (structure[p + latent_size][c].item<bool>()) {
                        parents_vec[c].push_back(p); // Add to the parents of the current child

                        // If parent does not belong to previous layer, make a new layer
                        if (p >= layer_max) {
                            layer_max = c;
                            this->layers_vec.push_back(LayerData(layers_vec.size(), c));
                            children_vec.push_back(std::vector<std::vector<int32_t>>(total_size));
                        }

                        edge_count++;
                    }

                for (int32_t p = -latent_size; p < visible_size; p++) {
                    if (structure[p + latent_size][c].item<bool>()) {
                        if (p < 0) {
                            auto this_latent = this->latent_presence_range[p + latent_size];
                            this_latent.index_put_(
                                {Slice(this_latent[0].item<int32_t>() == -1 ? 0 : 1, 2)},
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
                auto const this_latent = this->latent_presence_range[v + latent_size];

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

        private:
        /**
         * Returns the data_ptr of a (sub-)tensor; returns nullptr if tensor is not defined.
         * @param tensor the tensor from which the data_ptr is taken
         * @param index the index for the sub-tensor if applicable.
         * @return `data_ptr` of `tensor[index]`, or `nullpr` if tensor is not defined.
         */
        template <typename scalar_t>
        static inline scalar_t* try_get_data_ptr(torch::Tensor& tensor, std::optional<TensorIndex> index = std::nullopt) {
            if (tensor.defined())
                if (index.has_value())
                    return tensor.index(index.value()).data_ptr<scalar_t>();
                else
                    return tensor.data_ptr<scalar_t>();
            else
                return nullptr;
        }

        public:
        /**
         * SEMNANSolver constructor.
         * @param structure a vertical matrix of `bool` values indicating the structure of the AMASEM
         * @param parameters the initial parameters of the AMASEM
         * @param dtype the type of matrices used for calculations: `torch::kFloat` or `torch::kDouble`
         * @param loss_function any subclass of `LossBase`
         * @param method The method used for calculating the derivatives
         * @param validate Apply extra validations; set `false` to avoid unneccesary calculations
         */
        SEMNANSolver(
                torch::Tensor structure,
                torch::Tensor parameters = torch::Tensor(),
                torch::Dtype dtype = torch::kFloat,
                std::shared_ptr<LossBase> loss_function = nullptr,
                METHOD method = METHOD::COVAR,
                bool validate = true
        ) {
            this->init_parameters(structure, parameters, dtype, loss_function, method, validate);
            this->make_structures();

            AT_DISPATCH_FLOATING_TYPES(dtype, "DeviceData::init", ([&] {
                this->data = DeviceData<scalar_t>(
                        structure.data_ptr<bool>(),
                        try_get_data_ptr<scalar_t>(lambda),
                        try_get_data_ptr<scalar_t>(weights),
                        try_get_data_ptr<scalar_t>(covariance),
                        try_get_data_ptr<scalar_t>(weights.mutable_grad()),
                        try_get_data_ptr<scalar_t>(weights_accum),
                        try_get_data_ptr<scalar_t>(covariance.mutable_grad()),
                        try_get_data_ptr<scalar_t>(omegas),

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
        void loss_backward(torch::Tensor visible_covariance_grad) {
            loss_function->loss_backward(visible_covariance, visible_covariance_grad);
        }

        public:
        torch::Tensor get_lv_transformation() {
            TORCH_CHECK(this->method == METHOD::ACCUM,
                        STRINGIFY(weights_accum) " is not computed when " STRINGIFY(method) " is set to " STRINGIFY(METHOD::COVAR)
                        ". Consider initializing " STRINGIFY(SEMNANSolver) " with " STRINGIFY(METHOD::ACCUM) "."
            );
            return this->weights_accum;
        }

        public:
        void set_sample_covariance(const torch::Tensor& sample_covariance) {
            TORCH_CHECK(sample_covariance.size(0) == visible_size, STRINGIFY(sample_covariance) " must be a ", visible_size, "×", visible_size, " matrix.");
            this->loss_function->set_sample_covariance(sample_covariance.to(this->weights.options()));
        }

        public:
        torch::Tensor& get_sample_covariance() {
           return this->loss_function->get_sample_covariance();
        }

        private:
        void forward_accum() {
            AT_DISPATCH_FLOATING_TYPES(dtype, "SEMNANSolver::forward_accum", ([&] {
                accum::forward<scalar_t>(this->layers_vec, std::get<DeviceData<scalar_t>>(this->data));
                torch::matmul_out(visible_covariance, torch::transpose(weights_accum, 0, 1), weights_accum);
            }));
        }

        private:
        void forward_covar() {
            AT_DISPATCH_FLOATING_TYPES(dtype, "SEMNANSolver::forward_covar", ([&] {
                covar::forward<scalar_t>(this->layers_vec, std::get<DeviceData<scalar_t>>(this->data));
            }));
        }

        public:
        void forward() {
            (this->*forward_method)();
        }

        private:
        void backward_accum() {
            // FIXME: Implement method.
            TORCH_CHECK(false, STRINGIFY(backward_accum) " has not been implemented yet. Try using " STRINGIFY(backward_covar)  ".");
            AT_DISPATCH_FLOATING_TYPES(dtype, "SEMNANSolver::backward_accum", ([&] {
                accum::backward<scalar_t>(this->layers_vec, std::get<DeviceData<scalar_t>>(this->data));
            }));
        }

        private:
        void backward_covar() {
            AT_DISPATCH_FLOATING_TYPES(dtype, "SEMNANSolver::backward_covar", ([&] {
                covar::backward<scalar_t>(this->layers_vec, std::get<DeviceData<scalar_t>>(this->data));
            }));
        }

        public:
        void backward() {
            loss_backward(visible_covariance.mutable_grad()[(this->num_layers() + 1) % 2]);
            (this->*backward_method)();
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
            TORCH_CHECK(this->method == METHOD::COVAR,
                        STRINGIFY(covariance) " is not computed when " STRINGIFY(method) " is set to " STRINGIFY(METHOD::ACCUM)
                        ". Consider initializing " STRINGIFY(SEMNANSolver) " with " STRINGIFY(METHOD::COVAR) "."
            );
            return this->covariance;
        }

        public:
        torch::Tensor& get_visible_covariance() {
            return this->visible_covariance;
        }

        public:
        torch::Tensor& get_lambda() {
            TORCH_CHECK(this->method == METHOD::COVAR,
                        STRINGIFY(lambda) " is not computed when " STRINGIFY(method) " is set to " STRINGIFY(METHOD::ACCUM)
                        ". Consider initializing " STRINGIFY(SEMNANSolver) " with " STRINGIFY(METHOD::COVAR) "."
            );
            return this->lambda;
        }
    };
}

#endif