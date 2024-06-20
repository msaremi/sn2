#ifndef SN2_SOLVER_H
#define SN2_SOLVER_H

#include <torch/extension.h>
#include "stringify.h"
#include "device_data.h"
#include "declarations.h"
#include "sn2_solver_loss.h"
#include <stddef.h>
#include <vector>
#include <set>
#include <variant>
#include <iostream>
#include <optional>
#include <utility>

namespace sn2_cuda {
    using namespace torch::indexing;
    using namespace sn2_cuda::loss;

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

    // Class SN2Solver
    class SN2Solver {
        public:
        enum struct METHODS {
            COVAR = 0,
            ACCUM
        };

        private:
        torch::Tensor structure;                // We keep all the tensors alive for the lifetime of SN2Solver
        torch::Tensor lambda;
        torch::Tensor weights;
        torch::Tensor covariance;
        torch::Tensor visible_covariance;
        torch::Tensor parents;                  // Information regarding the SN2 structure
        torch::Tensor parents_bases;            // ^
        torch::Tensor children;                 // ^
        torch::Tensor children_bases;           // ^
        torch::Tensor latent_neighbors;         // ^
        torch::Tensor latent_neighbors_bases;   // ^
        torch::Tensor latent_presence_range;    // ^
        std::vector<LayerData> layers_vec;      // ^
        torch::Tensor weights_accum;
        torch::Tensor omegas;
        std::variant<
                std::monostate,
                DeviceData<float>,
                DeviceData<double>
        > data;

        int32_t visible_size;                   // Number of visible variables (|V|)
        int32_t latent_size;                    // Number of latent variables (|V| + |L|)
        torch::DeviceIndex cuda_device_number;
        torch::Dtype dtype;
        bool validate;
        METHODS method;
        void (SN2Solver::*forward_method)(void);
        void (SN2Solver::*backward_method)(void);

        std::shared_ptr<LossBase> loss_function;

        private:
        inline int32_t num_layers() {
            return this->layers_vec.size();
        }

        private:
        void init_parameters(
                const torch::Tensor& structure,
                const torch::Tensor& parameters,
                const torch::Tensor& sample_covariance,
                torch::Dtype dtype,
                std::shared_ptr<LossBase> loss_function,
                METHODS method,
                bool validate
        ) {
            const int32_t total_size = structure.size(0);
            this->visible_size = structure.size(1);
            this->latent_size = total_size - visible_size;
            this->dtype = dtype;
            this->method = method;
            this->validate = validate;

            TORCH_CHECK(torch::cuda::is_available(), "CUDA is not available. " STRINGIFY(SN2Solver) " needs CUDA to run.")
            TORCH_CHECK(structure.dim() == 2, STRINGIFY(structure) " must be 2-dimensional; it is ", structure.dim(), "-dimensional.")
            TORCH_CHECK(structure.numel() > 0, STRINGIFY(structure) " needs at least one element.")
            TORCH_CHECK(latent_size >= 0, STRINGIFY(structure) " must be a vertical-rectangular matrix.")
            TORCH_CHECK(dtype == torch::kFloat || dtype == torch::kDouble, STRINGIFY(dtype) " must be either " STRINGIFY(torch::kFloat) " or " STRINGIFY(torch::kDouble) ".")
            TORCH_CHECK(!parameters.defined() || parameters.sizes() == structure.sizes(), STRINGIFY(parameters) " must be of the same size as " STRINGIFY(structure) ".")

            this->cuda_device_number = structure.device().is_cuda() ? structure.device().index() : -1;
            const bool parameters_exist = parameters.defined();
            const auto& base = parameters_exist ? parameters : structure;
            torch::TensorOptions options = torch::TensorOptions()
                    .dtype(dtype)
                    .device(torch::kCUDA, this->cuda_device_number)
                    .requires_grad(false);
            this->structure = structure.to(options.dtype(torch::kBool));

            if (validate) {
                auto&& latent_structure = this->structure.index({Slice(None, this->latent_size), Slice()});
                auto&& visible_structure = this->structure.index({Slice(this->latent_size, None), Slice()});

                // TODO: They need to be connected to at least one variable (latent or visible)
                TORCH_CHECK(
                        latent_structure.any(0).all(0).item<bool>(),
                        "All visible variables must be connected to at least one latent variable."
                )

                TORCH_CHECK(
                        !torch::tril(visible_structure).any().item<bool>(),
                        "Visible space must be an upper-triangular matrix."
                )

                if (latent_structure.any(1).logical_not().any(0).item<bool>()) {
                    TORCH_WARN_ONCE("There are loose variables in the latent space.")
                }
            }

            this->weights = base.to(options);
            this->weights *= parameters_exist ? this->structure : torch::randn_like(this->weights);
            this->weights.mutable_grad() = torch::zeros_like(this->weights);
            this->loss_function = loss_function ? loss_function : std::make_shared<KullbackLeibler>();

            if (sample_covariance.defined()) {
                this->set_sample_covariance(sample_covariance);
            }

            switch (method) {
                case METHODS::COVAR:
                    this->lambda = torch::zeros_like(this->weights);
                    this->covariance = torch::zeros_like(this->weights);
                    this->covariance.mutable_grad() = torch::zeros({2, total_size, visible_size}, options);
                    this->visible_covariance = this->covariance.index({ Slice(latent_size, None), Slice() });
                    this->visible_covariance.mutable_grad() = this->covariance.mutable_grad().index({ Slice(), Slice(latent_size, None), Slice() });

                    this->forward_method = &SN2Solver::forward_covar;
                    this->backward_method = &SN2Solver::backward_covar;
                    break;

                case METHODS::ACCUM:
                    this->visible_covariance = torch::zeros({visible_size, visible_size}, options);
                    this->visible_covariance.mutable_grad() = torch::zeros_like(this->visible_covariance);
                    this->weights_accum = torch::zeros({latent_size, visible_size}, options);
                    this->omegas = torch::zeros({2, latent_size, total_size}, options);

                    this->forward_method = &SN2Solver::forward_accum;
                    this->backward_method = &SN2Solver::backward_accum;
                    break;
            }
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
                            auto&& this_latent = this->latent_presence_range[p + latent_size];
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
                const auto&& this_latent = this->latent_presence_range[v + latent_size];

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
        inline void init_data() {
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
         * SN2Solver constructor.
         * @param structure a vertical matrix of `bool` values indicating the structure of the AMASEM
         * @param parameters the initial parameters of the AMASEM
         * @param dtype the type of matrices used for calculations: `torch::kFloat` or `torch::kDouble`
         * @param loss_function any subclass of `LossBase`
         * @param method The method used for calculating the derivatives
         * @param validate Apply extra validations; set `false` to avoid unneccesary calculations
         */
        SN2Solver(
                torch::Tensor structure,
                std::optional<torch::Tensor> parameters = std::nullopt,
                std::optional<torch::Tensor> sample_covariance = std::nullopt,
                torch::Dtype dtype = torch::kFloat,
                std::shared_ptr<LossBase> loss_function = nullptr,
                METHODS method = METHODS::COVAR,
                bool validate = true
        ) {
            this->init_parameters(
                    structure,
                    parameters.has_value() ? parameters.value() : torch::Tensor(),
                    sample_covariance.has_value() ? sample_covariance.value() : torch::Tensor(),
                    dtype, loss_function, method, validate
            );

            this->make_structures();
            this->init_data();
        }

        public:
        inline torch::Tensor loss() {
            loss_function->check_has_sample_covariance();
            return loss_function->loss(visible_covariance);
        }

        public:
        inline torch::Tensor loss_proxy() {
            loss_function->check_has_sample_covariance();
            return loss_function->loss_proxy(visible_covariance);
        }

        private:
        inline void loss_backward(torch::Tensor&& visible_covariance_grad) {
            loss_function->check_has_sample_covariance();
            loss_function->loss_backward(visible_covariance, visible_covariance_grad);
        }

        private:
        inline torch::Tensor get_output_covariance_grad() {
            return this->method == METHODS::COVAR ?
                   visible_covariance.mutable_grad()[(this->num_layers() + 1) % 2] :
                   visible_covariance.mutable_grad();
        }

        private:
        inline torch::Tensor get_output_omega() {
//            return omegas[(this->num_layers() + 1) % 2];
            return omegas.index({(this->num_layers() + 1) % 2, Slice(), Slice(latent_size, None)});
        }

        public:
        torch::Tensor get_lv_transformation() {
            TORCH_CHECK(this->method == METHODS::ACCUM,
                        STRINGIFY(weights_accum) " is not computed when " STRINGIFY(method) " is set to " STRINGIFY(METHODS::COVAR)
                        ". Consider initializing " STRINGIFY(SN2Solver) " with " STRINGIFY(METHODS::ACCUM) ".")
            return this->weights_accum;
        }

        private:
        inline bool has_sample_covariance() {
            return this->loss_function->has_sample_covariance();
        }

        public:
        void set_sample_covariance(const torch::Tensor& sample_covariance) {
            TORCH_CHECK(sample_covariance.size(0) == visible_size, STRINGIFY(sample_covariance) " must be a ", visible_size, "×", visible_size, " matrix.")
            this->loss_function->set_sample_covariance(sample_covariance.to(this->weights.options()));
        }

        public:
        const torch::Tensor& get_sample_covariance() const {
           return this->loss_function->get_sample_covariance();
        }

        private:
        void forward_accum() {
            AT_DISPATCH_FLOATING_TYPES(dtype, "SN2Solver::forward_accum", ([&] {
                accum::forward<scalar_t>(this->layers_vec, std::get<DeviceData<scalar_t>>(this->data));
                torch::matmul_out(visible_covariance, torch::transpose(weights_accum, 0, 1), weights_accum);
            }));
        }

        private:
        void forward_covar() {
            AT_DISPATCH_FLOATING_TYPES(dtype, "SN2Solver::forward_covar", ([&] {
                covar::forward<scalar_t>(this->layers_vec, std::get<DeviceData<scalar_t>>(this->data));
            }));
        }

        public:
        void forward() {
            (this->*forward_method)();
        }

        private:
        void backward_accum() {
            torch::Tensor&& output_omega = get_output_omega();
            torch::matmul_out(output_omega, weights_accum, get_output_covariance_grad());
            AT_DISPATCH_FLOATING_TYPES(dtype, "SN2Solver::backward_accum", ([&] {
                accum::backward<scalar_t>(this->layers_vec, std::get<DeviceData<scalar_t>>(this->data));
            }));
        }

        private:
        void backward_covar() {
            AT_DISPATCH_FLOATING_TYPES(dtype, "SN2Solver::backward_covar", ([&] {
                covar::backward<scalar_t>(this->layers_vec, std::get<DeviceData<scalar_t>>(this->data));
            }));
        }

        public:
        void backward() {
            loss_backward(get_output_covariance_grad());
            (this->*backward_method)();
        }

        public:
        torch::Tensor& get_weights() {
            return this->weights;
        }

        public:
        void set_weights(const torch::Tensor& weights) {
            this->weights.copy_(weights);
        }

        public:
        torch::Tensor& get_covariance() {
            TORCH_CHECK(this->method == METHODS::COVAR,
                        STRINGIFY(covariance) " is not computed when " STRINGIFY(method) " is set to " STRINGIFY(METHODS::ACCUM)
                        ". Consider using " STRINGIFY(visible_covariance) " or initializing " STRINGIFY(SN2Solver)
                        " with " STRINGIFY(method=METHODS::COVAR) ".")
            return this->covariance;
        }

        public:
        torch::Tensor& get_visible_covariance() {
            return this->visible_covariance;
        }

        public:
        torch::Tensor& get_lambda() {
            TORCH_CHECK(this->method == METHODS::COVAR,
                        STRINGIFY(lambda) " is not computed when " STRINGIFY(method) " is set to " STRINGIFY(METHODS::ACCUM)
                        ". Consider initializing " STRINGIFY(SN2Solver) " with " STRINGIFY(method=METHODS::COVAR) ".")
            return this->lambda;
        }

        public:
        torch::Tensor& get_omegas() {
            TORCH_CHECK(this->method == METHODS::ACCUM,
                        STRINGIFY(omegas) " are not computed when " STRINGIFY(method) " is set to " STRINGIFY(METHODS::COVAR)
                        ". Consider initializing " STRINGIFY(SN2Solver) " with " STRINGIFY(method=METHODS::ACCUM) ".")
            return this->omegas;
        }
    };
}

#endif