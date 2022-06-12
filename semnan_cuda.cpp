#include <torch/extension.h>
#include "device_data.h"
#include <math_constants.h>
#include <vector>
#include <set>
#include <variant>
#include <iostream>
#include <optional>
#include <utility>

using namespace torch::indexing;

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

extern template class SEMNANDeviceData<float>;
extern template class SEMNANDeviceData<double>;


// Custom loss method
class SEMNANSolverLoss {
    public:
    virtual torch::Tensor loss_proxy(const torch::Tensor& sample_covariance, const torch::Tensor& visible_covariance) const = 0;

    public:
    virtual torch::Tensor loss(const torch::Tensor& sample_covariance, const torch::Tensor& visible_covariance) const = 0;

    public:
    virtual torch::Tensor loss_backward(const torch::Tensor& sample_covariance, const torch::Tensor& visible_covariance) const = 0;

    public:
    virtual ~SEMNANSolverLoss() {}
};


// Class SEMNANSolver
class SEMNANSolver {
    torch::Tensor structure;                // We keep all the tensors alive for the lifetime of SEMNANSolver
    torch::Tensor lambda;
    torch::Tensor weights;
    torch::Tensor covariance;
    torch::Tensor visible_covariance;
    torch::Tensor sample_covariance;
    torch::Tensor sample_covariance_inv;    // We keep `sample_covariance_inv` and
    torch::Tensor sample_covariance_logdet; // `sample_covariance_logdet` so that we do not re-compute them.
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

    torch::Tensor (SEMNANSolver::*loss_method)(void);
    torch::Tensor (SEMNANSolver::*loss_proxy_method)(void);
    void (SEMNANSolver::*loss_backward_method)(torch::Tensor&);
    std::shared_ptr<SEMNANSolverLoss> loss_function;

    public:
    enum LOSS {
        KULLBACK_LEIBLER = 0,
        BHATTACHARYYA
    };

    private:
    inline int32_t num_layers() {
        return this->layers_vec.size();
    }

    private:
    void init_parameters(
            const torch::Tensor& structure,
            const torch::Tensor& parameters,
            torch::Dtype dtype,
            LOSS loss,
            std::shared_ptr<SEMNANSolverLoss> loss_function,
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
        TORCH_CHECK(parameters.numel() == 0 || parameters.sizes() == structure.sizes(), "`parameters` must have the same size as `structure`.");
        TORCH_CHECK(loss >= LOSS::KULLBACK_LEIBLER && loss <= LOSS::BHATTACHARYYA, "Invalid `loss` type.");

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

        if (loss_function)
            this->loss_function = loss_function;
        else
            switch (loss) {
                case LOSS::KULLBACK_LEIBLER:
                    this->loss_method = &SEMNANSolver::kullback_leibler_loss;
                    this->loss_proxy_method = &SEMNANSolver::kullback_leibler_loss_proxy;
                    this->loss_backward_method = &SEMNANSolver::kullback_leibler_loss_backward;
                    break;
                case LOSS::BHATTACHARYYA:
                    this->loss_method = &SEMNANSolver::bhattacharyya_loss;
                    this->loss_proxy_method = &SEMNANSolver::bhattacharyya_loss_proxy;
                    this->loss_backward_method = &SEMNANSolver::bhattacharyya_loss_backward;
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
            LOSS loss = SEMNANSolver::LOSS::KULLBACK_LEIBLER,
            std::shared_ptr<SEMNANSolverLoss> loss_function = nullptr,
            bool check = true
    ) {
        this->init_parameters(structure, parameters, dtype, loss, loss_function, check);
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
        return loss_function ?
                    loss_function->loss(sample_covariance, visible_covariance) :
                    (this->*loss_method)();
    }

    public:
    torch::Tensor loss_proxy() {
        return loss_function ?
                    loss_function->loss_proxy(sample_covariance, visible_covariance) :
                    (this->*loss_proxy_method)();
    }

    private:
    void loss_backward(torch::Tensor& visible_covariance_grad) {
        if (loss_function)
            visible_covariance_grad.copy_(loss_function->loss_backward(sample_covariance, visible_covariance));
        else
            (this->*loss_backward_method)(visible_covariance_grad);
    }

    public:
    torch::Tensor kullback_leibler_loss_proxy() {
        return torch::subtract(
            torch::trace(torch::mm(get_sample_covariance_inv(), visible_covariance)),
            torch::logdet(visible_covariance)
        );
    }

    public:
    torch::Tensor kullback_leibler_loss() {
        return torch::div(
            torch::add(
                torch::subtract(kullback_leibler_loss_proxy(), visible_size),
                get_sample_covariance_logdet()
            )
        , 2.0);
    }

    private:
    void kullback_leibler_loss_backward(torch::Tensor& visible_covariance_grad) {
        visible_covariance_grad.copy_(this->get_sample_covariance_inv());
        visible_covariance_grad.subtract_(torch::inverse(visible_covariance));
    }

    public:
    torch::Tensor bhattacharyya_loss_proxy() {
        return torch::div(
            torch::det(
                torch::div(torch::add(visible_covariance, sample_covariance), 2.0)
            ),
            torch::sqrt(torch::det(visible_covariance))
        );
    }

    public:
    torch::Tensor bhattacharyya_loss() {
        return torch::div(
            torch::log(
                torch::div(bhattacharyya_loss_proxy(), torch::sqrt(torch::det(sample_covariance)))
            ), 2
        );
    }

    private:
    void bhattacharyya_loss_backward(torch::Tensor& visible_covariance_grad) {
        visible_covariance_grad.copy_(torch::inverse(torch::add(sample_covariance, visible_covariance)));
        visible_covariance_grad.subtract_(torch::div(torch::inverse(visible_covariance), 2.0));
        visible_covariance_grad.transpose_(0, 1);
    }

    public:
    void set_sample_covariance(
            torch::Tensor sample_covariance
    ) {
        TORCH_CHECK(sample_covariance.dim() == 2, "`sample_covariance` must be 2-dimensional; it is ", sample_covariance_inv.dim(), "-dimensional.");
        TORCH_CHECK(sample_covariance.size(0) == sample_covariance.size(1), "`sample_covariance` must be a square matrix.");
        TORCH_CHECK(sample_covariance.size(0) == visible_size, "`sample_covariance` must be a ", visible_size, "×", visible_size, " matrix.");

        this->sample_covariance = sample_covariance.to(this->weights.options());
    }

    public:
    torch::Tensor& get_sample_covariance() {
       return this->sample_covariance;
    }

    private:
    torch::Tensor& get_sample_covariance_inv() {
        if (this->sample_covariance_inv.numel() == 0)
            this->sample_covariance_inv = torch::inverse(this->sample_covariance);

        return this->sample_covariance_inv;
    }

    private:
    torch::Tensor& get_sample_covariance_logdet() {
        if (this->sample_covariance_logdet.numel() == 0)
            this->sample_covariance_logdet = torch::logdet(this->sample_covariance);

        return this->sample_covariance_logdet;
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


// Bind abstract class to python
class PySEMNANSolverLoss : public SEMNANSolverLoss {
    public:
    virtual torch::Tensor loss_proxy(const torch::Tensor& sample_covariance, const torch::Tensor& visible_covariance) const override {
        PYBIND11_OVERLOAD_PURE(torch::Tensor, SEMNANSolverLoss, loss_proxy, sample_covariance, visible_covariance);
    }

    public:
    virtual torch::Tensor loss(const torch::Tensor& sample_covariance, const torch::Tensor& visible_covariance) const override {
        PYBIND11_OVERLOAD_PURE(torch::Tensor, SEMNANSolverLoss, loss, sample_covariance, visible_covariance);
    }

    public:
    virtual torch::Tensor loss_backward(const torch::Tensor& sample_covariance, const torch::Tensor& visible_covariance) const override {
        PYBIND11_OVERLOAD_PURE(torch::Tensor, SEMNANSolverLoss, loss_backward, sample_covariance, visible_covariance);
    }
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    auto semnan_solver = py::class_<SEMNANSolver>(m, "SEMNANSolver").
            def(py::init([] (
                        torch::Tensor& structure,
                        std::optional<torch::Tensor> parameters,
                        std::optional<py::object> dtype,
                        std::optional<std::variant<SEMNANSolver::LOSS, std::shared_ptr<SEMNANSolverLoss>>> loss,
                        std::optional<bool> check
                ) {
                    return SEMNANSolver(
                        structure,
                        parameters.has_value() ? parameters.value() : torch::Tensor(),
                        dtype.has_value() ? torch::python::detail::py_object_to_dtype(dtype.value()) : torch::kFloat,
                        loss.has_value() && std::holds_alternative<SEMNANSolver::LOSS>(loss.value()) ?
                                std::get<SEMNANSolver::LOSS>(loss.value()) : SEMNANSolver::LOSS::KULLBACK_LEIBLER,
                        loss.has_value() && std::holds_alternative<std::shared_ptr<SEMNANSolverLoss>>(loss.value()) ?
                                std::get<std::shared_ptr<SEMNANSolverLoss>>(loss.value()) : nullptr,
                        check.has_value() ? check.value() : true
                    );
                }
            ), py::arg("structure"), py::arg("weights")=std::nullopt, py::arg("dtype")=std::nullopt,
                    py::arg("loss")=std::nullopt, py::arg("check")=std::nullopt).
            def("forward", &SEMNANSolver::forward).
            def("backward", &SEMNANSolver::backward).
            def_property_readonly("lambda_", &SEMNANSolver::get_lambda).
            def_property_readonly("covariance_", &SEMNANSolver::get_covariance).
            def_property_readonly("visible_covariance_", &SEMNANSolver::get_visible_covariance).
            def_property("weights", &SEMNANSolver::get_weights, &SEMNANSolver::set_weights).
            def_property("sample_covariance", &SEMNANSolver::get_sample_covariance, &SEMNANSolver::set_sample_covariance).
            def("loss", &SEMNANSolver::loss).
            def("loss_proxy", &SEMNANSolver::loss_proxy).
            def("bhattacharyya_loss", &SEMNANSolver::bhattacharyya_loss).
            def("bhattacharyya_loss_proxy", &SEMNANSolver::bhattacharyya_loss_proxy).
            def("kullback_leibler_loss", &SEMNANSolver::kullback_leibler_loss).
            def("kullback_leibler_loss_proxy", &SEMNANSolver::kullback_leibler_loss_proxy);

    py::enum_<SEMNANSolver::LOSS>(semnan_solver, "LOSS").
            value("KULLBACK_LEIBLER", SEMNANSolver::LOSS::KULLBACK_LEIBLER).
            value("BHATTACHARYYA", SEMNANSolver::LOSS::BHATTACHARYYA).
            export_values();

    py::class_<SEMNANSolverLoss, PySEMNANSolverLoss, std::shared_ptr<SEMNANSolverLoss>>(m, "SEMNANSolverLoss").
            def(py::init<>()).
            def("loss_proxy", &SEMNANSolverLoss::loss_proxy, py::arg("sample_covariance"), py::arg("visible_covariance")).
            def("loss", &SEMNANSolverLoss::loss, py::arg("sample_covariance"), py::arg("visible_covariance")).
            def("loss_backward", &SEMNANSolverLoss::loss_backward, py::arg("sample_covariance"), py::arg("visible_covariance"));
}