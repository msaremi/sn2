#include <torch/extension.h>
#include "semnan_solver.h"

using namespace semnan_cuda;
using namespace semnan_cuda::loss;


class PubLossBase : public LossBase {
    public:
    using LossBase::get_size;
    using LossBase::loss_proxy;
    using LossBase::loss;
    using LossBase::loss_backward;
    using LossBase::get_sample_covariance;
    using LossBase::get_sample_covariance_inv;
    using LossBase::get_sample_covariance_logdet;
};

// Bind abstract class to python
class PyLossBase : public LossBase {
    public:
    torch::Tensor loss_proxy(const torch::Tensor& visible_covariance) const override {
        PYBIND11_OVERLOAD_PURE(torch::Tensor, LossBase, loss_proxy, visible_covariance);
    }

    public:
    torch::Tensor loss(const torch::Tensor& visible_covariance) const override {
        PYBIND11_OVERLOAD_PURE(torch::Tensor, LossBase, loss, visible_covariance);
    }

    public:
    void loss_backward(const torch::Tensor& visible_covariance, torch::Tensor& visible_covariance_grad) const override {
        PYBIND11_OVERLOAD_PURE(void, LossBase, loss_backward, visible_covariance, visible_covariance_grad);
    }
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    auto loss = m.def_submodule("loss");

    py::class_<LossBase, PyLossBase, std::shared_ptr<LossBase>>(loss, "LossBase")
            .def(py::init<>())
            .def_property_readonly("size", &PubLossBase::get_size)
            .def_property_readonly("sample_covariance", &PubLossBase::get_sample_covariance)
            .def_property_readonly("sample_covariance_inv", &PubLossBase::get_sample_covariance_inv)
            .def_property_readonly("sample_covariance_logdet", &PubLossBase::get_sample_covariance_logdet)
            .def("loss_proxy", &PubLossBase::loss_proxy, py::arg("visible_covariance"))
            .def("loss", &PubLossBase::loss, py::arg("visible_covariance"))
            .def("loss_backward", &PubLossBase::loss_backward, py::arg("visible_covariance"), py::arg("visible_covariance_grad"));

    py::class_<KullbackLeibler, LossBase, std::shared_ptr<KullbackLeibler>>(loss, "KullbackLeibler")
            .def(py::init<>());

    py::class_<Bhattacharyya, LossBase, std::shared_ptr<Bhattacharyya>>(loss, "Bhattacharyya")
            .def(py::init<>());

    auto semnan_solver = py::class_<SEMNANSolver>(m, "SEMNANSolver")
            .def(py::init([] (
                                  torch::Tensor& structure,
                                  std::optional<torch::Tensor> parameters,
                                  std::optional<torch::Tensor> sample_covariance,
                                  std::optional<py::object> dtype,
                                  std::optional<std::shared_ptr<LossBase>> loss_function,
                                  std::optional<SEMNANSolver::METHODS> method,
                                  std::optional<bool> validate
                          ) {
                              return SEMNANSolver(
                                      structure,
                                      std::move(parameters),
                                      std::move(sample_covariance),
                                      dtype.has_value() ? torch::python::detail::py_object_to_dtype(dtype.value()) : torch::kFloat,
                                      loss_function.has_value() ? loss_function.value() : nullptr,
                                      method.has_value() ? method.value() : SEMNANSolver::METHODS::COVAR,
                                      !validate.has_value() || validate.value()
                              );
                          }
                 ), py::arg("structure"), py::arg("weights")=std::nullopt, py::arg("sample_covariance")=std::nullopt,
                 py::arg("dtype")=std::nullopt, py::arg("loss")=std::nullopt, py::arg("method")=std::nullopt,
                 py::arg("validate")=std::nullopt, /* keep the user-defined loss function alive */ py::keep_alive<1, 5>())
            .def("forward", &SEMNANSolver::forward)
            .def("backward", &SEMNANSolver::backward)
            .def_property_readonly("omegas_", &SEMNANSolver::get_omegas)
            .def_property_readonly("lambda_", &SEMNANSolver::get_lambda)
            .def_property_readonly("covariance_", &SEMNANSolver::get_covariance)
            .def_property_readonly("lv_transformation_", &SEMNANSolver::get_lv_transformation)
            .def_property_readonly("visible_covariance_", &SEMNANSolver::get_visible_covariance)
            .def_property("weights", &SEMNANSolver::get_weights, &SEMNANSolver::set_weights)
            .def_property("sample_covariance", &SEMNANSolver::get_sample_covariance, &SEMNANSolver::set_sample_covariance)
            .def("loss", &SEMNANSolver::loss)
            .def("loss_proxy", &SEMNANSolver::loss_proxy);

    py::enum_<SEMNANSolver::METHODS>(semnan_solver, "METHODS")
            .value("COVAR", SEMNANSolver::METHODS::COVAR)
            .value("ACCUM", SEMNANSolver::METHODS::ACCUM)
            .export_values();
}