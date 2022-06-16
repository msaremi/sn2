#include <torch/extension.h>
#include "semnan_solver.h"

using namespace semnan_cuda;
using namespace semnan_cuda::loss;


// Bind abstract class to python
class PyLossBase : public LossBase {
    public:
    virtual torch::Tensor loss_proxy(const torch::Tensor& visible_covariance) const override {
        PYBIND11_OVERLOAD_PURE(torch::Tensor, LossBase, loss_proxy, visible_covariance);
    }

    public:
    virtual torch::Tensor loss(const torch::Tensor& visible_covariance) const override {
        PYBIND11_OVERLOAD_PURE(torch::Tensor, LossBase, loss, visible_covariance);
    }

    public:
    virtual void loss_backward(const torch::Tensor& visible_covariance, torch::Tensor& visible_covariance_grad) const override {
        PYBIND11_OVERLOAD_PURE(void, LossBase, loss_backward, visible_covariance, visible_covariance_grad);
    }
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    auto semnan_solver = py::class_<SEMNANSolver>(m, "SEMNANSolver")
            .def(py::init([] (
                        torch::Tensor& structure,
                        std::optional<torch::Tensor> parameters,
                        std::optional<py::object> dtype,
                        std::optional<std::shared_ptr<LossBase>> loss_function,
                        std::optional<bool> check
                ) {
                    return SEMNANSolver(
                        structure,
                        parameters.has_value() ? parameters.value() : torch::Tensor(),
                        dtype.has_value() ? torch::python::detail::py_object_to_dtype(dtype.value()) : torch::kFloat,
                        loss_function.has_value() ? loss_function.value() : nullptr,
                        check.has_value() ? check.value() : true
                    );
                }
            ), py::arg("structure"), py::arg("weights")=std::nullopt, py::arg("dtype")=std::nullopt,
                    py::arg("loss")=std::nullopt, py::arg("check")=std::nullopt)
            .def("forward", &SEMNANSolver::forward)
            .def("backward", &SEMNANSolver::backward)
            .def("lv_transformation_", &SEMNANSolver::get_lv_transformation)
            .def_property_readonly("lambda_", &SEMNANSolver::get_lambda)
            .def_property_readonly("covariance_", &SEMNANSolver::get_covariance)
            .def_property_readonly("visible_covariance_", &SEMNANSolver::get_visible_covariance)
            .def_property("weights", &SEMNANSolver::get_weights, &SEMNANSolver::set_weights)
            .def_property("sample_covariance", &SEMNANSolver::get_sample_covariance, &SEMNANSolver::set_sample_covariance)
            .def("loss", &SEMNANSolver::loss)
            .def("loss_proxy", &SEMNANSolver::loss_proxy);

    auto loss = m.def_submodule("loss");

    py::class_<LossBase, PyLossBase, std::shared_ptr<LossBase>>(loss, "LossBase")
            .def(py::init<>())
            .def("loss_proxy", &LossBase::loss_proxy, py::arg("visible_covariance"))
            .def("loss", &LossBase::loss, py::arg("visible_covariance"))
            .def("loss_backward", &LossBase::loss_backward, py::arg("visible_covariance"), py::arg("visible_covariance_grad"));

    py::class_<KullbackLeibler, LossBase, std::shared_ptr<KullbackLeibler>>(loss, "KullbackLeibler")
            .def(py::init<>());

    py::class_<Bhattacharyya, LossBase, std::shared_ptr<Bhattacharyya>>(loss, "Bhattacharyya")
            .def(py::init<>());
}