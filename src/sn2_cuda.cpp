#include <torch/extension.h>
#include "sn2_solver.h"

using namespace sn2_cuda;
using namespace sn2_cuda::loss;


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

    py::class_<SquaredHellinger, LossBase, std::shared_ptr<SquaredHellinger>>(loss, "SquaredHellinger")
            .def(py::init<>());

    auto sn2_solver = py::class_<SN2Solver>(m, "SN2Solver")
            .def(py::init([] (
                                  torch::Tensor& structure,
                                  std::optional<torch::Tensor> parameters,
                                  std::optional<torch::Tensor> sample_covariance,
                                  std::optional<py::object> dtype,
                                  std::optional<std::shared_ptr<LossBase>> loss_function,
                                  std::optional<SN2Solver::METHODS> method,
                                  std::optional<bool> validate
                          ) {
                              return SN2Solver(
                                      structure,
                                      std::move(parameters),
                                      std::move(sample_covariance),
                                      dtype.has_value() ? torch::python::detail::py_object_to_dtype(dtype.value()) : torch::kFloat,
                                      loss_function.has_value() ? loss_function.value() : nullptr,
                                      method.has_value() ? method.value() : SN2Solver::METHODS::COVAR,
                                      !validate.has_value() || validate.value()
                              );
                          }
                 ), py::arg("structure"), py::arg("weights")=std::nullopt, py::arg("sample_covariance")=std::nullopt,
                 py::arg("dtype")=std::nullopt, py::arg("loss")=std::nullopt, py::arg("method")=std::nullopt,
                 py::arg("validate")=std::nullopt, /* keep the user-defined loss function alive */ py::keep_alive<1, 6>())
            .def("forward", &SN2Solver::forward)
            .def("backward", &SN2Solver::backward)
            .def_property_readonly("omegas_", &SN2Solver::get_omegas)
            .def_property_readonly("lambda_", &SN2Solver::get_lambda)
            .def_property_readonly("covariance_", &SN2Solver::get_covariance)
            .def_property_readonly("lv_transformation_", &SN2Solver::get_lv_transformation)
            .def_property_readonly("visible_covariance_", &SN2Solver::get_visible_covariance)
            .def_property("weights", &SN2Solver::get_weights, &SN2Solver::set_weights)
            .def_property("sample_covariance", &SN2Solver::get_sample_covariance, &SN2Solver::set_sample_covariance)
            .def("loss", &SN2Solver::loss)
            .def("loss_proxy", &SN2Solver::loss_proxy);

    py::enum_<SN2Solver::METHODS>(sn2_solver, "METHODS")
            .value("COVAR", SN2Solver::METHODS::COVAR)
            .value("ACCUM", SN2Solver::METHODS::ACCUM)
            .export_values();
}