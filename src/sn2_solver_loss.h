#ifndef SN2_CUDA_LOSS_H
#define SN2_CUDA_LOSS_H

#include <torch/extension.h>
#include <tuple>
#include "stringify.h"
#include "declarations.h"

inline torch::Tensor inverse_ex(const torch::Tensor& x) {
    return std::get<0>(at::linalg_inv_ex(x));
}

namespace sn2_cuda::loss {
    // Custom loss method
    class LossBase {
        friend class sn2_cuda::SN2Solver;

        struct LossData {
            torch::Tensor sample_covariance_inv;
            torch::Tensor sample_covariance_logdet;
        };

        struct LossDataCmp {
            bool operator()(const torch::Tensor& first, const torch::Tensor& second) const {
                    return !first.is_same(second) && first.data_ptr() < second.data_ptr();
            }
        };

        static inline std::map<torch::Tensor, std::shared_ptr<LossData>, LossDataCmp> loss_data_map = {};

        torch::Tensor sample_covariance;
        std::shared_ptr<LossData> loss_data;

        private:
        void maybe_remove_data() {
            if (sample_covariance.defined()) {
                auto loss_data_iter = loss_data_map.find(sample_covariance);

                if (loss_data_iter->second.use_count() <= 2)
                    loss_data_map.erase(loss_data_iter);
            }
        }

        protected:
        void set_sample_covariance(const torch::Tensor& sample_covariance) {
            TORCH_CHECK(sample_covariance.dim() == 2, STRINGIFY(sample_covariance) " must be 2-dimensional; it is ", sample_covariance.dim(), "-dimensional.")
            TORCH_CHECK(sample_covariance.size(0) == sample_covariance.size(1), STRINGIFY(sample_covariance) " must be a square matrix.")
            auto loss_data_iter = loss_data_map.lower_bound(sample_covariance);

            if (loss_data_iter != loss_data_map.end() && loss_data_iter->first.is_same(sample_covariance))
                this->loss_data = loss_data_iter->second;
            else {
                this->loss_data = std::make_shared<LossData>();
                loss_data_map.insert(loss_data_iter, {sample_covariance, this->loss_data});
            }

            maybe_remove_data();
            this->sample_covariance = sample_covariance;
        }

        private:
        inline void check_has_sample_covariance() {
            TORCH_CHECK(has_sample_covariance(), STRINGIFY(sample_covariance) " has not been set.")
        }

        protected:
        const bool has_sample_covariance() const {
            return this->sample_covariance.defined();
        }

        protected:
        const torch::Tensor& get_sample_covariance() const {
           return this->sample_covariance;
        }

        protected:
        inline int64_t get_size() const {
            return this->sample_covariance.size(0);
        }

        protected:
        const torch::Tensor& get_sample_covariance_inv() const {
            if (!this->loss_data->sample_covariance_inv.defined())
                this->loss_data->sample_covariance_inv = inverse_ex(this->sample_covariance);

            return this->loss_data->sample_covariance_inv;
        }

        protected:
        const torch::Tensor& get_sample_covariance_logdet() const {
            if (!this->loss_data->sample_covariance_logdet.defined())
                this->loss_data->sample_covariance_logdet = torch::logdet(this->sample_covariance);

            return this->loss_data->sample_covariance_logdet;
        }

        protected:
        virtual torch::Tensor loss_proxy(const torch::Tensor& visible_covariance) const = 0;

        protected:
        virtual torch::Tensor loss(const torch::Tensor& visible_covariance) const = 0;

        protected:
        virtual void loss_backward(const torch::Tensor& visible_covariance, torch::Tensor& visible_covariance_grad) const = 0;

        public:
        virtual ~LossBase() {
            maybe_remove_data();
        }
    };

    class KullbackLeibler : public LossBase {
        protected:
        virtual torch::Tensor loss_proxy(const torch::Tensor& visible_covariance) const {
            return torch::subtract(
                torch::trace(torch::mm(get_sample_covariance_inv(), visible_covariance)),
                torch::logdet(visible_covariance)
            );
        }

        protected:
        virtual torch::Tensor loss(const torch::Tensor& visible_covariance) const {
            return torch::div(
                torch::add(
                    torch::subtract(loss_proxy(visible_covariance), get_size()),
                    get_sample_covariance_logdet()
                )
            , 2.0);
        }

        protected:
        virtual void loss_backward(const torch::Tensor& visible_covariance, torch::Tensor& visible_covariance_grad) const {
            visible_covariance_grad.copy_(get_sample_covariance_inv());
            visible_covariance_grad.subtract_(inverse_ex(visible_covariance));
        }
    };

    class Bhattacharyya : public LossBase {
        protected:
        virtual torch::Tensor loss_proxy(const torch::Tensor& visible_covariance) const {
            return torch::div(
                torch::det(
                    torch::div(torch::add(visible_covariance, get_sample_covariance()), 2.0)
                ),
                torch::sqrt(torch::det(visible_covariance))
            );
        }

        protected:
        virtual torch::Tensor loss(const torch::Tensor& visible_covariance) const {
            return torch::div(
                torch::log(
                    torch::div(loss_proxy(visible_covariance), torch::sqrt(torch::det(get_sample_covariance())))
                ), 2
            );
        }

        protected:
        virtual void loss_backward(const torch::Tensor& visible_covariance, torch::Tensor& visible_covariance_grad) const {
            visible_covariance_grad.copy_(inverse_ex(torch::add(get_sample_covariance(), visible_covariance)));
            visible_covariance_grad.subtract_(torch::div(inverse_ex(visible_covariance), 2.0));
            visible_covariance_grad.transpose_(0, 1);
        }
    };


    class SquaredHellinger : public LossBase {
        protected:
        virtual torch::Tensor loss_proxy(const torch::Tensor& visible_covariance) const {
            return torch::neg(
                torch::div(
                    torch::pow(torch::det(visible_covariance), 1.0/4.0),
                    torch::sqrt(torch::det(
                        torch::div(torch::add(visible_covariance, get_sample_covariance()), 2.0)
                    ))
                )
            );
        }

        protected:
        virtual torch::Tensor loss(const torch::Tensor& visible_covariance) const {
            return torch::add(torch::mul(
                loss_proxy(visible_covariance), torch::pow(torch::det(get_sample_covariance()), 1.0/4.0)
            ), 1.0);
        }

        protected:
        virtual void loss_backward(const torch::Tensor& visible_covariance, torch::Tensor& visible_covariance_grad) const {
            visible_covariance_grad.copy_(inverse_ex(visible_covariance));
            visible_covariance_grad.mul_(torch::subtract(torch::det(visible_covariance), torch::det(get_sample_covariance())));
            visible_covariance_grad.div_(torch::pow(torch::add(torch::det(visible_covariance), torch::det(get_sample_covariance())), 3.0/2.0));
            visible_covariance_grad.mul_(torch::pow(torch::det(get_sample_covariance()), 1.0/4.0));
            visible_covariance_grad.mul_(torch::pow(torch::det(visible_covariance), 1.0/4.0));
            visible_covariance_grad.transpose_(0, 1);
        }
    };
}

#endif