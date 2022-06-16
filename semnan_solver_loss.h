#ifndef SEMNAN_CUDA_LOSS_H
#define SEMNAN_CUDA_LOSS_H

#include <torch/extension.h>
#include "declarations.h"

namespace semnan_cuda::loss {
    // Custom loss method
    class LossBase {
        friend class semnan_cuda::SEMNANSolver;

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
            TORCH_CHECK(sample_covariance.dim() == 2, "`sample_covariance` must be 2-dimensional; it is ", sample_covariance.dim(), "-dimensional.");
            TORCH_CHECK(sample_covariance.size(0) == sample_covariance.size(1), "`sample_covariance` must be a square matrix.");
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

        protected:
        torch::Tensor& get_sample_covariance_inv() {
            if (!loss_data->sample_covariance_inv.defined())
                loss_data->sample_covariance_inv = torch::inverse(get_sample_covariance());

            return loss_data->sample_covariance_inv;
        }

        protected:
        const torch::Tensor& get_sample_covariance() const {
           return this->sample_covariance;
        }

        protected:
        torch::Tensor& get_sample_covariance() {
           return this->sample_covariance;
        }

        protected:
        inline int64_t get_size() const {
            return this->sample_covariance.size(0);
        }

        protected:
        const torch::Tensor& get_sample_covariance_inv() const {
            if (!this->loss_data->sample_covariance_inv.defined())
                this->loss_data->sample_covariance_inv = torch::inverse(this->sample_covariance);

            return this->loss_data->sample_covariance_inv;
        }

        protected:
        const torch::Tensor& get_sample_covariance_logdet() const {
            if (!this->loss_data->sample_covariance_logdet.defined())
                this->loss_data->sample_covariance_logdet = torch::logdet(this->sample_covariance);

            return this->loss_data->sample_covariance_logdet;
        }

        public:
        virtual torch::Tensor loss_proxy(const torch::Tensor& visible_covariance) const = 0;

        public:
        virtual torch::Tensor loss(const torch::Tensor& visible_covariance) const = 0;

        public:
        virtual void loss_backward(const torch::Tensor& visible_covariance, torch::Tensor& visible_covariance_grad) const = 0;

        public:
        virtual ~LossBase() {
            maybe_remove_data();
        }
    };

    class KullbackLeibler : public LossBase {
        public:
        virtual torch::Tensor loss_proxy(const torch::Tensor& visible_covariance) const {
            return torch::subtract(
                torch::trace(torch::mm(get_sample_covariance_inv(), visible_covariance)),
                torch::logdet(visible_covariance)
            );
        }

        public:
        virtual torch::Tensor loss(const torch::Tensor& visible_covariance) const {
            return torch::div(
                torch::add(
                    torch::subtract(loss_proxy(visible_covariance), get_size()),
                    get_sample_covariance_logdet()
                )
            , 2.0);
        }

        public:
        virtual void loss_backward(const torch::Tensor& visible_covariance, torch::Tensor& visible_covariance_grad) const {
            visible_covariance_grad.copy_(get_sample_covariance_inv());
            visible_covariance_grad.subtract_(torch::inverse(visible_covariance));
        }
    };

    class Bhattacharyya : public LossBase {
        public:
        virtual torch::Tensor loss_proxy(const torch::Tensor& visible_covariance) const {
            return torch::div(
                torch::det(
                    torch::div(torch::add(visible_covariance, get_sample_covariance()), 2.0)
                ),
                torch::sqrt(torch::det(visible_covariance))
            );
        }

        public:
        virtual torch::Tensor loss(const torch::Tensor& visible_covariance) const {
            return torch::div(
                torch::log(
                    torch::div(loss_proxy(visible_covariance), torch::sqrt(torch::det(get_sample_covariance())))
                ), 2
            );
        }

        public:
        virtual void loss_backward(const torch::Tensor& visible_covariance, torch::Tensor& visible_covariance_grad) const {
            visible_covariance_grad.copy_(torch::inverse(torch::add(get_sample_covariance(), visible_covariance)));
            visible_covariance_grad.subtract_(torch::div(torch::inverse(visible_covariance), 2.0));
            visible_covariance_grad.transpose_(0, 1);
        }
    };
}

#endif