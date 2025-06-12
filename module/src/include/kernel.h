#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#ifndef KERNEL_H
#define KERNEL_H
class Kernel
{
public:
    enum class Type
    {
        Linear,
        Polynomial,
        RBF
    };

    Kernel(const std::string &kernel = "linear", int degree = 2, double gamma = 2.0)
        : degree_(degree), gamma_(gamma)
    {
        if (kernel == "linear")
        {
            kernel_type_ = Type::Linear;
        }
        else if (kernel == "polynomial")
        {
            kernel_type_ = Type::Polynomial;
        }
        else if (kernel == "rbf")
        {
            kernel_type_ = Type::RBF;
        }
        else
        {
            throw std::invalid_argument("Unknown kernel type: " + kernel);
        }
    }

    double operator()(const std::vector<double> &x1, const std::vector<double> &x2) const
    {
        switch (kernel_type_)
        {
        case Type::Linear:
            return linear_kernel(x1, x2);
        case Type::Polynomial:
            return polynomial_kernel(x1, x2);
        case Type::RBF:
            return rbf_kernel(x1, x2);
        default:
            throw std::runtime_error("Invalid kernel type");
        }
    }

private:
    Type kernel_type_;
    int degree_;
    double gamma_;

    double linear_kernel(const std::vector<double> &x1, const std::vector<double> &x2) const;

    double polynomial_kernel(const std::vector<double> &x1, const std::vector<double> &x2) const;

    double rbf_kernel(const std::vector<double> &x1, const std::vector<double> &x2) const;
};
#include "kernel.cpp"
#endif