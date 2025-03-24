#include <vector>
#include <math.h>
#include <stdexcept>
#include <string>
#include "kernel.h"

double Kernel::linear_kernel(const std::vector<double> &x1, const std::vector<double> &x2) const
{
    if (x1.size() != x2.size())
    {
        throw std::invalid_argument("Vectors must have the same size");
    }
    double dot = 0.0;
    for (size_t i = 0; i < x1.size(); ++i)
    {
        dot += x1[i] * x2[i];
    }
    return dot;
}

double Kernel::polynomial_kernel(const std::vector<double> &x1, const std::vector<double> &x2) const
{
    return pow(linear_kernel(x1, x2) + 1.0, degree_);
}

double Kernel::rbf_kernel(const std::vector<double> &x1, const std::vector<double> &x2) const
{
    if (x1.size() != x2.size())
    {
        throw std::invalid_argument("Vectors must have the same size");
    }
    double norm = 0.0;
    for (size_t i = 0; i < x1.size(); ++i)
    {
        double diff = x1[i] - x2[i];
        norm += diff * diff;
    }
    return exp(-gamma_ * sqrt(norm));
}
