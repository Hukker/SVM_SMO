#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include "smo.h"

namespace detail
{
    std::mt19937 &get_engine()
    {
        static std::mt19937 engine(std::random_device{}());
        return engine;
    }
}

void seed(unsigned int seed_value)
{
    detail::get_engine().seed(seed_value);
}

template <typename T>
T random_choice(const std::vector<T> &input)
{
    if (input.empty())
    {
        throw std::invalid_argument("Input vector must not be empty");
    }
    std::uniform_int_distribution<size_t> dist(0, input.size() - 1);
    return input[dist(detail::get_engine())];
}

int random_choice(int n)
{
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    return dist(detail::get_engine());
}

std::vector<int> permutation(int n)
{
    auto &engine = detail::get_engine();
    std::vector<int> result(n);
    std::iota(result.begin(), result.end(), 0);
    std::shuffle(result.begin(), result.end(), engine);
    return result;
}

template <typename T>
std::vector<T> permutation(const std::vector<T> &input)
{
    auto &engine = detail::get_engine();
    std::vector<T> result = input;
    std::shuffle(result.begin(), result.end(), engine);
    return result;
}

template <typename T>
vector<int> get_conditioned_indeces(vector<T> &arr, bool (*cond)(T))
{
    vector<int> ans;
    for (int i = 0; i < arr.size(); i++)
    {
        if (cond(arr[i]))
        {
            ans.push_back(i);
        }
    }
    return ans;
}

void QPSolver::solve()
{
    int num_changed = 0;
    int examine_all = 1;
    int iter_count = 0;

    int result;
    static double C_ = C;
    static double tol_ = tol;

    auto non_bound_check = [](double a)
    {if (a > 0 && a < C_ - tol_){return true;}return false; };

    vector<int> non_bound_inds = get_conditioned_indeces<double>(alpha, non_bound_check);

    while ((num_changed > 0 || examine_all) && (iter_count < max_iter))
    {
        num_changed = 0;
        if (examine_all)
        {
            for (int i = 0; i < n_samples; i++)
            {
                num_changed += examine_example(i);
            }
        }
        else
        {
            for (int j = 0; j < non_bound_inds.size(); j++)
            {
                num_changed += examine_example(non_bound_inds[j]);
            }
        }

        if (examine_all == 1)
        {
            examine_all = 0;
        }
        else if (num_changed == 0)
        {
            examine_all = 1;
        }

        iter_count++;
        if (logs)
        {
            std::cout << "QPSolver on its " << iter_count << " iteration!" << std::endl;
        }
    }
}

vector<double> QPSolver::get_alpha()
{
    return alpha;
}

double QPSolver::get_b()
{
    return b;
}

int QPSolver::take_step(int i1, int i2)
{
    if (i1 == i2)
    {
        return 0;
    }
    static double C_ = C;
    static double tol_ = tol;

    auto non_bound_check = [](double a)
    {if (a > 0 && a < C_ - tol_){return true;}return false; };
    double alpha1 = alpha[i1];
    double alpha2 = alpha[i2];

    double y1 = y_[i1];
    double y2 = y_[i2];

    double E1 = errors[i1];
    double E2 = errors[i2];
    double s = y1 * y2;

    double L = 0;
    double H = 0;

    if (y1 == y2)
    {
        L = std::max(0.0, alpha1 + alpha2 - C);
        H = std::min(C, alpha1 + alpha2);
    }
    else if (y1 != y2)
    {
        L = std::max(0.0, alpha2 - alpha1);
        H = std::min(C, C + alpha2 - alpha1);
    }
    if (L == H)
    {
        return 0;
    }

    double k11 = kernel_(X_[i1], X_[i1]);
    double k12 = kernel_(X_[i1], X_[i2]);
    double k22 = kernel_(X_[i2], X_[i2]);

    double a2 = 0;
    double eta = k11 + k22 - 2 * k12;
    if (eta > 0)
    {
        a2 = alpha2 + y2 * (E1 - E2) / eta;
        if (a2 <= L)
        {
            a2 = L;
        }
        else if (a2 >= H)
        {
            a2 = H;
        }
    }
    else
    {
        std::cout << "Went for eta <= 0!" << std::endl;

        double Lobj = objective_function(L, y1, y2, E1, E2, alpha1, alpha2, k11, k12, k22);
        double Hobj = objective_function(H, y1, y2, E1, E2, alpha1, alpha2, k11, k12, k22);
        if (Lobj < Hobj - tol)
        {
            a2 = L;
        }
        else if (Lobj > Hobj + tol)
        {
            a2 = alpha2;
        }
        else
        {
            a2 = alpha2;
        }
    }

    if (fabs(a2 - alpha2) < (tol * (a2 + alpha2 + tol)))
    {
        return 0;
    }

    double a1 = alpha1 + s * (alpha2 - a2);
    double b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + b;
    double b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + b;

    // if ((a1 > 0) && (a1 < C))
    // {
    //     b = b1;
    // }
    // else if ((a2 > 0) && (a2 < C))
    // {
    //     b = b2;
    // }
    // else
    // {
    //     b = 0.5 * (b2+b1);
    // }
    b = 0.5 * abs(b2 - b1);

    alpha[i1] = a1;
    alpha[i2] = a2;
    vector<int> non_bound_indices = get_conditioned_indeces<double>(alpha, non_bound_check);

    // for (auto &ind : non_bound_indices)
    // {
    //     errors[ind] = compute_error(ind);
    // }
    errors[i1] = compute_error(i1);
    errors[i2] = compute_error(i2);

    return 1;
}

int QPSolver::examine_example(int i2)
{
    double y2 = y_[i2];
    double alpha2 = alpha[i2];
    double E2 = errors[i2];
    double r2 = E2 * y2;
    if (((r2 < -tol) && (alpha2 < C)) || ((r2 > tol) && (alpha2 > 0)))
    {
        bool has_non_bound = false;
        for (auto &a : alpha)
        {
            if ((a > 0) && (a < C))
            {
                has_non_bound = true;
            }
        }
        if (has_non_bound)
        {
            int i1 = second_choice_heuristic(i2);
            if (take_step(i1, i2))
            {
                return 1;
            }
        }
        static double C_ = C;
        static double tol_ = tol;

        auto non_bound_check = [](double a)
        {if (a > 0 && a < C_ - tol_){return true;}return false; };
        vector<int> non_bound_indices = get_conditioned_indeces<double>(alpha, non_bound_check);

        vector<int> random_permuations = permutation<int>(non_bound_indices);
        for (auto &i1 : random_permuations)
        {
            if (take_step(i1, i2))
            {
                return 1;
            }
        }
        random_permuations = permutation(static_cast<int>(alpha.size()));
        for (auto &i1 : random_permuations)
        {
            if (take_step(i1, i2))
            {
                return 1;
            }
        }
    }
    return 0;
}

int QPSolver::second_choice_heuristic(int i2)
{
    static double C_ = C;
    static double tol_ = tol;
    auto non_bound_check = [](double a)
    {if (a > 0 && a < C_ - tol_){return true;}return false; };
    vector<int> non_bound_indices = get_conditioned_indeces<double>(alpha, non_bound_check);

    if (non_bound_indices.size() > 1)
    {

        double max_delta = 0;
        int best_num = -1;
        for (auto &i1 : non_bound_indices)
        {
            if (i1 == i2)
            {
                continue;
            }

            double delta = fabs(errors[i2] - errors[i1]);
            if (delta > max_delta)
            {
                max_delta = delta;
                best_num = i1;
            }
        }
        if (best_num != -1)
        {
            return best_num;
        }
    }

    return random_choice(alpha.size());
}

double QPSolver::objective_function(double a2, double y1, double y2, double E1, double E2, double alpha1, double alpha2, double k11, double k12, double k22)
{
    double f1 = y1 * (E1 + b) - alpha1 * k11 - y1 * y2 * alpha2 * k12;
    double f2 = y2 * (E2 + b) - y1 * y2 * alpha1 * k12 - alpha2 * k22;
    double D = alpha1 + y1 * y2 * (alpha2 - a2);
    double obj = (D * f1) + (a2 * f2) + (0.5 * (D * D) * k11) + 0.5 * a2 * k22 + y1 * y2 * a2 * D * k12;

    return obj;
}

double QPSolver::compute_error(int i)
{
    return output(X_[i]) - y_[i];
}

double QPSolver::output(vector<double> x)
{
    double ans = 0;
    for (int i = 0; i < n_samples; i++)
    {
        ans += y_[i] * alpha[i] * kernel_(x, X_[i]);
    }
    return ans - b;
}

vector<double> QPSolver::output(vector<vector<double>> x)
{
    vector<double> ans;
    for (int i = 0; i < x.size(); i++){
        ans.push_back(output(x[i]));
    }
    return ans;
}