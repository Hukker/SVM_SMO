#ifndef QPSOLVER_H
#define QPSOLVER_H
#include <vector>
#include <string>
#include "kernel.h"

using std::string;
using std::vector;

class QPSolver
{
public:
    QPSolver(vector<vector<double>> X, vector<double> y, string kernel, double C, double tol, int max_iter, bool logs=false) : X_(X), y_(y), alpha(X.size(), 0), errors(X.size(), 0), kernel_(kernel)
    {
        this->b = 0;
        this->C = C;
        this->max_iter = max_iter;
        this->tol = tol;
        n_samples = X.size();
        for (int i = 0; i < n_samples; i++)
        {
            errors[i] = -y_[i];
        }
        this->logs = logs;
    };

    void solve();

    vector<double> get_alpha();

    double get_b();

    double output(vector<double> x);
    vector<double> output(vector<vector<double>> x);


private:
    vector<double> alpha;
    double b;

    vector<vector<double>> X_;
    vector<double> y_;
    vector<double> errors;

    Kernel kernel_;

    double C;
    int max_iter;
    double tol;
    bool logs;
    int n_samples;

    int take_step(int i1, int i2);

    int examine_example(int i2);

    int second_choice_heuristic(int i2);

    double objective_function(double a2, double y1, double y2, double E1, double E2, double alpha1, double alpha2, double k11, double k12, double k22);

    double compute_error(int i);
};
#include "smo.cpp"
#endif