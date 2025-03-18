import numpy as np
from sklearn.base import ClassifierMixin
# from source.kernel import Kernel
#from source.smo import


#
# class NH:
#     def __init__(self, a: float, b: float):
#         self.a = a
#         self.b = b
#
#     def __contains__(self, num: float):
#         if num > self.a and num < self.b:
#             return True
#         return False

class Kernel:
    def __init__(self, kernel='linear', degree=2, gamma=1.0):
        self.gamma = gamma
        self.degree = degree
        self.kernel_type = kernel
        self.kernels = {
            'linear': self.linear_kernel,
            'polynomial': self.polynomial_kernel,
            'rbf': self.rbf_kernel
        }

    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray):
        return np.dot(x1, x2.T)

    def polynomial_kernel(self, x1: np.ndarray, x2: np.ndarray):
        return (np.dot(x1, x2.T) + 1) ** self.degree

    def rbf_kernel(self, x1: np.ndarray, x2: np.ndarray):
        x=[]
        for i in range(len(x1)):
            x.append(x1[i] - x2[i])
        norm = np.linalg.norm(x)
        return np.exp(self.gamma*norm)




    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        if self.kernel_type not in self.kernels:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        return self.kernels[self.kernel_type](x1, x2)


class SVM(ClassifierMixin):
    def __init__(self, kernel='rbf', max_iter=1000, tol=1e-3, C=1.0, degree=2, gamma=1.0, eps=0.01):
        self.tol = tol
        self.C = C
        self.max_iter = max_iter
        self.b = 0
        self.eps = eps
        self.alpha = None
        self.errors = None
        self.degree = degree
        self.gamma = gamma
        self.kernel = Kernel(kernel=kernel, degree=degree, gamma=gamma)

    def take_step(self, i1, i2):
        if i1 == i2:
            return 0

        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]

        y1 = self.y[i1]
        y2 = self.y[i2]

        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = self.y[i1] * self.y[i2]

        L = 0
        H = 0



        if y1 == y2:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        elif y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)

        if L == H:
            return 0

        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])


        a2 = 0
        eta = k11 + k22 - 2 * k12
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            Lobj = self.objective_function(L, y1, y2, E1, E2, alpha1, alpha2, k11, k12, k22)
            Hobj = self.objective_function(H, y1, y2, E1, E2, alpha1, alpha2, k11, k12, k22)
            if Lobj < Hobj - self.eps:
                a2 = L
            elif Lobj > Hobj + self.tol:
                a2 = alpha2
            else:
                a2 = alpha2

        if (abs(a2 - alpha2)) < (self.tol * (a2 + alpha1 + self.tol)):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b

        if (a1 > 0) and (a1 < self.C):
            self.b = b1
        elif (a2 > 0) and (a2 < self.C):
            self.b = b2
        else:
            self.b = 0.5 * (b1 + b2)

        self.alpha[i1] = a1
        self.alpha[i2] = a2

        self.errors[i1] = self.predict(self.X[i1]) - y1
        self.errors[i2] = self.predict(self.X[i2]) - y2
        self.errors = self.predict(self.X) - self.y



        return 1

    def examine_example(self, i2):
        y2 = self.y[i2]
        alpha2 = self.alpha[i2]
        E2 = self.errors[i2]
        r2= E2*y2
        if ( (r2 < -self.tol) & (alpha2 < self.C) ) | ((r2 > self.tol) & (alpha2 > 0) ):
            if np.sum((self.alpha > 0) & (self.alpha < self.C)) > 1:
                i1 = self.second_choice_heuristic(i2)
                if self.take_step(i1, i2):
                    return 1

            for i1 in np.random.permutation(np.where((self.alpha > 0) & (self.alpha < self.C))[0]):
                if self.take_step(i1, i2):
                    return 1
            for i1 in np.random.permutation(range(len(self.alpha))):
                if self.take_step(i1, i2):
                    return 1
        return 0

    def second_choice_heuristic(self,i2):
        non_bound_indices = [i for i in range(len(self.alpha)) if 0 < self.alpha[i] < self.C]
        if len(non_bound_indices) > 1:
            max_delta = 0
            best_num = -1
            for i1 in non_bound_indices:
                if i1 == i2:
                    continue
                delta = abs(self.errors[i2] - self.errors[i1])
                if delta > max_delta:
                    max_delta = delta
                    best_num = i1
            return best_num
        else:
            indices = [i for i in range(len(self.alpha)) if i != i2]
            return np.random.choice(indices)

    def objective_function(self, a2, y1, y2, E1, E2, alpha1, alpha2, k11, k12, k22):
        f1 = y1 * (E1 + self.b) - alpha1 * k11 - y1 * y2 * alpha2 * k12
        f2 = y2 * (E2 + self.b) - y1 * y2 * alpha1 * k12 - alpha2 * k22
        D = alpha1 + y1 * y2 * (alpha2 - a2)
        obj = (D * f1)+ (a2 * f2) + (0.5 * (D**2) * k11) + 0.5 * a2 * k22+ y1 * y2 * a2 * D * k12

        return obj

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.alpha = np.zeros(self.n_samples)
        self.errors = self.predict(self.X) - self.y

        num_changed = 0
        examine_all = 1
        iter_count = 0

        while (num_changed > 0 or examine_all) and iter_count < self.max_iter:
            num_changed = 0
            if examine_all:
                for i in range(self.n_samples):
                    num_changed += self.examine_example(i)
            else:
                for i in np.where((self.alpha > 0) & (self.alpha < self.C))[0]:
                    num_changed += self.examine_example(i)
            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1
            iter_count += 1

        # solve=SMO()

    def predict(self, X: np.ndarray):
        # kernel = Kernel(kernel='linear')
        ans = 0
        for i in range(self.n_samples):
            ans += self.y[i] * self.alpha[i] * self.kernel(self.X[i], X)
        ans -= self.b

        return np.sign(ans)

    def support(self):
        return self.X[self.alpha > 0]


# class Plots:
