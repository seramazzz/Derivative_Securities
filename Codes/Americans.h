#ifndef AMERICANS_H
#define AMERICANS_H

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <utility>

/**
 * @brief Computes the Chiarella–Ziogas (CZ) characteristic function for American option valuation.
 *
 * This function implements the affine characteristic function structure under
 * stochastic volatility (e.g., Heston-type) dynamics, with two characteristic
 * function forms depending on FunNum (1 or 2). Used internally in pricing routines.
 *
 * @param theta   Current underlying price
 * @param T       Option maturity
 * @param t       Current time
 * @param kappa   Mean reversion rate
 * @param v_bar   Long-term variance mean
 * @param xi      Volatility of variance
 * @param sigma   Spot volatility
 * @param v       Instantaneous variance
 * @param rho     Correlation between Brownian motions
 * @param lambda_ Risk premium adjustment
 * @param K       Strike price
 * @param r       Risk-free rate
 * @param q       Dividend yield
 * @param phi     Characteristic exponent (complex)
 * @param psi     Auxiliary variable (complex)
 * @param FunNum  Characteristic function index (1 or 2)
 * @return Complex-valued characteristic function value
 */
std::complex<double> CZCharFun(double theta, double T, double t, double kappa, double v_bar, double xi, double sigma, double v, double rho, double lambda_, double K, double r, double q, std::complex<double> phi, std::complex<double> psi, int FunNum) {
    // Define useful constants
    using namespace std;
    const complex<double> I(0.0, 1.0); // Imaginary unit

    // Log of the stock price
    double x = log(theta);

    // Parameters "a" and "b"
    double a = kappa * v_bar;
    double b = kappa + lambda_;

    // "d" and "g" functions
    complex<double> d = sqrt(pow(rho * xi * I * phi - b, 2.0) + xi * xi * phi * (phi + I));
    complex<double> g = (b - rho * xi * I * phi - xi * xi * I * psi + d) / (b - rho * xi * I * phi - xi * xi * I * psi - d);

    // Affine characteristic function components for Trap = 1
    complex<double> c = 1.0 / g;
    complex<double> G = (1.0 - c * exp(-d * (T-t))) / (1.0 - c);
    complex<double> D = I * psi + (b - rho * xi * I * phi - xi * xi * I * psi - d) / (xi * xi) * (1.0 - exp(-d * (T - t))) / (1.0 - c * exp(-d * (T - t)));
    complex<double> C = (r - q - 0.5 * sigma * sigma) * I * phi * (T - t) - 0.5 * phi * phi * sigma * sigma * (T - t) + a / (xi * xi) * ((b - rho * xi * I * phi - d) * (T - t) - 2.0 * log(G));

    // The second characteristic function
    complex<double> f2 = exp(C + D * v + I * phi * x);

    if (FunNum == 2) {
        return f2;
    } else {
        // Redefine "d" and "g" for FunNum = 1
        d = sqrt(pow(rho * xi * I * (phi - I) - b, 2.0) + xi * xi * (phi - I) * phi);
        g = (b - rho * xi * I * (phi - I) - xi * xi * I * psi + d) / (b - rho * xi * I * (phi - I) - xi * xi * I * psi - d);

        // Update affine characteristic function components
        c = 1.0 / g;
        G = (1.0 - c * exp(-d * (T-t))) / (1.0 - c);
        C = (r - q - 0.5 * sigma * sigma) * I * (phi - I) * (T - t) - 0.5 * pow(phi - I, 2.0) * sigma * sigma * (T - t) + a / (xi * xi) * ((b - rho * xi * I * (phi - I) - d) * (T - t) - 2.0 * log(G));
        D = I * psi + (b - rho * xi * I * (phi - I) - xi * xi * I * psi - d) / (xi * xi) * (1.0 - exp(-d * (T - t))) / (1.0 - c * exp(-d * (T - t)));

        // The second characteristic function with phi = phi - I
        complex<double> F2 = exp(C + D * v + I * (phi - I) * x);

        // The first characteristic function
        return (1.0 / theta) * exp(-(r - q - 0.5 * sigma * sigma) * I * (phi - I) * (T - t) + (r - q + 0.5 * sigma * sigma) * I * phi * (T - t) - 0.5 * sigma * sigma * (T - t) - I * phi * sigma * sigma * (T - t)) * F2;
    }
}

/**
 * @brief Computes European option prices (call or put) under the Chiarella–Ziogas framework.
 *
 * Uses numerical integration of the characteristic function (two integrals)
 * to compute risk-neutral probabilities P1 and P2, analogous to the Heston model.
 *
 * @param PutCall "C" for Call, "P" for Put
 * @param theta   Current price
 * @param T       Maturity
 * @param kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q : Model parameters
 * @param x, w    Integration nodes and weights
 * @return Option price (double)
 */
double CZEuroOption(const std::string& PutCall, double theta, double T, double kappa, double v_bar, double xi, double sigma, double v, double rho, double lambda_, double K, double r, double q, const std::vector<double>& x, const std::vector<double>& w) {
    using namespace std;
    const complex<double> I(0.0, 1.0);

    // Create the integrands
    vector<double> Int1(x.size(), 0.0);
    vector<double> Int2(x.size(), 0.0);

    for (size_t k = 0; k < x.size(); ++k) {
        complex<double> phi(x[k], 0.0);
        Int1[k] = w[k] * real(exp(-I * phi * log(K)) * CZCharFun(theta, T, 0, kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q, phi, 0.0, 1) / (I * phi));
        Int2[k] = w[k] * real(exp(-I * phi * log(K)) * CZCharFun(theta, T, 0, kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q, phi, 0.0, 2) / (I * phi));
    }

    // Define the probabilities
    double P1 = 0.5 + (1.0 / M_PI) * accumulate(Int1.begin(), Int1.end(), 0.0);
    double P2 = 0.5 + (1.0 / M_PI) * accumulate(Int2.begin(), Int2.end(), 0.0);

    // Calculate the call or put price
    double y = 0.0;
    if (PutCall == "C") {
        y = theta * exp(-q * T) * P1 - K * exp(-r * T) * P2;
    } else if (PutCall == "P") {
        y = K * exp(-r * T) * (1.0 - P2) - theta * exp(-q * T) * (1.0 - P1);
    }

    return y;
}


/**
 * @brief Double trapezoidal integration scheme for early exercise integral computation.
 *
 * Evaluates nested integrals appearing in the early exercise premium calculation
 * using a 2D trapezoidal rule.
 *
 * @param S0, K, tau, rf, q, b0, b1 : Standard pricing parameters
 * @param X, T                      : Integration grids
 * @param funNum                    : Function number (1 or 2)
 * @param kappa, v_bar, xi, sigma, v, rho, lambda_ : Model parameters
 * @return Double integral approximation (double)
 */
double DoubleTrapezoidal(double S0, double K, double tau, double rf, double q,
                         double b0, double b1, const std::vector<double>& X, const std::vector<double>& T,
                         int funNum, double kappa, double v_bar, double xi, double sigma,
                         double v, double rho, double lambda_) {
    using namespace std;
    const complex<double> I(0.0, 1.0); // Imaginary unit

    // Select rate or dividend
    double rq = (funNum == 1) ? q : rf;

    size_t Nt = T.size();
    size_t Nx = X.size();
    vector<vector<double> > Int(Nt, vector<double>(Nx, 0.0));
    double sumInt = 0.0;

    for (size_t t = 1; t < Nt; ++t) {
        double a = T[t - 1];
        double b = T[t];

        for (size_t x = 1; x < Nx; ++x) {
            double c = X[x - 1];
            double d = X[x];

            // Compute terms
            complex<double> fun1 = exp(-b0 * I * c) * CZCharFun(S0, tau, a, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, c, -b1 * c, funNum) / (I * c);
            complex<double> fun2 = exp(-b0 * I * d) * CZCharFun(S0, tau, a, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, d, -b1 * d, funNum) / (I * d);
            complex<double> fun3 = exp(-b0 * I * c) * CZCharFun(S0, tau, b, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, c, -b1 * c, funNum) / (I * c);
            complex<double> fun4 = exp(-b0 * I * d) * CZCharFun(S0, tau, b, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, d, -b1 * d, funNum) / (I * d);

            double term1 = exp(rq * a) * real(fun1) + exp(rq * a) * real(fun2) +
                           exp(rq * b) * real(fun3) + exp(rq * b) * real(fun4);

            complex<double> h1 = exp(-b0 * I * c) * CZCharFun(S0, tau, (a + b) / 2.0, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, c, -b1 * c, funNum) / (I * c);
            complex<double> h2 = exp(-b0 * I * d) * CZCharFun(S0, tau, (a + b) / 2.0, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, d, -b1 * d, funNum) / (I * d);
            complex<double> h3 = exp(-b0 * I * ((c + d) / 2.0)) * CZCharFun(S0, tau, a, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, (c + d) / 2.0, -b1 * (c + d) / 2.0, funNum) / (I * (c + d) / 2.0);
            complex<double> h4 = exp(-b0 * I * ((c + d) / 2.0)) * CZCharFun(S0, tau, b, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, (c + d) / 2.0, -b1 * (c + d) / 2.0, funNum) / (I * (c + d) / 2.0);

            double term2 = exp(rq * (a + b) / 2.0) * (real(h1) + real(h2)) +
                           exp(rq * a) * real(h3) + exp(rq * b) * real(h4);

            complex<double> term3 = exp(-b0 * I * ((c + d) / 2.0)) *
                                    CZCharFun(S0, tau, (a + b) / 2.0, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q,
                                              (c + d) / 2.0, -b1 * (c + d) / 2.0, funNum) / (I * (c + d) / 2.0);
            double term3Real = exp(rq * (a + b) / 2.0) * real(term3);

            Int[t][x] = (b - a) * (d - c) / 16.0 * (term1 + 2.0 * term2 + 4.0 * term3Real);
            sumInt += Int[t][x];
        }
    }

    return sumInt;
}

/**
 * @brief Computes a double integral using Gauss–Legendre quadrature.
 *
 * Provides an efficient alternative to the trapezoidal scheme for the
 * early exercise region integral in the Chiarella–Ziogas formulation.
 *
 * @param S0, tau, K, rf, q, b0, b1 : Model parameters
 * @param xt, wt, xs, ws            : Quadrature nodes and weights
 * @param a, b, c, d                : Integration bounds
 * @param funNum                    : Function number (1 or 2)
 * @param kappa, v_bar, xi, sigma, v, rho, lambda_ : Model parameters
 * @return Computed integral value
 */
double DoubleGaussLegendre(double S0, double tau, double K, double rf, double q, double b0, double b1, const std::vector<double>& xt, const std::vector<double>& wt, const std::vector<double>& xs, const std::vector<double>& ws, double a, double b, double c, double d, int funNum, double kappa, double v_bar, double xi, double sigma, double v, double rho, double lambda_) {
    using namespace std;
    const complex<double> I(0.0, 1.0);

    double h1 = (b - a) / 2.0;
    double h2 = (b + a) / 2.0;
    double k1 = (d - c) / 2.0;
    double k2 = (d + c) / 2.0;

    double y = 0.0;

    double qr = (funNum == 1) ? q : rf;

    for (size_t t = 0; t < xt.size(); ++t) {
        double time = h1 * xt[t] + h2;
        for (size_t x = 0; x < xs.size(); ++x) {
            double phi = k1 * xs[x] + k2;
            complex<double> fun = exp(qr * time) * real(exp(-b0 * I * phi) * CZCharFun(S0, tau, time, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, phi, -b1 * phi, funNum) / (I * phi));
            y += h1 * k1 * wt[t] * ws[x] * real(fun);
        }
    }

    return y;
}

/**
 * @brief Computes the early exercise premium (EEP) for American options.
 *
 * Aggregates results from the chosen double integration scheme (trapezoidal or
 * Gauss–Legendre) to evaluate the early exercise component in the CZ approach.
 *
 * @param PutCall "C" or "P"
 * @param S0, tau, K, rf, q : Standard option parameters
 * @param xs, ws, xt, wt    : Integration nodes and weights
 * @param Nt, b0, b1, a, b, c, d : Numerical grid parameters
 * @param DoubleType        : Integration method ("Trapz" or "GLe")
 * @param kappa, v_bar, xi, sigma, v, rho, lambda_ : Model parameters
 * @return Early exercise premium (double)
 */
double CZEarlyExercise(const std::string& PutCall, double S0, double tau, double K, double rf, double q,
                       const std::vector<double>& xs, const std::vector<double>& ws,
                       const std::vector<double>& xt, const std::vector<double>& wt,
                       int Nt, double b0, double b1, double a, double b, double c, double d,
                       const std::string& DoubleType, double kappa, double v_bar, double xi,
                       double sigma, double v, double rho, double lambda_) {
    using namespace std;

    double Int1 = 0.0;
    double Int2 = 0.0;

    // Select integration method
    if (DoubleType == "GLe") {
        Int1 = DoubleGaussLegendre(S0, tau, K, rf, q, b0, b1, xt, wt, xt, wt, a, b, c, d, 1, kappa, v_bar, xi, sigma, v, rho, lambda_);
        Int2 = DoubleGaussLegendre(S0, tau, K, rf, q, b0, b1, xt, wt, xt, wt, a, b, c, d, 2, kappa, v_bar, xi, sigma, v, rho, lambda_);
    } else if (DoubleType == "Trapz") {
        double ht = (b - a) / Nt;
        double hs = (d - c) / Nt;
        vector<double> X(Nt + 1);
        vector<double> T(Nt + 1);
        for (int j = 0; j <= Nt; ++j) {
            T[j] = a + j * ht;
            X[j] = c + j * hs;
        }
        Int1 = DoubleTrapezoidal(S0, K, tau, rf, q, b0, b1, X, T, 1, kappa, v_bar, xi, sigma, v, rho, lambda_);
        Int2 = DoubleTrapezoidal(S0, K, tau, rf, q, b0, b1, X, T, 2, kappa, v_bar, xi, sigma, v, rho, lambda_);
    }

    // Declare V1 and V2 outside of the condition
    double V1 = 0.0;
    double V2 = 0.0;

    // Calculate V1 and V2 based on PutCall
    const double PI = acos(-1.0); // More accurate value of pi

    if (PutCall == "C") { // Call option
        V1 = S0 * (1.0 - exp(-q * tau)) / 2.0 + (1.0 / PI) * S0 * q * exp(-q * tau) * Int1;
        V2 = K * (1.0 - exp(-rf * tau)) / 2.0 + (1.0 / PI) * K * rf * exp(-rf * tau) * Int2;
    } else if (PutCall == "P") { // Put option
        V1 = K * (1.0 - exp(-rf * tau)) / 2.0 - (1.0 / PI) * K * rf * exp(-rf * tau) * Int2;
        V2 = S0 * (1.0 - exp(-q * tau)) / 2.0 - (1.0 / PI) * S0 * q * exp(-q * tau) * Int1;
    }

    return V1 - V2;
}

/**
 * @brief Computes the total American option price under the CZ model.
 *
 * Combines the European price and the early exercise premium to obtain
 * the final American option value.
 *
 * @param PutCall "C" or "P"
 * @param S0, tau, K, rf, q : Standard option parameters
 * @param xs, ws, xt, wt, Nt : Integration nodes, weights, and grid steps
 * @param b0, b1, a, b, c, d : Domain and numerical parameters
 * @param DoubleType         : Integration method ("Trapz" or "GLe")
 * @param kappa, v_bar, xi, sigma, v, rho, lambda_ : Model parameters
 * @return American option price (double)
 */
double CZAmerOption(const std::string& PutCall, double S0, double tau, double K, double rf, double q,
                    const std::vector<double>& xs, const std::vector<double>& ws,
                    const std::vector<double>& xt, const std::vector<double>& wt, int Nt,
                    double b0, double b1, double a, double b, double c, double d,
                    const std::string& DoubleType, double kappa, double v_bar, double xi,
                    double sigma, double v, double rho, double lambda_) {
    // Calculate the European option price
    double Euro = CZEuroOption(PutCall, S0, tau, kappa, v_bar, xi, sigma, v, rho, lambda_, K, rf, q, xs, ws);
    
    // Calculate the early exercise premium
    double Premium = CZEarlyExercise(PutCall, S0, tau, K, rf, q, xs, ws, xt, wt, Nt,
                                     b0, b1, a, b, c, d, DoubleType, kappa, v_bar, 
                                     xi, sigma, v, rho, lambda_);

    // Return the American option price
    double Amer = Euro + Premium;

    return Amer;
}

/**
 * @brief Newton’s method root finder used in the CZ algorithm to locate optimal b0, b1 boundaries.
 *
 * Solves nonlinear equations for the free boundaries in the Chiarella–Ziogas
 * iterative scheme for early exercise surface computation.
 *
 * @param PutCall "C" or "P"
 * @param start   Starting guess for boundary
 * @param v0, v1  Variance values at nodes
 * @param T       Maturity
 * @param kappa, v_bar, lambda_, rho, xi, sigma : Model parameters
 * @param K, r, q : Financial parameters
 * @param xs, ws, xt, wt, Nt : Integration nodes and parameters
 * @param B0, B1  Boundary coefficients
 * @param g_num   Equation index (1 or 2)
 * @param tol     Tolerance for convergence
 * @param A, B, C, D : Integration limits
 * @param DoubleType : "Trapz" or "GLe"
 * @return Converged boundary value b (double)
 */
double cz_newton(const std::string& PutCall, double start, double v0, double v1, double T,
                 double kappa, double v_bar, double lambda_, double rho, double xi, double sigma,
                 double K, double r, double q, const std::vector<double>& xs,
                 const std::vector<double>& ws, const std::vector<double>& xt,
                 const std::vector<double>& wt, int Nt, double B0, double B1, int g_num,
                 double tol, double A, double B, double C, double D, const std::string& DoubleType) {
    const double db = 0.001;
    double diff = 1.1 * tol;
    double b = start;

    while (std::abs(diff) > tol) {
        double g0 = 0.0, g_plus_db = 0.0, g_minus_db = 0.0;

        //std::cout << "Call_01 =" << CallA1 << ": log-term = " << CallA1 + K  << ", g0 = " << g0
        //            << " Call_02 =" << CallA2 << ": log-term = " << CallA2 + K  << ", g = " << g
        //            << " Call_03 =" << CallA3 << ": log-term = " << CallA3 + K  << ", g0 = " << g_ << std::endl;

        if (PutCall == "C") { // Call option
            if (g_num == 1) {
                double opt_a1 = CZAmerOption(PutCall, std::exp(B0 + b * v0), T, K, r, q, xs, ws, xt, wt, Nt,
                                             B0, b, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v0, rho, lambda_);
                //g0 = (std::log(opt_a1 + K) - B0) / v0 - b;
                g0 = opt_a1 + K - std::exp(B0 + b * v0);

                /*
                std::cout << "Call_11 = " << opt_a1 
                          << ": log-term = " << (opt_a1 + K) 
                          << ", g0 = " << g0
                << std::endl;

                std::cout << "PutCall = " << PutCall 
                          << ", S0 = " << std::exp(B0 + b * v0)
                          << ", T = " << T 
                          << ", K = " << K 
                          << ", r = " << r 
                          << ", q = " << q 
                          << ", Nt = " << Nt
                          << ", b0 = " << B0 
                          << ", b1 = " << b 
                          << ", a = " << A 
                          << ", b = " << B
                          << ", c = " << C
                          << ", d = " << D
                          << ", DoubleType = " << DoubleType
                          << ", kappa = " << kappa
                          << ", v_bar = " << v_bar
                          << ", xi = " << xi
                          << ", sigma = " << sigma
                          << ", v = " << v0
                          << ", rho = " << rho
                          << ", lambda_ = " << lambda_
                << std::endl;
                */
                double opt_a1_plus_db = CZAmerOption(PutCall, std::exp(B0 + (b + db) * v0), T, K, r, q, xs, ws, xt, wt, Nt,
                                                     B0, b + db, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v0, rho, lambda_);
                //g_plus_db = (std::log(opt_a1_plus_db + K) - B0) / v0 - (b + db);
                g_plus_db = opt_a1_plus_db + K - std::exp(B0 + (b + db) * v0);

                double opt_a1_minus_db = CZAmerOption(PutCall, std::exp(B0 + (b - db) * v0), T, K, r, q, xs, ws, xt, wt, Nt,
                                                      B0, b - db, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v0, rho, lambda_);
                //g_minus_db = (std::log(opt_a1_minus_db + K) - B0) / v0 - (b - db);
                g_minus_db = opt_a1_minus_db + K - std::exp(B0 + (b - db) * v0);
            } else { // g_num == 2
                double opt_a0 = CZAmerOption(PutCall, std::exp(b + B1 * v1), T, K, r, q, xs, ws, xt, wt, Nt,
                                             b, B1, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v1, rho, lambda_);
                //g0 = (std::log(opt_a0 + K) - v1 * B1) - b;
                g0 = opt_a0 + K - std::exp(b + B1 * v1);

                double opt_a0_plus_db = CZAmerOption(PutCall, std::exp(b + db + B1 * v1), T, K, r, q, xs, ws, xt, wt, Nt,
                                                     b + db, B1, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v1, rho, lambda_);
                //g_plus_db = (std::log(opt_a0_plus_db + K) - v1 * B1) - (b + db);
                g_plus_db = opt_a0_plus_db + K - std::exp((b + db) + B1 * v1);

                double opt_a0_minus_db = CZAmerOption(PutCall, std::exp(b - db + B1 * v1), T, K, r, q, xs, ws, xt, wt, Nt,
                                                      b - db, B1, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v1, rho, lambda_);
                //g_minus_db = (std::log(opt_a0_minus_db + K) - v1 * B1) - (b - db);
                g_minus_db = opt_a0_minus_db + K - std::exp((b - db) + B1 * v1);
            }
        } else if (PutCall == "P") { // Put option
            if (g_num == 1) {
                double opt_a1 = CZAmerOption(PutCall, std::exp(B0 + b * v0), T, K, r, q, xs, ws, xt, wt, Nt,
                                             B0, b, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v0, rho, lambda_);
                opt_a1 = std::min(K - 1e-9, opt_a1);
                g0 = (std::log(K - opt_a1) - B0) / v0 - b;

                double opt_a1_plus_db = CZAmerOption(PutCall, std::exp(B0 + (b + db) * v0), T, K, r, q, xs, ws, xt, wt, Nt,
                                                     B0, b + db, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v0, rho, lambda_);
                opt_a1_plus_db = std::min(K - 1e-9, opt_a1_plus_db);
                g_plus_db = (std::log(K - opt_a1_plus_db) - B0) / v0 - (b + db);

                double opt_a1_minus_db = CZAmerOption(PutCall, std::exp(B0 + (b - db) * v0), T, K, r, q, xs, ws, xt, wt, Nt,
                                                      B0, b - db, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v0, rho, lambda_);
                opt_a1_minus_db = std::min(K - 1e-9, opt_a1_minus_db);
                g_minus_db = (std::log(K - opt_a1_minus_db) - B0) / v0 - (b - db);
            } else { // g_num == 2
                double opt_a0 = CZAmerOption(PutCall, std::exp(b + B1 * v1), T, K, r, q, xs, ws, xt, wt, Nt,
                                             b, B1, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v1, rho, lambda_);
                opt_a0 = std::min(K - 1e-9, opt_a0);
                g0 = (std::log(K - opt_a0) - v1 * B1) - b;

                double opt_a0_plus_db = CZAmerOption(PutCall, std::exp(b + db + B1 * v1), T, K, r, q, xs, ws, xt, wt, Nt,
                                                     b + db, B1, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v1, rho, lambda_);
                opt_a0_plus_db = std::min(K - 1e-9, opt_a0_plus_db);
                g_plus_db = (std::log(K - opt_a0_plus_db) - v1 * B1) - (b + db);

                double opt_a0_minus_db = CZAmerOption(PutCall, std::exp(b - db + B1 * v1), T, K, r, q, xs, ws, xt, wt, Nt,
                                                      b - db, B1, A, B, C, D, DoubleType, kappa, v_bar, xi, sigma, v1, rho, lambda_);
                opt_a0_minus_db = std::min(K - 1e-9, opt_a0_minus_db);
                g_minus_db = (std::log(K - opt_a0_minus_db) - v1 * B1) - (b - db);
            }
        }

        // Derivative of g using central difference
        double dg = (g_plus_db - g_minus_db) / (2 * db);
        // Newton's method update
        double b_new = b - g0 / dg;
        diff = b_new - b;
        b = b_new;
    }

    return b;
}

/**
 * @brief Computes the conditional expectation of variance under mean reversion.
 *
 * @param vs     Initial variance
 * @param v_bar  Long-term variance mean
 * @param kappa  Mean reversion speed
 * @param t, s   Time variables
 * @return Conditional expected variance E[v_t | v_s]
 */
double EV(double vs, double v_bar, double kappa, double t, double s) {
    return v_bar + (vs - v_bar) * std::exp(-kappa * (t - s));
}


/**
 * @brief Iteratively determines the early exercise boundaries b0(t) and b1(t) for American options.
 *
 * Implements the recursive Chiarella–Ziogas boundary computation using Newton’s method
 * over time partitions, returning the boundary pairs (b0, b1).
 *
 * @param PutCall "C" or "P"
 * @param T, kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q : Model parameters
 * @param V00, V10, b00, b10 : Initial variance and boundary values
 * @param xs, ws, xt, wt, Nt, NT : Integration parameters and time partitions
 * @param tol0, tol1, Ntol : Tolerances
 * @param a, b, c, d : Integration limits
 * @param DoubleType : "Trapz" or "GLe"
 * @return Pair of vectors {b0, b1} containing early exercise boundaries
 */
std::pair<std::vector<double>, std::vector<double> > findB(const std::string& PutCall, double T,
                                                          double kappa, double v_bar, double xi,
                                                          double sigma, double v, double rho,
                                                          double lambda_, double K, double r,
                                                          double q, double V00, double V10,
                                                          double b00, double b10,
                                                          const std::vector<double>& xs,
                                                          const std::vector<double>& ws,
                                                          const std::vector<double>& xt,
                                                          const std::vector<double>& wt, int Nt,
                                                          int NT, double tol0, double tol1,
                                                          double Ntol, double a, double b, double c,
                                                          double d, const std::string& DoubleType) {
    using namespace std;

    // Initialize variables
    vector<double> v0(NT), v1(NT), b0(NT), b1(NT);
    v0[0] = V00;
    v1[0] = V10;
    b0[0] = b00;
    b1[0] = b10;

    double dT = T / NT;

    // Logging headers
    cout << "  n       T           v0(n)       v1(n)       b0(n)       b1(n)   NumSteps" << endl;
    cout << "--------------------------------------------------------------------------" << endl;
    cout << fixed << std::setprecision(6);
    cout << setw(4) << 1 << setw(12) << dT << setw(12) << v0[0] << setw(12) << v1[0]
         << setw(12) << b0[0] << setw(12) << b1[0] << endl;

    // Iterate over maturity steps
    for (int n = 1; n < NT; ++n) {
        double T_n = (n + 1) * dT;

        // Variances
        double Evt_n = EV(v, v_bar, kappa, T, T_n);
        v0[n] = Evt_n + xi / kappa * sqrt(kappa * v_bar / 2.0);
        v1[n] = Evt_n - xi / kappa * sqrt(kappa * v_bar / 2.0);

        // Starting values for Newton's method
        double b0k_ = b0[n - 1];
        double b1k_ = b1[n - 1];

        // Iterative Newton's method
        int counter = 0;
        double diff0 = 1.1 * tol0;
        double diff1 = 1.1 * tol1;

        while ((diff0 > tol0) && (diff1 > tol1)) {
            ++counter;

            // Newton's method for b1k
            double b1k = cz_newton(PutCall, b1k_, v0[n], v1[n], T_n, kappa, v_bar, lambda_, rho, xi,
                                   sigma, K, r, q, xs, ws, xt, wt, Nt, b0k_, b1k_, 1, Ntol, a, b, c, d, DoubleType);
                                   

            // Newton's method for b0k
            double b0k = cz_newton(PutCall, b0k_, v0[n], v1[n], T_n, kappa, v_bar, lambda_, rho, xi,
                                   sigma, K, r, q, xs, ws, xt, wt, Nt, b0k_, b1k, 0, Ntol, a, b, d, c, DoubleType);

            b0[n] = b0k;
            b1[n] = b1k;

            // Update differences for tolerance
            diff0 = abs(b0k_ - b0k);
            diff1 = abs(b1k_ - b1k);

            b0k_ = b0k;
            b1k_ = b1k;

        }

        // Log current step
        cout << setw(4) << n + 1 << setw(12) << T_n << setw(12) << v0[n] << setw(12) << v1[n]
             << setw(12) << b0[n] << setw(12) << b1[n] << setw(8) << counter << endl;
    }

    cout << "--------------------------------------------------------------------------" << endl;

    // Return b0 and b1
    return std::make_pair(b0, b1);
}

#endif