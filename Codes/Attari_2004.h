#ifndef ATTARI_2004_H
#define ATTARI_2004_H

#include <iostream>
#include <cmath>      // For pow, exp functions
#include <stdexcept>  // For exception handling
#include <iostream>   // Include once for printing (cout)
#include <complex>    // For complex numbers
#include <vector>     // For dynamic arrays (vector)
#include <numeric>    // For std::accumulate
#include <algorithm>  // For std::sort
#include <limits>     // For std::numeric_limits

/**
 * Returns the integrand for the Attari (2004) integrand.
 * Heston model, Fourier transform.
 * 
 * @param phi    : integration variable (complex<double>)
 * @param kappa  : volatility mean reversion speed parameter (double)
 * @param v_bar  : volatility mean reversion level parameter (double)
 * @param lambda_: risk parameter (double)
 * @param rho    : correlation between two Brownian motions (double)
 * @param xi     : volatility of variance (double)
 * @param sigma  : volatility of individual shock (double)
 * @param T      : time to maturity (double)
 * @param K      : strike price (double)
 * @param theta  : aggregate shock (double)
 * @param r      : risk-free rate (double)
 * @param q      : dividend yield (double)
 * @param v      : variance (double)
 * 
 * @return integrand value (double)
 */
double AttariProb(std::complex<double> phi, double kappa, double v_bar, double lambda_, double rho, double xi,
                  double sigma, double T, double K, double theta, double r, double q, double v) {
    
    // Parameters for the second Heston characteristic function (f2)
    double a = kappa * v_bar;
    double u = -0.5;
    double b = kappa + lambda_;
    
    // Calculate 'd' and 'g' (complex intermediate variables)
    std::complex<double> i(0, 1); // Imaginary unit
    std::complex<double> d = std::sqrt(std::pow(rho * xi * i * phi - b, 2) - std::pow(xi, 2) * (2.0 * u * i * phi - std::pow(phi, 2)));
    std::complex<double> g = (b - rho * xi * i * phi + d) / (b - rho * xi * i * phi - d);

    // "Little Heston Trap" formulation (since Trap=1 is always chosen)
    std::complex<double> c = 1.0 / g;
    std::complex<double> G = (1.0 - c * std::exp(-d * T)) / (1.0 - c);
    std::complex<double> C = (r - q + u * std::pow(sigma, 2)) * i * phi * T - 0.5 * std::pow(phi, 2) * std::pow(sigma, 2) * T
            + a / std::pow(xi, 2) * ((b - rho * xi * i * phi - d) * T - 2.0 * std::log(G));
    std::complex<double> D = (b - rho * xi * i * phi - d) / std::pow(xi, 2) * ((1.0 - std::exp(-d * T)) / (1.0 - c * std::exp(-d * T)));

    // The characteristic function for Attari (2004)
    std::complex<double> f = std::exp(C + D * v - i * phi * r * T);

    // The "L" function
    std::complex<double> L = std::log(std::exp(-r * T) * K / theta);

    // The Attari (2004) integrand
    double y = ((std::real(f) + std::imag(f) / phi.real()) * std::cos(std::real(L) * phi.real()) 
                 + (std::imag(f) - std::real(f) / phi.real()) * std::sin(std::real(L) * phi.real())) 
                / (1.0 + std::pow(phi.real(), 2));

    return y;
}
/**
 * Attari (2004) call or put price by Gauss-Laguerre Quadrature.
 * 
 * @param PutCall : 'C' for Call, 'P' for Put (char)
 * @param theta   : aggregate shock (double)
 * @param K       : Strike price (double)
 * @param T       : Time to maturity (double)
 * @param r       : Risk-free rate (double)
 * @param q       : Dividend yield (double)
 * @param kappa   : Heston parameter (mean reversion speed) (double)
 * @param v_bar   : Heston parameter (mean reversion level) (double)
 * @param xi      : Heston parameter (volatility of volatility) (double)
 * @param sigma   : volatility of individual shock (double)
 * @param lambda_ : Heston parameter (risk) (double)
 * @param v       : Initial variance (double)
 * @param rho     : Correlation between asset price and volatility (double)
 * @param x       : Gauss-Laguerre abscissas (std::vector<double>)
 * @param w       : Gauss-Laguerre weights (std::vector<double>)
 * 
 * @return Option price (double)
 */


double AttariPriceGaussLaguerre(char PutCall, double theta, double K, double T, double r, double q, double kappa,
                                double v_bar, double xi, double sigma, double lambda_, double v, double rho) {

    // Gauss-Laguerre abscissas (x) and weights (w) - manually initialized using push_back (calculated in python with numpy)
    std::vector<double> X;
    X.push_back(0.0444893658);
    X.push_back(0.234526110);
    X.push_back(0.576884629);
    X.push_back(1.07244875);
    X.push_back(1.72240878);
    X.push_back(2.52833671);
    X.push_back(3.49221328);
    X.push_back(4.61645675);
    X.push_back(5.90395872);
    X.push_back(7.35812527);
    X.push_back(8.98294783);
    X.push_back(10.7829985);
    X.push_back(12.7637110);
    X.push_back(14.9313497);
    X.push_back(17.2911574);
    X.push_back(19.8603347);
    X.push_back(22.6205942);
    X.push_back(25.6439448);
    X.push_back(28.8539181);
    X.push_back(32.3205798);
    X.push_back(36.1939939);
    X.push_back(39.9836016);
    X.push_back(44.7111660);
    X.push_back(49.0423478);
    X.push_back(54.4478923);
    X.push_back(59.8411301);
    X.push_back(65.9877955);
    X.push_back(72.6879843);
    X.push_back(80.1860031);
    X.push_back(88.7358267);
    X.push_back(98.8294726);
    X.push_back(111.751402);

    std::vector<double> W;
    W.push_back(0.11418711);
    W.push_back(0.26606522);
    W.push_back(0.41879314);
    W.push_back(0.57253285);
    W.push_back(0.72764879);
    W.push_back(0.88453672);
    W.push_back(1.04361888);
    W.push_back(1.20534924);
    W.push_back(1.37022139);
    W.push_back(1.53878017);
    W.push_back(1.71161155);
    W.push_back(1.88945863);
    W.push_back(2.07281121);
    W.push_back(2.26320788);
    W.push_back(2.46444164);
    W.push_back(2.66201387);
    W.push_back(2.87828775);
    W.push_back(3.03299202);
    W.push_back(3.38961163);
    W.push_back(3.48359725);
    W.push_back(3.43546123);
    W.push_back(4.55635927);
    W.push_back(3.8469007);
    W.push_back(5.63079911);
    W.push_back(4.63990579);
    W.push_back(6.39259714);
    W.push_back(6.38441924);
    W.push_back(7.03996479);
    W.push_back(7.99363836);
    W.push_back(9.19864774);
    W.push_back(11.16405769);
    W.push_back(15.39011994);

    // Array to store the integrand results
    std::vector<double> int1(X.size(), 0.0);
    
    // Loop over each x and calculate the integrand
    for (size_t k = 0; k < X.size(); ++k) {
        // Use AttariProb function (assuming it is already defined)
        int1[k] = W[k] * AttariProb(std::complex<double>(X[k], 0), kappa, v_bar, lambda_, rho, xi, sigma, T, K, theta, r, q, v);
    }

    // The call price using the Attari method
    double HestonC = theta * std::exp(-q * T) - K * std::exp(-r * T) * (0.5 + (1.0 / M_PI) * std::accumulate(int1.begin(), int1.end(), 0.0));

    // The put price by put-call parity
    double HestonP = HestonC - theta * std::exp(-q * T) + K * std::exp(-r * T);

    // Return the correct price based on PutCall
    if (PutCall == 'C') {
        return HestonC;
    } else {
        return HestonP;
    }
}

#endif