#ifndef HESTON_1993_H
#define HESTON_1993_H

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
 * Returns the integrand for the risk-neutral probabilities P1 and P2.
 * Heston model, Fourier transform.
 * 
 * @param phi    : integration variable (float)
 * @param kappa  : volatility mean reversion speed parameter (float)
 * @param v_bar  : volatility mean reversion level parameter (float)
 * @param lambda_: risk parameter (float)
 * @param rho    : correlation between two Brownian motions (float)
 * @param xi     : volatility of variance (float)
 * @param sigma  : volatility of individual shock (float)
 * @param T      : time to maturity (float)
 * @param K      : strike price (float)
 * @param theta  : aggregate shock (float)
 * @param r      : risk-free rate (float)
 * @param q      : dividend yield (float)
 * @param v      : variance (float)
 * @param Pnum   : 1 or 2 (for the probabilities) (int)
 * @param Trap   : 1 for Albrecher et al. (2007) ``Little Trap'' formulation (int)
 * 
 * @return the real part of the integrand for the Heston probability calculation
 */
std::complex<double> HestonProb(std::complex<double> phi, double kappa, double v_bar, double lambda_, double rho, double xi,
                                double sigma, double T, double K, double theta, double r, double q, double v, int Pnum) {

    // Log shock (logarithm of theta)
    double x = std::log(theta);

    // Intermediate variable 'a' in the equation
    double a = kappa * v_bar;

    // Setting parameters based on Pnum (1 or 2)
    double u, b;
    if (Pnum == 1) {
        u = 0.5;
        b = kappa + lambda_ - rho * xi;
    } else {
        u = -0.5;
        b = kappa + lambda_;
    }

    // Compute 'd' and 'g' (intermediate variables for the Fourier transform)
    std::complex<double> i(0, 1); // Imaginary unit
    std::complex<double> d = std::sqrt(std::pow(rho * xi * i * phi - b, 2) - std::pow(xi, 2) * (2.0 * u * i * phi - std::pow(phi, 2)));
    std::complex<double> g = (b - rho * xi * i * phi + d) / (b - rho * xi * i * phi - d);

    // "Little Trap" formulation
    std::complex<double> c = 1.0 / g;
    std::complex<double> G = (1.0 - c * std::exp(-d * T)) / (1.0 - c);
    std::complex<double> C = (r - q + u * std::pow(sigma, 2)) * i * phi * T - 0.5 * std::pow(phi, 2) * std::pow(sigma, 2) * T
            + a / std::pow(xi, 2) * ((b - rho * xi * i * phi - d) * T - 2.0 * std::log(G));
    std::complex<double> D = (b - rho * xi * i * phi - d) / std::pow(xi, 2) * ((1.0 - std::exp(-d * T)) / (1.0 - c * std::exp(-d * T)));

    // The characteristic function
    std::complex<double> f = std::exp(C + D * v + i * phi * x);

    // Return the real part of the integrand
    return std::exp(-i * phi * std::log(K)) * f / (i * phi);
}
/**
 * Heston (1993) price of a European option.
 * 
 * @param PutCall: 'C' for Call, 'P' for Put (char)
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
 * @param v      : initial variance (double)
 * @param Lphi   : lower integration limit (double)
 * @param Uphi   : upper integration limit (double)
 * @param dphi   : integration increment (double)
 * 
 * @return option price (double)
 */
double HestonPrice(char PutCall, double kappa, double v_bar, double lambda_, double rho, double xi,
                   double sigma, double T, double K, double theta, double r, double q, double v,
                   double Lphi, double Uphi, double dphi) {

    // Build the integration grid
    int N = static_cast<int>((Uphi - Lphi) / dphi) + 1;  // Number of phi points
    std::vector<std::complex<double> > phi(N); // The phi values array
    std::vector<double> int1(N, 0.0); // Integrand for P1
    std::vector<double> int2(N, 0.0); // Integrand for P2

    // Fill the phi grid
    for (int k = 0; k < N; ++k) {
        phi[k] = Lphi + k * dphi;
    }

    // Calculate the integrands for P1 and P2 using HestonProb
    for (int k = 0; k < N; ++k) {
        int1[k] = std::real(HestonProb(phi[k], kappa, v_bar, lambda_, rho, xi, sigma, T, K, theta, r, q, v, 1));
        int2[k] = std::real(HestonProb(phi[k], kappa, v_bar, lambda_, rho, xi, sigma, T, K, theta, r, q, v, 2));
    }

    // Numerical integration using the trapezoidal rule
    double I1 = 0.5 * (int1[0] + int1[N-1]);  // Start and end points
    double I2 = 0.5 * (int2[0] + int2[N-1]);  // Start and end points

    for (int k = 1; k < N - 1; ++k) {
        I1 += int1[k];
        I2 += int2[k];
    }

    I1 *= dphi;
    I2 *= dphi;

    // The probabilities P1 and P2
    double P1 = 0.5 + (1.0 / M_PI) * I1;
    double P2 = 0.5 + (1.0 / M_PI) * I2;

    // The call price
    double HestonC = theta * std::exp(-q * T) * P1 - K * std::exp(-r * T) * P2;

    // The put price by put-call parity
    double HestonP = HestonC - theta * std::exp(-q * T) + K * std::exp(-r * T);

    // Output the option price
    if (PutCall == 'C') {
        return HestonC; // Call price
    } else {
        return HestonP; // Put price
    }
}

#endif