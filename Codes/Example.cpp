#include <iostream>
#include <stdexcept> // For exception handling
#include "Lewis_Vol.h"
#include "Heston_1993.h"
#include <vector>
#include "Attari_2004.h"
//#include "Assets_in_Place.h"
#include "Americans.h"
//#include "Growth_options.h"
#include <chrono> // For timing
//#include <nlopt.hpp>
#include <tuple>
#include "Heston_simulation.h"

// Simulation
/*
int main() {
    // Heston Model Parameters
    double kappa = 5, v_bar = 0.05, xi = 0.5, v = 0.05, rho = -0.7, sigma = 0.25;
    double theta = 45, T = 10, drift = 0.12, q = 0.02;

    std::vector<std::vector<double> > S, V, Zv, Zx, Zz, Uv;
    rnd_agg_shk(T*252, 1, Zv, Zx, Uv);
    rnd_ind_shk(T*252, 1, Zz);
    QESim(kappa, v_bar, xi, v, rho, sigma, 0.5, 0.5, theta, T, drift, q, T*252, 1, 0, 1.5, S, V, Zv, Zx, Zz, Uv);

    //
    save_agg_csv(Zv, Zx, Uv);
    save_ind_csv(Zz);
    
    plot_results(S, V, "Aggregate Schock", "Variance");

    return 0;
}
*/

int main() {
    // Example usage of the function
    char PutCall = 'C'; // Call option
    double theta = 20;
    double K = 19;
    double T = 300;
    double r = 0.03;
    double q = 0.02;
    double kappa = 5;
    double v_bar = 0.05;
    double xi = 0.5;
    double sigma = 0.25;
    double lambda_ = -2;
    double v = 0.05;
    double rho = -0.7;
    double f = 10;
    double capacity = 1;
    double gamma = 0.6;
    double I = 13;
    double I_p = 0.5;
    double J = 7;
    double J_p = 0.5;
    double drift = 0.12;
    std::vector<double> x_GLa, w_GLa; // Gauss Laguerre quadrature abscissas and weights
    static const double x_GLa_array[] = {
        4.44893658e-02, 2.34526110e-01, 5.76884629e-01, 1.07244875e+00,
        1.72240878e+00, 2.52833671e+00, 3.49221328e+00, 4.61645675e+00,
        5.90395872e+00, 7.35812527e+00, 8.98294783e+00, 1.07829985e+01,
        1.27637110e+01, 1.49313497e+01, 1.72911574e+01, 1.98603347e+01,
        2.26205942e+01, 2.56439448e+01, 2.88539181e+01, 3.23205798e+01,
        3.61939939e+01, 3.99836016e+01, 4.47111660e+01, 4.90423478e+01,
        5.44478923e+01, 5.98411301e+01, 6.59877955e+01, 7.26879843e+01,
        8.01860031e+01, 8.87358267e+01, 9.88294726e+01, 1.11751402e+02
    }; // Abscissas, found in Python
    static const double w_GLa_array[] = {
        0.11418711, 0.26606522, 0.41879314, 0.57253285, 0.72764879,
        0.88453672, 1.04361888, 1.20534924, 1.37022139, 1.53878017,
        1.71161155, 1.88945863, 2.07281121, 2.26320788, 2.46444164,
        2.66201387, 2.87828775, 3.03299202, 3.38961163, 3.48359725,
        3.43546123, 4.55635927, 3.8469007 , 5.63079911, 4.63990579,
        6.39259714, 6.38441924, 7.03996479, 7.99363836, 9.19864774,
        11.16405769, 15.39011994
    }; // Integration weights, found in Python
    x_GLa.assign(x_GLa_array, x_GLa_array + sizeof(x_GLa_array) / sizeof(x_GLa_array[0]));
    w_GLa.assign(w_GLa_array, w_GLa_array + sizeof(w_GLa_array) / sizeof(w_GLa_array[0]));
    std::vector<double> x_GLe, w_GLe; // Gauss Legendre quadrature abscissas and weights
    static const double x_GLe_array[] = {
        -0.99726389, -0.98561141, -0.96476241, -0.93490591, -0.89632128,
        -0.84936754, -0.79448383, -0.73218210, -0.66304427, -0.58771576,
        -0.50689991, -0.42135128, -0.33186860, -0.23928736, -0.14447196,
        -0.04830767,  0.04830767,  0.14447196,  0.23928736,  0.33186860,
         0.42135128,  0.50689991,  0.58771576,  0.66304427,  0.73218211,
         0.79448383,  0.84936754,  0.89632128,  0.93490591,  0.96476242,
         0.98561141,  0.99726389
    }; // Abscissas, found in Python
    static const double w_GLe_array[] = {
        0.00701854, 0.01627452, 0.02539212, 0.03427394, 0.0428359 ,
        0.05099801, 0.05868412, 0.06582221, 0.0723458 , 0.0781939 ,
        0.08331192, 0.08765209, 0.09117388, 0.0938444 , 0.09563872,
        0.09654009, 0.09654009, 0.09563872, 0.0938444 , 0.09117388,
        0.08765209, 0.08331192, 0.0781939 , 0.0723458 , 0.06582221,
        0.05868413, 0.05099802, 0.04283605, 0.03427382, 0.02539216,
        0.01627456, 0.00701852
    }; // Integration weights, found in Python
    x_GLe.assign(x_GLe_array, x_GLe_array + sizeof(x_GLe_array) / sizeof(x_GLe_array[0]));
    w_GLe.assign(w_GLe_array, w_GLe_array + sizeof(w_GLe_array) / sizeof(w_GLe_array[0]));
    
    /*
    // Standard Heston
    auto startHeston = std::chrono::high_resolution_clock::now();
    double optionPriceHest = HestonPrice(PutCall, kappa, v_bar, lambda_, rho, xi, sigma, T, K, theta, r, q, v, 0.000001, 100, 0.01);
    auto endHeston = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationHeston = endHeston - startHeston;
    std::cout << "Option Price Heston (1993): " << optionPriceHest << std::endl;
    std::cout << "Execution Time Heston (1993): " << durationHeston.count() << " seconds" << std::endl;

    // AttariPriceGaussLaguerre
    auto startAttari = std::chrono::high_resolution_clock::now();
    double optionPriceAtt = AttariPriceGaussLaguerre(PutCall, theta, K, T, r, q, kappa, v_bar, xi, sigma, lambda_, v, rho);
    auto endAttari = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationAttari = endAttari - startAttari;
    std::cout << "Option Price Attari (2004): " << optionPriceAtt << std::endl;
    std::cout << "Execution Time Attari (2004): " << durationAttari.count() << " seconds" << std::endl;

    // LewisVolOfVol
    auto startLewis = std::chrono::high_resolution_clock::now();
    double optionPriceLewis = LewisVolOfVol(PutCall, theta, K, r, q, T, v, rho, v_bar, kappa, xi, sigma, lambda_);
    auto endLewis = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationLewis = endLewis - startLewis;
    std::cout << "Option Price Lewis (2000): " << optionPriceLewis << std::endl;
    std::cout << "Execution Time Lewis (2000): " << durationLewis.count() << " seconds" << std::endl;
    */
    // Incremental assets-in-place
    /*
    auto start_dAiP = std::chrono::high_resolution_clock::now();
    std::vector<double> dAiP = incr_AiP(theta, K, r, q, v, rho, v_bar, kappa, xi, sigma, lambda_);
    double sum_dAiP = std::accumulate(dAiP.begin(), dAiP.end(), 0.0);
    auto end_dAiP = std::chrono::high_resolution_clock::now();
    std::cout << "dAiP: " << sum_dAiP << std::endl;
    std::chrono::duration<double> duration_dAiP = end_dAiP - start_dAiP;
    std::cout << "Execution Time for production unit: " << duration_dAiP.count() << " seconds" << std::endl;
    */
    // Total Assets-in-Place
    /*
    auto start = std::chrono::high_resolution_clock::now();
    std::pair<std::vector<double>, std::vector<double> > result = AiP(20, capacity, r, q, 0.12, v, rho, v_bar, kappa, xi, sigma, lambda_, gamma, f, 1);
    std::vector<double> AiP = result.first;
    std::vector<double> cap_ut = result.second;
    double sum_AiP = std::accumulate(AiP.begin(), AiP.end(), 0.0); // AiP value
    double mean_capacity = std::accumulate(cap_ut.begin(), cap_ut.end(), 0.0) / cap_ut.size(); // Capacity Utilisation
    std::cout << "Sum of AiP: " << sum_AiP << std::endl;
    std::cout << "Mean of Capacity Utilization: " << mean_capacity << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution Time: " << duration.count() << " seconds" << std::endl;
    */
    /*
    // Chiarella, Ziogas, Ziveyi (2010) characteristic function
    std::complex<double> phi(1.0, 1.0);
    std::complex<double> psi(0.5, -0.5);
    std::complex<double> CZZ1 = CZCharFun(theta, T, 0.25, kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q, phi, psi, 1);
    std::complex<double> CZZ2 = CZCharFun(theta, T, 0.25, kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q, phi, psi, 2);
    std::cout << "Chiarella, Ziogas, Ziveyi (2010) characteristic function 1: " << CZZ1 << std::endl;
    std::cout << "Chiarella, Ziogas, Ziveyi (2010) characteristic function 2: " << CZZ2 << std::endl;
    */
    /*
    // Chiarella, Ziogas, Ziveyi (2010) options
    // Call
    auto startCZZ_C = std::chrono::high_resolution_clock::now();
    double optionPriceCZZ_C = CZEuroOption("C", theta, T, kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q, x_GLa, w_GLa);
    auto endCZZ_C = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationCZZ_C = endCZZ_C - startCZZ_C;
    std::cout << "Call Option Price Chiarella, Ziogas, Ziveyi (2010): " << optionPriceCZZ_C << std::endl;
    std::cout << "Execution Time Chiarella, Ziogas, Ziveyi (2010): " << durationCZZ_C.count() << " seconds" << std::endl;
    // Put
    auto startCZZ_P = std::chrono::high_resolution_clock::now();
    double optionPriceCZZ_P = CZEuroOption("P", theta, T, kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q, x_GLa, w_GLa);
    auto endCZZ_P = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationCZZ_P = endCZZ_P - startCZZ_P;
    std::cout << "Put Option Price Chiarella, Ziogas, Ziveyi (2010): " << optionPriceCZZ_P << std::endl;
    std::cout << "Execution Time Chiarella, Ziogas, Ziveyi (2010): " << durationCZZ_P.count() << " seconds" << std::endl;
    */
    /*
    // Gauss Legendre
    double result1 = DoubleGaussLegendre(theta, T, K, r, q, 5.0562, 2.3226, x_GLe, w_GLe, x_GLe, w_GLe, 0.00001, T, 0.00001, theta, 1, kappa, v_bar, xi, sigma, v, rho, lambda_);
    std::cout << "Double Gauss Legendre: " << result1 << std::endl;

    // Double trapezoidal
    std::vector<double> T_vec, X_vec; // Gauss Legendre quadrature abscissas and weights
    static const double X_vec_array[] = {
        0.00001, 20, 40, 60, 80, 100
    }; // Abscissas, found in Python
    static const double T_vec_array[] = {
        0.00001, 0.1, 0.2, 0.3, 0.4, 0.5
    }; // Integration weights, found in Python
    T_vec.assign(T_vec_array, T_vec_array + sizeof(T_vec_array) / sizeof(T_vec_array[0]));
    X_vec.assign(X_vec_array, X_vec_array + sizeof(X_vec_array) / sizeof(X_vec_array[0]));
    double result2 = DoubleTrapezoidal(theta, K, T, r, q, 5.0562, 2.3226, X_vec, T_vec, 2, kappa, v_bar, xi, sigma, v, rho, lambda_);
    std::cout << "Double Trapezoidal: " << result2 << std::endl;
    
    std::cout << "X grid: ";
    for (auto val : X_vec) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "T grid: ";
    for (auto val : T_vec) std::cout << val << " ";
    std::cout << std::endl; 
    */

    /*
    // Initial Guesses for Early Exercise Boundary
    double Evt = EV(v, v_bar, kappa, T, 0.0);
    double V00 = Evt + xi / std::abs(kappa) * std::sqrt(kappa * v_bar / 2.0);
    double V10 = Evt - xi / std::abs(kappa) * std::sqrt(kappa * v_bar / 2.0);
    double b00 = std::max(std::log(r * K / q), std::log(K));
    double b10 = 0.0;
    // Initial guess for b0 and b1
    //std::vector<double> v_values = {V00, V10};
    //std::vector<double> b = {b00, b10};

    
    // Parameters for American call's Early Exercise Boundary
    std::pair<std::vector<double>, std::vector<double> > b_coefs_call = findB("C", T, kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q, V00, V10, b00, b10, x_GLa, w_GLa, x_GLe, w_GLe, 150, 25, 0.005, 0.005, 1e-08, 1e-10, T, 1e-10, theta,"GLe");
    std::vector<double> b0_vec_call = b_coefs_call.first;
    std::vector<double> b1_vec_call = b_coefs_call.second;
    double b0_call = b0_vec_call.back();
    double b1_call = b1_vec_call.back();
    std::cout << "Early Exercise Boundary: " << std::exp(b0_call + b1_call * v) << std::endl;
    

    // Early Exercise Premium for American Call as in Chiarella, Ziogas, Ziveyi (2010)
    auto startEEP_DT_call = std::chrono::high_resolution_clock::now();
    double EEP_DT_call = CZEarlyExercise("C",theta,T, K, r, q, x_GLa, w_GLa, x_GLe, w_GLe, 150, b0_call, b1_call, 1e-10, T, 1e-10, theta, "Trapz", kappa, v_bar, xi, sigma, v, rho, lambda_);
    auto endEEP_DT_call = std::chrono::high_resolution_clock::now();
    auto startEEP_GLe_call = std::chrono::high_resolution_clock::now();
    double EEP_GLe_call = CZEarlyExercise("C",theta,T, K, r, q, x_GLa, w_GLa, x_GLe, w_GLe, 150, b0_call, b1_call, 1e-10, T, 1e-10, theta, "GLe", kappa, v_bar, xi, sigma, v, rho, lambda_);
    auto endEEP_GLe_call = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationEEP_DT_call = endEEP_DT_call - startEEP_DT_call;
    std::cout << "Early Exercise Premium (Call) Chiarella, Ziogas, Ziveyi (2010), Double Trapezoidal: " << EEP_DT_call << std::endl;
    std::cout << "Execution Time Early Exercise Premium (Call) Chiarella, Ziogas, Ziveyi (2010), Double Trapezoidal: " << durationEEP_DT_call.count() << " seconds" << std::endl;
    std::chrono::duration<double> durationEEP_GLe_call = endEEP_GLe_call - startEEP_GLe_call;
    std::cout << "Early Exercise Premium (Call) Chiarella, Ziogas, Ziveyi (2010), Double Gauss Legendre: " << EEP_GLe_call << std::endl;
    std::cout << "Execution Time Early Exercise Premium (Call) Chiarella, Ziogas, Ziveyi (2010), Double Gauss Legendre: " << durationEEP_GLe_call.count() << " seconds" << std::endl;

    // American Call price as in Chiarella, Ziogas, Ziveyi (2010)
    auto start_amer_CZZ_GLe_call = std::chrono::high_resolution_clock::now();
    double Option_CZZ_GLe_call = CZAmerOption("C", theta, T, K, r, q, x_GLa, w_GLa, x_GLe, w_GLe, 150, b0_call, b1_call, 1e-10, T, 1e-10, theta, "GLe", kappa, v_bar, xi, sigma, v, rho, lambda_);
    auto end_amer_CZZ_GLe_call = std::chrono::high_resolution_clock::now();
    auto start_amer_CZZ_Trapz_call = std::chrono::high_resolution_clock::now();
    double Option_CZZ_Trapz_call = CZAmerOption("C", theta, T, K, r, q, x_GLa, w_GLa, x_GLe, w_GLe, 150, b0_call, b1_call, 1e-10, T, 1e-10, theta, "Trapz", kappa, v_bar, xi, sigma, v, rho, lambda_);
    auto end_amer_CZZ_Trapz_call = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationAmer_CZZ_GLe_call = end_amer_CZZ_GLe_call - start_amer_CZZ_GLe_call;
    std::cout << "American Call Chiarella, Ziogas, Ziveyi (2010), Gauss Legendre: " << Option_CZZ_GLe_call << std::endl;
    std::cout << "Execution Time American Chiarella, Ziogas, Ziveyi (2010), Gauss Legendre: " << durationAmer_CZZ_GLe_call.count() << " seconds" << std::endl;
    std::chrono::duration<double> durationAmer_CZZ_Trapz_call = end_amer_CZZ_Trapz_call - start_amer_CZZ_Trapz_call;
    std::cout << "American Call Chiarella, Ziogas, Ziveyi (2010), Double Trapezoidal: " << Option_CZZ_Trapz_call << std::endl;
    std::cout << "Execution Time American Call Chiarella, Ziogas, Ziveyi (2010), Double Trapezoidal: " << durationAmer_CZZ_Trapz_call.count() << " seconds" << std::endl;
    
    
    // Parameters for American put's Early Exercise Boundary
    std::pair<std::vector<double>, std::vector<double> > b_coefs_put = findB("P", T, kappa, v_bar, xi, sigma, v, rho, lambda_, K, r, q, V00, V10, b00, b10, x_GLa, w_GLa, x_GLe, w_GLe, 150, 25, 0.005, 0.005, 1e-08, 1e-10, T, 1e-10, theta,"GLe");
    std::vector<double> b0_vec_put = b_coefs_put.first;
    std::vector<double> b1_vec_put = b_coefs_put.second;
    double b0_put = b0_vec_put.back();
    double b1_put = b1_vec_put.back();
    std::cout << "Early Exercise Boundary: " << std::exp(b0_put + b1_put * v) << std::endl;

    // Early Exercise Premium for American Put as in Chiarella, Ziogas, Ziveyi (2010)
    auto startEEP_DT_put = std::chrono::high_resolution_clock::now();
    double EEP_DT_put = CZEarlyExercise("P",theta,T, K, r, q, x_GLa, w_GLa, x_GLe, w_GLe, 150, b0_put, b1_put, 1e-10, T, 1e-10, theta, "Trapz", kappa, v_bar, xi, sigma, v, rho, lambda_);
    auto endEEP_DT_put = std::chrono::high_resolution_clock::now();
    auto startEEP_GLe_put = std::chrono::high_resolution_clock::now();
    double EEP_GLe_put = CZEarlyExercise("P",theta,T, K, r, q, x_GLa, w_GLa, x_GLe, w_GLe, 150, b0_put, b1_put, 1e-10, T, 1e-10, theta, "GLe", kappa, v_bar, xi, sigma, v, rho, lambda_);
    auto endEEP_GLe_put = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationEEP_DT_put = endEEP_DT_put - startEEP_DT_put;
    std::cout << "Early Exercise Premium (Put) Chiarella, Ziogas, Ziveyi (2010), Double Trapezoidal: " << EEP_DT_put << std::endl;
    std::cout << "Execution Time Early Exercise Premium (Put) Chiarella, Ziogas, Ziveyi (2010), Double Trapezoidal: " << durationEEP_DT_put.count() << " seconds" << std::endl;
    std::chrono::duration<double> durationEEP_GLe_put = endEEP_GLe_put - startEEP_GLe_put;
    std::cout << "Early Exercise Premium (Put) Chiarella, Ziogas, Ziveyi (2010), Double Gauss Legendre: " << EEP_GLe_put << std::endl;
    std::cout << "Execution Time Early Exercise Premium (put) Chiarella, Ziogas, Ziveyi (2010), Double Gauss Legendre: " << durationEEP_GLe_put.count() << " seconds" << std::endl;

    // American Put price as in Chiarella, Ziogas, Ziveyi (2010)
    auto start_amer_CZZ_GLe_put = std::chrono::high_resolution_clock::now();
    double Option_CZZ_GLe_put = CZAmerOption("P", theta, T, K, r, q, x_GLa, w_GLa, x_GLe, w_GLe, 150, b0_put, b1_put, 1e-10, T, 1e-10, theta, "GLe", kappa, v_bar, xi, sigma, v, rho, lambda_);
    auto end_amer_CZZ_GLe_put = std::chrono::high_resolution_clock::now();
    auto start_amer_CZZ_Trapz_put = std::chrono::high_resolution_clock::now();
    double Option_CZZ_Trapz_put = CZAmerOption("P", theta, T, K, r, q, x_GLa, w_GLa, x_GLe, w_GLe, 150, b0_put, b1_put, 1e-10, T, 1e-10, theta, "Trapz", kappa, v_bar, xi, sigma, v, rho, lambda_);
    auto end_amer_CZZ_Trapz_put = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationAmer_CZZ_GLe_put = end_amer_CZZ_GLe_put - start_amer_CZZ_GLe_put;
    std::cout << "American Put Chiarella, Ziogas, Ziveyi (2010), Gauss Legendre: " << Option_CZZ_GLe_put << std::endl;
    std::cout << "Execution Time American Put Chiarella, Ziogas, Ziveyi (2010), Gauss Legendre: " << durationAmer_CZZ_GLe_put.count() << " seconds" << std::endl;
    std::chrono::duration<double> durationAmer_CZZ_Trapz_put = end_amer_CZZ_Trapz_put - start_amer_CZZ_Trapz_put;
    std::cout << "American Put Chiarella, Ziogas, Ziveyi (2010), Double Trapezoidal: " << Option_CZZ_Trapz_put << std::endl;
    std::cout << "Execution Time American Put Chiarella, Ziogas, Ziveyi (2010), Double Trapezoidal: " << durationAmer_CZZ_Trapz_put.count() << " seconds" << std::endl;
    
    */
    return 0;
}