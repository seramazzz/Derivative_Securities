#ifndef HESTON_SIMULATION_H
#define HESTON_SIMULATION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <random>
#include <fstream>
#include <tuple>
#include <algorithm>

std::vector<std::vector<double> > WX_global;
std::vector<std::vector<double> > WZ_global;
std::vector<std::vector<double> > Wv_global;
bool shocks_loaded = false;

// Function to generate standard normal random numbers
double standard_normal() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<> d(0.0, 1.0);
    return d(gen);
}

// Function to generate uniform random numbers in [0,1]
double uniform_random() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> d(0.0, 1.0);
    return d(gen);
}


// Function to calculate the inverse standard normal CDF
double normICDF(double p) {
    // Coefficients for p close to 0.5
    const double a[] = {
        3.3871328727963666080e0, 1.3314166789178437745e+2, 1.9715909503065514427e+3,
        1.3731693765509461125e+4, 4.5921953931549871457e+4, 6.7265770927008700853e+4,
        3.3430575583588128105e+4, 2.5090809287301226727e+3
    };
    const double b[] = {
        4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3,
        2.1213794301586595867e+4, 3.9307895800092710610e+4, 2.8729085735721942674e+4,
        5.2264952788528545610e+3
    };

    // Coefficients for p not close to 0, 0.5, or 1
    const double c[] = {
        1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
        3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
        2.27238449892691845833e-2, 7.74545014278341407640e-4
    };
    const double d[] = {
        2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
        1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
        1.05075007164441684324e-9
    };

    // Coefficients for p near 0 or 1
    const double e[] = {
        6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
        2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
        2.71155556874348757815e-5, 2.01033439929228813265e-7
    };
    const double f[] = {
        5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
        7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
        2.04426310338993978564e-15
    };

    double q = p - 0.5;
    double r, y;

    if (std::fabs(q) < 0.425) {
        // For p close to 0.5
        r = 0.180625 - q * q;
        y = q * (((((((a[7] * r + a[6]) * r + a[5]) * r + a[4]) * r + a[3]) * r + a[2]) * r + a[1]) * r + a[0]) /
            (((((((b[6] * r + b[5]) * r + b[4]) * r + b[3]) * r + b[2]) * r + b[1]) * r + 1.0));
    } else {
        // For p near 0 or 1
        r = (q < 0.0) ? p : (1.0 - p);
        if (r <= 0.0)
            return 0.0;
        
        r = sqrt(-log(r));

        if (r <= 5.0) {
            r -= 1.6;
            y = (((((((c[7] * r + c[6]) * r + c[5]) * r + c[4]) * r + c[3]) * r + c[2]) * r + c[1]) * r + c[0]) /
                (((((((d[6] * r + d[5]) * r + d[4]) * r + d[3]) * r + d[2]) * r + d[1]) * r + 1.0));
        } else {
            r -= 5.0;
            y = (((((((e[7] * r + e[6]) * r + e[5]) * r + e[4]) * r + e[3]) * r + e[2]) * r + e[1]) * r + e[0]) /
                (((((((f[6] * r + f[5]) * r + f[4]) * r + f[3]) * r + f[2]) * r + f[1]) * r + 1.0));
        }
        
        if (q < 0.0)
            y = -y;
    }

    return y;
}

/**
 * Generates aggregate shocks (correlated random variables) for the Heston simulation.
 *
 * @param NT : Number of time steps (int)
 * @param N  : Number of simulated paths (int)
 * @param Zv : Output matrix for volatility shocks (vector<vector<double>>)
 * @param Zx : Output matrix for price shocks (vector<vector<double>>)
 * @param Uv : Output matrix for uniform random shocks (vector<vector<double>>)
 */
void rnd_agg_shk(int NT, int N, std::vector<std::vector<double> >& Zv,
                                std::vector<std::vector<double> >& Zx,
                                std::vector<std::vector<double> >& Uv) {
    
    Zv.assign(NT, std::vector<double>(N, 0));
    Zx.assign(NT, std::vector<double>(N, 0));
    Uv.assign(NT, std::vector<double>(N, 0));

    for (int i = 0; i < N; i++) {
        for (int t = 1; t < NT; t++) {
            Zv[t][i] = standard_normal();
            Zx[t][i] = standard_normal();
            Uv[t][i] = uniform_random();
        }
    }

}

/**
 * Generates independent shocks for idiosyncratic components.
 *
 * @param NT : Number of time steps (int)
 * @param N  : Number of simulated paths (int)
 * @param Zz : Output matrix for independent shocks (vector<vector<double>>)
 */
void rnd_ind_shk(int NT, int N, std::vector<std::vector<double> >& Zz) {
    
    Zz.assign(NT, std::vector<double>(N, 0));

    for (int i = 0; i < N; i++) {
        for (int t = 1; t < NT; t++) {
            Zz[t][i] = standard_normal();
        }
    }

}

/**
 * Performs full Heston model simulation using Quadratic Exponential Scheme (QES).
 *
 * @param kappa   : Mean reversion speed (double)
 * @param v_bar   : Long-term variance mean (double)
 * @param xi      : Volatility of variance (double)
 * @param v       : Initial variance (double)
 * @param rho     : Correlation between Brownian motions (double)
 * @param sigma   : Volatility of idiosyncratic shocks (double)
 * @param gamma1  : Auxiliary QES parameter (double)
 * @param gamma2  : Auxiliary QES parameter (double)
 * @param theta   : Initial asset price (double)
 * @param T       : Maturity (double)
 * @param drift   : Drift rate (double)
 * @param q       : Dividend yield (double)
 * @param NT      : Number of time steps (int)
 * @param N       : Number of simulation paths (int)
 * @param MC      : Monte Carlo flag (1 = MC, 0 = deterministic) (int)
 * @param psi     : QES threshold parameter (double)
 * @param S       : Output matrix for simulated prices (vector<vector<double>>)
 * @param V       : Output matrix for simulated variances (vector<vector<double>>)
 * @param Zv      : Input aggregate variance shocks (vector<vector<double>>)
 * @param Zx      : Input aggregate price shocks (vector<vector<double>>)
 * @param Zz      : Input idiosyncratic shocks (vector<vector<double>>)
 * @param Uv      : Input uniform random shocks (vector<vector<double>>)
 */
void QESim(double kappa, double v_bar, double xi, double v, double rho, double sigma,
           double gamma1, double gamma2, double theta, double T, double drift, double q,
           int NT, int N, int MC, double psi,
           std::vector<std::vector<double> >& S, std::vector<std::vector<double> >& V,
           std::vector<std::vector<double> >& Zv, std::vector<std::vector<double> >& Zx,
           std::vector<std::vector<double> >& Zz, std::vector<std::vector<double> >& Uv) {

    double dt = T / NT;
    S.assign(NT, std::vector<double>(N, theta));
    V.assign(NT, std::vector<double>(N, v));

    std::vector<std::vector<double> > Zt;
    Zt.assign(NT, std::vector<double>(N, 0));

    double E = exp(-kappa * dt);
    double K0 = -kappa * rho * v_bar * dt / xi;
    double K1 = (kappa * rho / xi - 0.5) * gamma1 * dt - rho / xi;
    double K2 = (kappa * rho / xi - 0.5) * gamma2 * dt + rho / xi;
    double K3 = gamma1 * dt * (1 - rho * rho);
    double K4 = gamma2 * dt * (1 - rho * rho);
    double K5 = sigma * sigma * dt; // bonus for heterogeneity
    double A = K2 + K4 / 2.0;

    for (int i = 0; i < N; i++) {
        for (int t = 1; t < NT; t++) {
            Zt[t][i] = rho * Zv[t][i] + sqrt(1 - rho * rho) * Zx[t][i];

            double m = v_bar + (V[t - 1][i] - v_bar) * E;
            double s2 = (V[t - 1][i] * xi * xi * E / kappa * (1 - E) +
                         v_bar * xi * xi / (2 * kappa) * (1 - E) * (1 - E));
            double phi = s2 / (m * m);

            if (phi <= psi) {
                double b = sqrt(2 / phi - 1 + sqrt(2 / phi * (2 / phi - 1)));
                double a = m / (1 + b * b);
                double norm_inv = normICDF(Uv[t][i]);
                V[t][i] = a * (b + norm_inv) * (b + norm_inv);

                if (MC == 1 && A < (1 / (2 * a))) {
                    double M = exp(A * b * b * a / (1 - 2 * A * a)) / sqrt(1 - 2 * A * a);
                    K0 = -log(M) - (K1 + 0.5 * K3) * V[t][i];
                }

                S[t][i] = S[t - 1][i] * exp((drift - q) * dt + K0 + K1 * V[t - 1][i] + K2 * V[t][i] +
                                           sqrt(K3 * V[t - 1][i] + K4 * V[t][i]) * Zt[t][i] + sqrt(K5) * Zz[t][i]);
            } else {
                double p = (phi - 1) / (phi + 1);
                double beta = (1 - p) / m;
                double phiinv = (Uv[t][i] <= p) ? 0 : (1 / beta) * log((1 - p) / (1 - Uv[t][i]));

                V[t][i] = phiinv;

                if (MC == 1 && A < beta) {
                    double M = p + beta * (1 - p) / (beta - A);
                    K0 = -log(M) - (K1 + 0.5 * K3) * V[t][i];
                }

                S[t][i] = S[t - 1][i] * exp((drift - q) * dt + K0 + K1 * V[t - 1][i] + K2 * V[t][i] +
                                           sqrt(K3 * V[t - 1][i] + K4 * V[t][i]) * Zt[t][i] + sqrt(K5) * Zz[t][i]);
            }
        }
    }
}

/*
void read_agg_csv(const std::string& filename,
                  std::vector<std::vector<double> >& Zv,
                  std::vector<std::vector<double> >& Zx,
                  std::vector<std::vector<double> >& Uv) {
    std::ifstream file(filename);
    std::string line;
    bool header_skipped = false;

    while (std::getline(file, line)) {
        if (!header_skipped) {
            header_skipped = true;
            continue;
        }

        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (std::getline(ss, value, ',')) {
            if (!value.empty()) {
                row.push_back(std::stod(value));
            }
        }

        if (row.size() >= 3) {
            Zv.push_back(std::vector<double>(1, row[0]));
            Zx.push_back(std::vector<double>(1, row[1]));
            Uv.push_back(std::vector<double>(1, row[2]));
        }
    }

    file.close();
}
*/
/*
void read_ind_csv(const std::string& filename,
                  std::vector<std::vector<double> >& Zz) {
    std::ifstream file(filename);
    std::string line;
    bool header_skipped = false;

    while (std::getline(file, line)) {
        if (!header_skipped) {
            header_skipped = true;
            continue;
        }

        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            if (!value.empty()) {
                Zz.push_back(std::vector<double>(1, std::stod(value)));
            }
        }
    }

    file.close();
}
*/
/**
 * Saves aggregate shocks (Zv, Zx, Uv) to a CSV file.
 *
 * @param Zv : Matrix of volatility shocks (vector<vector<double>>)
 * @param Zx : Matrix of price shocks (vector<vector<double>>)
 * @param Uv : Matrix of uniform random shocks (vector<vector<double>>)
 */
void save_agg_csv(const std::vector<std::vector<double> >& Zv,
                  const std::vector<std::vector<double> >& Zx,
                  const std::vector<std::vector<double> >& Uv) {
    std::ofstream file("agg_shocks_plus.csv");
    file << "Zv,Zx,Uv\n";
    for (size_t t = 0; t < Zv.size(); ++t) {
        for (size_t i = 0; i < Zv[t].size(); ++i) {
            file <<  Zv[t][i] << "," << Zx[t][i] <<  "," << Uv[t][i];
            file << "\n";
        }
    }
    file.close();
    std::cout << "Simulation results saved to agg_shocks.csv\n";
}

/**
 * Saves independent shocks (Zz) to a CSV file.
 *
 * @param Zz : Matrix of independent shocks (vector<vector<double>>)
 */
void save_ind_csv(const std::vector<std::vector<double> >& Zz) {
    std::ofstream file("ind_shocks_plus.csv");
    file << "Zz\n";
    for (size_t t = 0; t < Zz.size(); ++t) {
        for (size_t i = 0; i < Zz[t].size(); ++i) {
            file <<  Zz[t][i];
            file << "\n";
        }
    }
    file.close();
    std::cout << "Simulation results saved to ind_shocks.csv\n";
}

/**
 * Writes simulated paths to file and plots asset and variance trajectories using gnuplot.
 *
 * @param S       : Matrix of simulated prices (vector<vector<double>>)
 * @param V       : Matrix of simulated variances (vector<vector<double>>)
 * @param y1label : Label for the primary y-axis (string)
 * @param y2label : Label for the secondary y-axis (string)
 */

void plot_results(const std::vector<std::vector<double> >& S, 
                  const std::vector<std::vector<double> >& V, 
                  const std::string& y1label, 
                  const std::string& y2label) {
    std::ofstream plot_file("data.txt");
    for (size_t t = 0; t < S.size(); ++t) {
        plot_file << t << " " << S[t][0] << " " << V[t][0] << std::endl;
    }
    plot_file.close();

    std::stringstream gnuplot_cmd;
    gnuplot_cmd << "gnuplot -p -e \"set title 'Simulated Data'; "
                << "set xlabel 'Time Steps'; "
                << "set ylabel '" << y1label << "'; "
                << "set y2label '" << y2label << "'; "
                << "set ytics nomirror; set y2tics; "
                << "set autoscale xfixmin; set autoscale xfixmax; "
                << "plot 'data.txt' using 1:2 with lines lw 2 lc rgb '#660099' title '" << y1label << "' axis x1y1, "
                << "'data.txt' using 1:3 with lines lw 2 lc rgb '#999999' title '" << y2label << "' axis x1y2\"";

    // Execute Gnuplot command
    system(gnuplot_cmd.str().c_str());
}
// Function to read "representative path.csv" and extract Stock_Price (S) and Variance (V)
void read_csv(const std::string& filename, std::vector<double>& S, std::vector<double>& V) {
    std::ifstream file(filename);
    std::string line;
    bool header_skipped = false;

    while (std::getline(file, line)) {
        if (!header_skipped) {
            header_skipped = true;  // Skip header line
            continue;
        }

        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        // Extract Stock_Price (column 0) and Variance (column 1)
        S.push_back(row[0]);
        V.push_back(row[1]);
    }

    file.close();
}

// Function to save the calculated values to CSV
/*
void save_SV_to_csv(const std::vector<std::vector<double> >& S, 
                    const std::vector<std::vector<double> >& V) {
    std::ofstream file("simulation.csv");
    file << "S,V\n";
    for (size_t t = 0; t < S.size(); ++t) {
        for (size_t i = 0; i < S[t].size(); ++i) {
            file <<  S[t][i] << "," << V[t][i];
            file << "\n";
        }
    }
    file.close();
    std::cout << "Simulation results saved to simulation.csv\n";
}
*/
/*
void save_csv(const std::string& filename, const std::vector<std::vector<double> >& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }

    size_t rows = data.size();
    size_t cols = data[0].size();

    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            file << data[i][j];
            if (i != rows - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}
*/
/*
void generate_random_paths(int N, double T, int NT, double rho) {

    std::vector<std::vector<double> > WX(N, std::vector<double>(NT));
    std::vector<std::vector<double> > WZ(N, std::vector<double>(NT));
    std::vector<std::vector<double> > Wv(N, std::vector<double>(NT));

    for (int i = 0; i < N / 2; ++i) {
        std::vector<double> Z1(NT), Z2(NT), Z3(NT);
        for (int j = 0; j < NT; ++j) {
            Z1[j] = standard_normal();
            Z2[j] = standard_normal();
            Z3[j] = standard_normal();
        
            double Wv_val = Z1[j];
            double WX_val = rho * Z1[j] + std::sqrt(1 - rho * rho) * Z2[j];
            double WZ_val = Z3[j];

            WX[i][j] = WX_val;
            Wv[i][j] = Wv_val;
            WZ[i][j] = WZ_val;
            // Antithetic pair
            WX[i + N / 2][j] = -WX_val;
            Wv[i + N / 2][j] = -Wv_val;
            WZ[i + N / 2][j] = -WZ_val;
        }
    }

    save_csv("WX.csv", WX);
    save_csv("WZ.csv", WZ);
    save_csv("Wv.csv", Wv);
}
*/
/*
std::vector<std::vector<double> > load_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double> > data;

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }

        data.push_back(row);
    }

    // Transpose: make rows = paths, columns = time steps
    size_t rows = data[0].size();
    size_t cols = data.size();
    std::vector<std::vector<double> > result(rows, std::vector<double>(cols));

    for (size_t i = 0; i < cols; ++i)
        for (size_t j = 0; j < rows; ++j)
            result[j][i] = data[i][j];

    return result;
}
*/
/*
void option_pricing(
    int time_point,
    double r, double q, double sigma,
    double kappa, double vbar, double xi,
    double theta0, double v0, int capacity0,
    double T, int NT, std::vector<double> i0, std::vector<double> i1, std::vector<double> d0, std::vector<double> d1,
    std::vector<double>& thetas, std::vector<double>& upsilons, std::vector<double>& capacities
) {
    double dt = T / NT;

    if (!shocks_loaded) {
        WX_global = load_csv("WX.csv");
        WZ_global = load_csv("WZ.csv");
        Wv_global = load_csv("Wv.csv");
        shocks_loaded = true;
    }

    const std::vector<std::vector<double> >& WX = WX_global;
    const std::vector<std::vector<double> >& WZ = WZ_global;
    const std::vector<std::vector<double> >& Wv = Wv_global;

    size_t N = WX.size();
    thetas.resize(N);
    upsilons.resize(N);
    capacities.resize(N);

    for (size_t i = 0; i < N; ++i) {
        double theta = theta0;
        double v = v0;
        int capacity = capacity0;

        for (int t = 0; t <= time_point; ++t) {
            // Milstein scheme for v
            v += kappa * (vbar - v) * dt
               + xi * std::sqrt(v) * Wv[i][t] * std::sqrt(dt)
               + 0.25 * xi * xi * dt * (Wv[i][t] * Wv[i][t] - 1);
            v = std::abs(v);  // reflection

            // Milstein scheme for theta
            theta += (r - q) * theta * dt
                   + std::sqrt(v) * theta * WX[i][t] * std::sqrt(dt)
                   + sigma * theta * WZ[i][t] * std::sqrt(dt)
                   + 0.5 * v * theta * dt * (WX[i][t] * WX[i][t] - 1)
                   + 0.5 * sigma * sigma * theta * dt * (WZ[i][t] * WZ[i][t] - 1);

            while (capacity > 0 && theta < std::exp(d0[capacity] + v * d1[capacity])) {
                capacity-=1;
            }

            while (theta > std::exp(i0[capacity+1] + v * i1[capacity+1])) {
                capacity+=1;
            }
        }

        thetas[i] = theta;
        upsilons[i] = v;
        capacities[i] = capacity;
    }
}
*/
#endif