#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip> // For precision control

using namespace std;

// Function to read CSV file
tuple<vector<long double>, vector<long double>> readCSV(const string &filename) {
    vector<long double> X, y;
    ifstream file(filename);
    string line, val;
    
    getline(file, line); // Read header to skip it
    
    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> values;
        while (getline(ss, val, ',')) {
            values.push_back(val);
        }
        if (values.size() > 5) { // Ensure required columns exist
            try {
                X.push_back(stod(values[4])); // Physical activity level
                y.push_back(stod(values[2])); // Sleep duration
            } catch (const exception &e) {
                cerr << "Error parsing row: " << line << endl;
            }
        }
    }
    return {X, y};
}

int main() {
    cout << setprecision(100);
    string filename = "../../datasets/preprocessedDataset.csv";
    auto [X, y] = readCSV(filename);

    vector<long double> weights(3, 0.0);
    long double learningRate = 0.00000002;
    int i = 0;
    long double prev = 0;

    while (true) {
        i++;
        vector<long double> SE;
        vector<long double> djdwj(3, 0.0);

        for (size_t k = 0; k < X.size(); k++) {
            long double xi = X[k];
            long double yi = y[k];

            long double yj = weights[2] * pow(xi, 2) + weights[1] * xi + weights[0];
            SE.push_back(pow(yi - yj, 2));

            for (int j = 0; j < 3; j++) {
                djdwj[j] += (yj - yi) * pow(xi, j);
            }
        }

        long double MSE = accumulate(SE.begin(), SE.end(), 0.0) / SE.size();
        for (int j = 0; j < 3; j++) {
            djdwj[j] = (djdwj[j] / X.size()) * 2;
            weights[j] -= djdwj[j] * learningRate;
        }

        if (i % 10000 == 0) {
            cout << "Epoch " << i << ": MSE = " << MSE << endl;
        }
        if (prev == MSE) {
            cout << "Epoch " << i << ": MSE = " << MSE << endl;
            for (double w : weights) {
                cout << w << " ";
            }
            cout << endl;
            cout << "Ending loop..." << endl;
            break;
        }
        prev = MSE;
    }

    return 0;
}
