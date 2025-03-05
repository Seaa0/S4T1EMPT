#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip> // Include for precision control


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
            if (values.size() > 5) { // Ensure the required columns exist
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

    
    long double m = 0.0, b = 0.0;
    double learningRate = 0.00025;
    int i = 0;
    long double prev = 0.0;
    
    while (true) {
        i++;
        vector<long double> SE, dmsedm, dmsedb;
        for (size_t j = 0; j < X.size(); j++) {
            long double xi = X[j], yi = y[j];
            long double yj = m * xi + b;
            
            SE.push_back(pow(yi - yj, 2));
            dmsedm.push_back((yj - yi) * xi);
            dmsedb.push_back(yj - yi);
        }
        
        long double MSE = accumulate(SE.begin(), SE.end(), 0.0) / SE.size();
        long double Mdmsedm = accumulate(dmsedm.begin(), dmsedm.end(), 0.0) / dmsedm.size() * 2;
        long double Mdmsedb = accumulate(dmsedb.begin(), dmsedb.end(), 0.0) / dmsedb.size() * 2;
        
        m -= learningRate * Mdmsedm;
        b -= learningRate * Mdmsedb;
        
        if (i % 10000 == 0) {
            cout << "Epoch " << i << ": MSE = " << MSE << endl;
        }
        if (prev == MSE) {
            cout << "Epoch " << i << ": MSE = " << MSE << endl;
            cout << "Ending loop..." << endl;
            break;
        }
        prev = MSE;
    }
    return 0;
}