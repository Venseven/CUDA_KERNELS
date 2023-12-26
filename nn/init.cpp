#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

class NeuralNetwork {
private:
    std::vector<float> weights_layer_1;
    std::vector<float> weights_layer_2;
    std::vector<float> weights_layer_3;
    int input_size;
    int hidden_size;
    int output_size;

public:
    NeuralNetwork(int input, int hidden, int output) 
        : input_size(input), hidden_size(hidden), output_size(output) {
        // Initialize weights for each layer
        weights_layer_1.resize(input * hidden);
        weights_layer_2.resize(hidden * hidden);
        weights_layer_3.resize(hidden * output);
        // Initialize weights (randomly or otherwise)
    }

    void relu(std::vector<float>& input) {
        #pragma omp parallel for
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::max(0.0f, input[i]);
        }
    }

    void linear(const std::vector<float>& input, const std::vector<float>& weights, 
                std::vector<float>& output, int input_dim, int output_dim) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < output_dim; ++i) {
            for (int j = 0; j < input_dim; ++j) {
                #pragma omp atomic
                output[i] += input[j] * weights[i * input_dim + j];
            }
        }
    }

    void forward(const std::vector<float>& input, std::vector<float>& output) {
        std::vector<float> hidden_1(hidden_size);
        std::vector<float> hidden_2(hidden_size);
        
        // First layer
        linear(input, weights_layer_1, hidden_1, input_size, hidden_size);
        relu(hidden_1);

        // Second layer
        linear(hidden_1, weights_layer_2, hidden_2, hidden_size, hidden_size);
        relu(hidden_2);

        // Third layer
        linear(hidden_2, weights_layer_3, output, hidden_size, output_size);
    }
};

int main() {
    int input_size = 28 * 28;
    int hidden_size = 512;
    int output_size = 10;

    NeuralNetwork nn(input_size, hidden_size, output_size);

    std::vector<float> input(input_size); // Initialize with input data
    std::vector<float> output(output_size);

    nn.forward(input, output);

    // Output now contains the network's prediction
    return 0;
}
