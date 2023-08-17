#include <iostream>
#include <vector>
#include <SFML/Graphics.hpp>
#include <fstream>
#include <iomanip>

#include <omp.h>
#include <stdio.h>

#include "readMnist.h"

void printList(sf::Vector2i size, std::vector<double> list) {
    for (int i = 0; i < size.x; i++)
    {
        std::cout << "| ";
        for (int j = 0; j < size.y; j++)
        {
            std::cout << list[j + i * size.y] << " ";
        }
        std::cout << "|\n";
    }

    std::cout << "\n";

}

double activationFunction(double x) {
    return 1 / (1 + exp(-x));
}

void moveImage(std::vector<double> &inputs, sf::Vector2f offset) {
    std::vector<double> output(784, 0);

    for (int x = 0; x < 28; x++)
    {
        for (int y = 0; y < 28; y++)
        {

            if ((x + offset.x) < 0 || (x + offset.x) > 27 || (y + offset.y) < 0 || (y + offset.y) > 27) {
                continue;
            }

            output[(x + offset.x) + (y + offset.y) * 28] = inputs[x + y * 28];

        }
    }
    inputs = output;
}

void rotateImage(std::vector<double> inputs, float deg) {
    std::vector<double> output(784, 0);
    sf::Vector2f offset = { -14, -14 };


    for (int x = 0; x < 28; x++)
    {
        for (int y = 0; y < 28; y++)
        {
            double newX = cos(deg) * (x + offset.x) + -sin(deg) * (y + offset.y);
            double newY = sin(deg) * (x + offset.x) + cos(deg) * (y + offset.y);

            newX -= offset.x;
            newY -= offset.y;

            if (newX > 27 || newX < 0 || newY > 27 || newY < 0) {
                continue;
            }

            output[std::round(newX) + std::round(newY) * 28] = inputs[x + y * 28];

        }
    }

}

void resize(std::vector<double>& input, float scaleFactor) {

    std::vector<double> output(784, 0);

    int offset = -14;

    for (int x = 0; x < 28; x += 1)
    {
        for (int y = 0; y < 28; y += 1)
        {

            float dX = (x + offset) * scaleFactor;
            float dY = (y + offset) * scaleFactor;

            float d1 = std::round(dX) - offset;
            float d2 = std::round(dY) - offset;


            if (d1 < 0 || d1 >= 28 || d2 < 0 || d2 >= 28) {
                continue;
            }

            output[d1 + d2 * 28] = input[x + y * 28];

            //output[(x * reSize + 14 * reSize) + (y * reSize + 14 * reSize) * 28] = avgColor;

        }
    }
    input = output;
}


void randomizeImage(std::vector<double>& inputs) {

    float randomFactor = (rand() % 7);

    float sizeFactor = 0.4 + randomFactor / 10;

    resize(inputs, sizeFactor);

    int moveFactor = (1 - sizeFactor) * 40;

    if (moveFactor == 0) {
        return;
    }

    float halfMoveFactor = moveFactor / 2;

    moveImage(inputs, { rand() % moveFactor - halfMoveFactor, rand() % moveFactor - halfMoveFactor });
}


class Layer {

    int numNodesIn;
    int numNodesOut;

    public:
        std::vector<double> outputValues;
        std::vector<double> inputValues;

        std::vector<double> weights;
        std::vector<double> biases;

        std::vector<double> gradientWeights;
        std::vector<double> gradientBiases;

        std::vector<double> previousDeltaW;


        Layer(int inputSize, int outputSize): 
            weights(inputSize* outputSize, 0),
            biases(outputSize, 0),
            gradientWeights(inputSize* outputSize, 0),
            previousDeltaW(inputSize* outputSize, 0),
            gradientBiases(outputSize, 0),
            outputValues(outputSize, 0)
        {
            for (int j = 0; j < weights.size(); j++)
            {
                float value = rand() % 100 - 50;
                weights[j] = value / 100;

            }
        }

        std::vector<double> calculateOutputs(std::vector<double> &input) {

            numNodesIn = input.size();
            numNodesOut = outputValues.size();

            inputValues = input;


            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                double weightedInput = biases[nodeOut];

                for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
                {
                    const double weight = weights[nodeOut + nodeIn * numNodesOut];
                    weightedInput += input[nodeIn] * weight;
                }

                outputValues[nodeOut] = activationFunction(weightedInput);
            }

            return outputValues;

        }

        void updateGradients(std::vector<double>& derivatives) {

            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
                {
                    gradientWeights[nodeOut + nodeIn * numNodesOut] += inputValues[nodeIn] * derivatives[nodeOut];
                }
                gradientBiases[nodeOut] += derivatives[nodeOut];
            }
        
        }

        std::vector<double> calculateOutputCostDerivative(std::vector<double>& labelOutput) {
            std::vector<double> activationCostDerivative(numNodesOut, 0);
             
            for (int i = 0; i < activationCostDerivative.size(); i++)
            {
                double derivative = (outputValues[i] * (1 - outputValues[i])) * 2 * (outputValues[i] - labelOutput[i]);

                activationCostDerivative[i] = derivative;

            }

            return activationCostDerivative;
        }

        std::vector<double> calculateHiddenCostDerivative(std::vector<double>& oldCostDerivative) {

            std::vector<double> costDerivatives(numNodesIn, 0);

            for (int newIndex = 0; newIndex < costDerivatives.size(); newIndex++)
            {
                double costDerivative = 0;

                for (int oldIndex = 0; oldIndex < oldCostDerivative.size(); oldIndex++)
                {
                    costDerivative += weights[oldIndex + newIndex * numNodesOut] * oldCostDerivative[oldIndex];
                }

                costDerivative *= (inputValues[newIndex] * (1 - inputValues[newIndex]));
                costDerivatives[newIndex] = costDerivative;
            }

            return costDerivatives;
        }

        std::vector<double> calculateGradients(std::vector<double>& derivativeInput, double learn_rate) {

            std::vector<double> activationCostDerivative(numNodesIn, 0);

            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {

                double derivative = (outputValues[nodeOut] * (1 - outputValues[nodeOut])) * derivativeInput[nodeOut];

                for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
                {
                    weights[nodeOut + nodeIn * numNodesOut] += inputValues[nodeIn] * derivative * -learn_rate;

                    activationCostDerivative[nodeIn] += weights[nodeOut + nodeIn * numNodesOut] * derivative;

                }
                biases[nodeOut] += derivative;
            }

            return activationCostDerivative;

        }


};

class neuralNetwork {
    private: 
        int amountOfLayers;
        int batchAmount = 600;
        int batchSize = 100;
        int testSize = 500;
        int epochs = 30;
        double learn_rate = 0.45;
        double momentum_rate = 0.7; // 0.9 optimal

        const int numThreads = 4;  // Number of threads to use

    public:
        std::vector<Layer> layers;

        neuralNetwork(std::vector<int> layerSizes) {
            amountOfLayers = layerSizes.size();

            for (int i = 0; i < amountOfLayers - 1; i++)
            { 
                layers.push_back(Layer(layerSizes[i], layerSizes[i + 1]));
            }
        }

        void test(unsigned char** &imageData, unsigned char* &labelData) {

            float amountCorrect = 0;
            float amountChecked = 0;
            std::vector<double> input(784,0);

            for (int i = 0; i < testSize; i++) {

                int label = static_cast<int>(labelData[i]);

                for (int j = 0; j < 784; j++)
                {
                    input[j] = static_cast<double>(imageData[i][j]) / 255;
                }

                randomizeImage(input);

                if (classify(input) == label) {
                    amountCorrect++;
                }
                
                amountChecked++;

            }

            float accuracy = (amountCorrect / amountChecked) * 100;

            std::cout << (amountCorrect) << "/" << amountChecked << " Correct\n";
            std::cout << "Accuracy : " << accuracy << "%\n";

        };



        void updateGradients(std::vector<double>& inputs, double labelIndex) {
            classify(inputs);

            Layer& outputLayer = layers[layers.size() - 1];

            std::vector<double> label(outputLayer.outputValues.size(), 0);
            label[labelIndex] = 1;

            std::vector<double> derivatives = outputLayer.calculateOutputCostDerivative(label);
            outputLayer.updateGradients(derivatives);


            for (int hiddenLayerIndex = layers.size() - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
            {
                Layer& hiddenLayer = layers[hiddenLayerIndex];
                derivatives = layers[hiddenLayerIndex + 1].calculateHiddenCostDerivative(derivatives);
                hiddenLayer.updateGradients(derivatives);
            }
        }

        void applyGradients() {

            const float scaleFactor = -learn_rate / batchSize;

            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++)
            {

                Layer& currentLayer = layers[layerIndex];

                for (int i = 0; i < currentLayer.weights.size(); i++)
                {

                    double deltaW = currentLayer.gradientWeights[i] * scaleFactor;

                    double previousDeltaW = currentLayer.previousDeltaW[i];

                    currentLayer.weights[i] += deltaW + (previousDeltaW * momentum_rate);

                    currentLayer.previousDeltaW[i] = deltaW + (previousDeltaW * momentum_rate);

                    currentLayer.gradientWeights[i] = 0;

                }

                for (int i = 0; i < currentLayer.biases.size(); i++)
                {
                    currentLayer.biases[i] += currentLayer.gradientBiases[i] * scaleFactor;
                    currentLayer.gradientBiases[i] = 0;
                }

            }
        }

        void learn(unsigned char** &imageData, unsigned char* &labelData) {

            std::vector<double> input(784, 0);

            sf::Clock clock;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int batch = 0; batch < batchAmount; batch++)
                {

                    for (int index = batch * batchSize; index < batch * batchSize + batchSize; index++)
                    {

                        for (int i = 0; i < 784; i++) {
                            input[i] = static_cast<double>(imageData[index][i]) / 255;
                        }

                        randomizeImage(input);

                        double label = static_cast<double>(labelData[index]);

                        updateGradients(input, label);

                    }

                    applyGradients();
                }
                std::cout << "Epoch : " << epoch << " : TIME : " << clock.restart().asSeconds() << '\n';
                test(imageData, labelData);
                clock.restart().asSeconds();
            }

        }


        int classify(std::vector<double> input) {

            for (int currentLayer = 0; currentLayer < amountOfLayers - 1; currentLayer++)
            {
                input = layers[currentLayer].calculateOutputs(input);
            }


            int maxInt = 0;
            double maxOutput = input[0];

            for (int i = 0; i < input.size(); i++)
            {
                if (input[i] > maxOutput) {
                    maxOutput = input[i];
                    maxInt = i;
                }
            }


            return maxInt;

        }

};

void loadNetFromFile(neuralNetwork* network) {
    std::fstream file;

    file.open("networkWeights.txt", std::ios::in); //open a file to perform read operation using file object
    if (file.is_open()) { //checking whether the file is open
        std::string tp;
        std::string number;
        int index = 0;
        int currentArray = 0;

        while (std::getline(file, tp)) { //read data from file object and put it into string.

            for (int i = 0; i < tp.size(); i++)
            {
                if (tp[i] == *"n") {
                    currentArray++;
                    index = 0;
                    continue;
                }

                if (tp[i] == *",") {

                    if (currentArray == 0) {
                        network->layers[0].weights[index] = std::stod(number);
                    }
                    else if (currentArray == 1) {
                        network->layers[1].weights[index] = std::stod(number);
                    }
                    else if (currentArray == 2) {
                        network->layers[0].biases[index] = std::stod(number);
                    }
                    else if (currentArray == 3) {
                        network->layers[1].biases[index] = std::stod(number);
                    }


                    number = "";
                    index++;
                }
                else {
                    number += tp[i];
                }
            }

        }
        file.close(); //close the file object.
    }

    std::cout << "Successfully loaded file" << "\n" << "\n";

}

void saveNetToFile(neuralNetwork* network) {
    std::ofstream fw("C:\\Users\\theow\\source\\repos\\newAItest\\newAItest\\networkWeights.txt", std::ofstream::out);

    if (fw.is_open()) {

        for (int i = 0; i < network->layers[0].weights.size(); i++)
        {
            fw << network->layers[0].weights[i] << ",";
        }
        fw << 'n';

        for (int i = 0; i < network->layers[1].weights.size(); i++)
        {
            fw << network->layers[1].weights[i] << ",";
        }
        fw << 'n';

        for (int i = 0; i < network->layers[0].biases.size(); i++)
        {
            fw << network->layers[0].biases[i] << ",";
        }
        fw << 'n';

        for (int i = 0; i < network->layers[1].biases.size(); i++)
        {
            fw << network->layers[1].biases[i] << ",";
        }
        fw << 'n';

    }
    else {
        std::cout << "Problem with opening file";
        return;
         };

    std::cout << "\n" << "Successfully saved to file" << "\n";

}

void drawImage(std::vector<double> &number, sf::RenderWindow* _window) {

    sf::VertexArray rectangle;
    rectangle.setPrimitiveType(sf::Quads);
    rectangle.resize(28 * 28 * 4);

    sf::Color color = sf::Color::Black;

    for (int x = 0; x < 28; x += 1)
    {
        for (int y = 0; y < 28; y += 1)
        {
            sf::Vertex* quad = &rectangle[(x + y * 28) * 4];

            sf::Color color = sf::Color(number[x + y * 28], number[x + y * 28], number[x + y * 28]);

            quad[1].position = sf::Vector2f(x * 28, y * 28);
            quad[1].color = color;
            quad[0].position = sf::Vector2f(x * 28, y * 28 + 28);
            quad[0].color = color;
            quad[2].position = sf::Vector2f(x * 28 + 28, y * 28);
            quad[2].color = color;
            quad[3].position = sf::Vector2f(x * 28 + 28, y * 28 + 28);
            quad[3].color = color;
        }
    }
    _window->draw(rectangle);
}

int main()
{
    srand(time(NULL));
    sf::RenderWindow window({784, 784 }, "");

    neuralNetwork network({784,200,10});

    unsigned char** imageSet = read_mnist_images("train-images.idx3-ubyte", 60000);
    unsigned char* labelSet = read_mnist_labels("train-labels.idx1-ubyte", 60000);

    unsigned char** testImageSet = read_mnist_images("t10k-images.idx3-ubyte", 10000);
    unsigned char* testLabelSet = read_mnist_labels("t10k-labels.idx1-ubyte", 10000);



    loadNetFromFile(&network);
    
    //network.learn(imageSet, labelSet);

    //saveNetToFile(&network);
    

    std::vector<double> drawedNumber(784, 0);



    sf::Clock clock;
    while (window.isOpen())
    {
        float currentTime = clock.restart().asSeconds();
        float fps = 1.f / (currentTime);
        float dT = 1000 / fps;

        window.setTitle(std::to_string(fps));

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::MouseButtonReleased) {
                system("cls");
                network.classify(drawedNumber);


                std::cout << std::fixed << std::setprecision(0) << "0 : " << network.layers[1].outputValues[0] * 100 << "%\n";
                std::cout << "1 : " << network.layers[1].outputValues[1] * 100 << "%\n";
                std::cout << "2 : " << network.layers[1].outputValues[2] * 100 << "%\n";
                std::cout << "3 : " << network.layers[1].outputValues[3] * 100 << "%\n";
                std::cout << "4 : " << network.layers[1].outputValues[4] * 100 << "%\n";
                std::cout << "5 : " << network.layers[1].outputValues[5] * 100 << "%\n";
                std::cout << "6 : " << network.layers[1].outputValues[6] * 100 << "%\n";
                std::cout << "7 : " << network.layers[1].outputValues[7] * 100 << "%\n";
                std::cout << "8 : " << network.layers[1].outputValues[8] * 100 << "%\n";
                std::cout << "9 : " << network.layers[1].outputValues[9] * 100 << "%\n";

                std::cout << "Guess : " << network.classify(drawedNumber) << "\n";
            }

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape) {
                    window.close();
                }

                if (event.key.code == sf::Keyboard::Space) {
                    std::fill(drawedNumber.begin(), drawedNumber.end(), 0);
                }


            }
        }

        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            int x = sf::Mouse::getPosition(window).x / 28;
            int y = sf::Mouse::getPosition(window).y / 28;

            if (x > 0 && x < 28 && y > 0 && y < 28) {
                drawedNumber[x + y * 28] = 255;
            }
        }

        window.clear();
        drawImage(drawedNumber, &window);
        window.display();

    }

    return 0;
}
