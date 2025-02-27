# AI Project

This project implements various neural network architectures using PyTorch for different tasks, including classification of the Iris dataset and handwritten digit recognition using the MNIST dataset.

## Project Structure

```
AI-Project
├── .github
│   └── workflows
│       └── run-tests.yml
├── ANN
│   └── basic_nn.py
├── CNN
│   └── cnn.py
├── RNN
│   └── rnn.py
├── GRU
│   └── gru.py
├── LSTM
│   └── lstm.py
├── requirements.txt
└── README.md
```

## Overview

- **ANN (Artificial Neural Network)**: Implements a basic feedforward neural network to classify the Iris dataset.
- **CNN (Convolutional Neural Network)**: Implements a convolutional neural network to classify handwritten digits from the MNIST dataset.
- **RNN (Recurrent Neural Network)**: Implements a recurrent neural network for classifying handwritten digits from the MNIST dataset.
- **GRU (Gate Recurrent Unit Neural Network)**: Implements a gate recurrent unit neural network for classifying handwritten digits from the MNIST dataset.
- **LSTM (Long-Short Term Neural Network)**: Implements a long-short term neural network for classifying handwritten digits from the MNIST dataset.

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AI-Project.git
   cd AI-Project
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

- To train the ANN model on the Iris dataset, run:
  ```
  python ANN/basic_nn.py
  ```

- To train the CNN model on the MNIST dataset, run:
  ```
  python CNN/cnn.py
  ```

- To train the RNN model on the MNIST dataset, run:
  ```
  python RNN/rnn.py
  ```
- To train the GRU model on the MNIST dataset, run:
  ```
  python GRU/gru.py
  ```
- To train the LSTM model on the MNIST dataset, run:
  ```
  python LSTM/lstm.py
  ```


## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License License.
