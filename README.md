# S4TF-EmbeddingMultiInput

Swift for TensorFlow Notebook to train a Regression model from tabular data with multiple Numerical and Categorical Features using Embedding and MultiInput.

## Dataset

The [Boston housing price dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/) (Creator: Harrison, D. and Rubinfeld, D.L. )has 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

Attribute Information (in order):

- CRIM     Numerical: per capita crime rate by town
- ZN       Numerical: proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    Numerical: proportion of non-retail business acres per town
- CHAS     Categorical: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      Numerical: nitric oxides concentration (parts per 10 million)
- RM       Numerical: average number of rooms per dwelling
- AGE      Numerical: proportion of owner-occupied units built prior to 1940
- DIS      Numerical: weighted distances to five Boston employment centres
- RAD      Categorical: index of accessibility to radial highways
- TAX      Numerical: full-value property-tax rate per ten thousand dollars
- PTRATIO  Numerical: pupil-teacher ratio by town
- B        Numerical: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT    Numerical: % lower status of the population
- MEDV     Numerical: Median value of owner-occupied homes in a thousand dollar

## Data transformations

- Categorical features are encoded with one-hot-encoding representation and feeded into Embedding layers.
- Numerical features are normalized using ZScore (x' = (x' - MEAN(X)) / STD(X)) fitting only training data
- Dataset is splitted in Train and Test

## S4TF Model

```swift
struct MultiInputs<N: Differentiable, C>: Differentiable {
  var numerical: N
  
  @noDerivative
  var categorical: C

  @differentiable
  init(numerical: N, categorical: C) {
    self.numerical = numerical
    self.categorical = categorical
  }
}

struct RegressionModel: Module {
    var numericalLayer = Dense<Float>(inputSize: 11, outputSize: 32, activation: relu)
    var embedding1 = Embedding<Float>(vocabularySize: 2, embeddingSize: 2)
    var embedding2 = Embedding<Float>(vocabularySize: 9, embeddingSize: 5)
    var embeddingLayer = Dense<Float>(inputSize: (4 + 45), outputSize: 64, activation: relu)
    var allInputConcatLayer = Dense<Float>(inputSize: (32 + 64), outputSize: 128, activation: relu)
    var hiddenLayer = Dense<Float>(inputSize: 128, outputSize: 32, activation: relu)
    var outputLayer = Dense<Float>(inputSize: 32, outputSize: 1)

    @differentiable
    func callAsFunction(_ input: MultiInputs<[Tensor<Float>], [Tensor<Int32>]>) -> Tensor<Float> {
        let numericalInput = numericalLayer(input.numerical[0])
        let embeddingOutput1 = embedding1(input.categorical[0])
        let embeddingOutput1Reshaped = embeddingOutput1.reshaped(to: 
            TensorShape([embeddingOutput1.shape[0], embeddingOutput1.shape[1] * embeddingOutput1.shape[2]]))
        let embeddingOutput2 = embedding2(input.categorical[1])
        let embeddingOutput2Reshaped = embeddingOutput2.reshaped(to: 
            TensorShape([embeddingOutput2.shape[0], embeddingOutput2.shape[1] * embeddingOutput2.shape[2]]))
        let embeddingConcat = Tensor<Float>(concatenating: [embeddingOutput1Reshaped, embeddingOutput2Reshaped], alongAxis: 1)
        let embeddingInput = embeddingLayer(embeddingConcat)
        let allConcat = Tensor<Float>(concatenating: [numericalInput, embeddingInput], alongAxis: 1)
        return allConcat.sequenced(through: allInputConcatLayer, hiddenLayer, outputLayer)
    }
}

var model = RegressionModel()
```
