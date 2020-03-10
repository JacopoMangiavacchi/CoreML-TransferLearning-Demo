import Foundation
import CoreML

func compileCoreML(path: String) -> (MLModel, URL) {
    let modelUrl = URL(fileURLWithPath: path)
    let compiledUrl = try! MLModel.compileModel(at: modelUrl)
    
    print("Compiled Model Path: \(compiledUrl)")
    return try! (MLModel(contentsOf: compiledUrl), compiledUrl)
}

func inferenceCoreML(model: MLModel, numerical: [Float], cat1: Int, cat2: Int) -> Float {
    let numericalMultiArr = try! MLMultiArray(shape: [11], dataType: .float32)
    for i in 0..<11 {
        numericalMultiArr[i] = NSNumber(value: numerical[i])
    }
    let numericalValue = MLFeatureValue(multiArray: numericalMultiArr)

    let categorical1MultiArr = try! MLMultiArray(shape: [1], dataType: .float32)
    categorical1MultiArr[0] = NSNumber(value: Float(cat1))
    let categorical1Value = MLFeatureValue(multiArray: categorical1MultiArr)

    let categorical2MultiArr = try! MLMultiArray(shape: [1], dataType: .float32)
    categorical2MultiArr[0] = NSNumber(value: Float(cat2))
    let categorical2Value = MLFeatureValue(multiArray: categorical2MultiArr)

    let dataPointFeatures: [String: MLFeatureValue] = ["numericalInput": numericalValue,
                                                       "categoricalInput1": categorical1Value,
                                                       "categoricalInput2": categorical2Value]
    let provider = try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)
    
    let prediction = try! model.prediction(from: provider)

    return Float(prediction.featureValue(for: "output")!.multiArrayValue![0].floatValue)
}

let (coreModel, compiledModelUrl) = compileCoreML(path: "/Users/jacopo/S4TF-EmbeddingMultiInput/s4tf_house_simplified_trained_model.mlmodel")

// print(coreModel.modelDescription)

let numerical_0: [Float] = [  10.127699, -0.56208307,   1.3125796,   1.4050479,  -0.8863933,   1.2248203,  -1.2581577,
2.3634233,   0.9779133,  0.12650238,   1.6906141]
let cat1_0 = 0
let cat2_0 = 8

print(inferenceCoreML(model: coreModel, numerical: numerical_0, cat1: cat1_0, cat2: cat2_0))

let numerical_1: [Float] = [   1.541963, -0.56208307,   1.3125796,   0.7150559, -0.93426037,   0.7972452,  -1.0169381,
2.3634233,   0.9779133,   -2.180478,  0.39479825]
let cat1_1 = 0
let cat2_1 = 8

print(inferenceCoreML(model: coreModel, numerical: numerical_1, cat1: cat1_1, cat2: cat2_1))

let numerical_2: [Float] = [-0.29233322, -0.56208307,   2.7879307,   0.6713855,  -0.4761084,   0.6558696, -0.94413173,
2.7039568,   0.9328142,  0.42111015,  0.28535435]
let cat1_2 = 0
let cat2_2 = 3

print(inferenceCoreML(model: coreModel, numerical: numerical_2, cat1: cat1_2, cat2: cat2_2))
