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

let (coreModel, compiledModelUrl) = compileCoreML(path: "/Users/jacopo/S4TF-EmbeddingMultiInput/model/s4tf_house_simplified_trained_model.mlmodel")

// print(coreModel.modelDescription)

let numerical_0: [Float] = [-0.4000649,  -0.49060422,  -0.37006438,  -0.29516807,    0.2773472,     1.021877,
-0.6626676,  -0.11198587,    1.1625932,   0.40813962, -0.047246937]
let cat1_0 = 0
let cat2_0 = 4

print(inferenceCoreML(model: coreModel, numerical: numerical_0, cat1: cat1_0, cat2: cat2_0))

let numerical_1: [Float] = [2.421373, -0.49060422,  0.99975175,   1.2359748,   -2.246887,    1.124838,  -1.1245025,
1.5638397,   0.8357405,  0.42682302,     2.17172]
let cat1_1 = 0
let cat2_1 = 8

print(inferenceCoreML(model: coreModel, numerical: numerical_1, cat1: cat1_1, cat2: cat2_1))

let numerical_2: [Float] = [-0.32639477,  0.38105497,  -1.0291269,   0.7851385,  -0.9889262, -0.19590291,  -0.8761533,
-0.82510316,  -2.5261788,   0.3761753, -0.29949686]
let cat1_2 = 0
let cat2_2 = 4

print(inferenceCoreML(model: coreModel, numerical: numerical_2, cat1: cat1_2, cat2: cat2_2))
