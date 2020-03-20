//
//  HousingModel.swift
//  CoreMLTraining
//
//  Created by Jacopo Mangiavacchi on 3/19/20.
//

import Foundation
import CoreML

public struct HousingModel {
    let numericalInput: String = "numericalInput"
    let categoricalInput1: String = "categoricalInput1"
    let categoricalInput2: String = "categoricalInput2"
    let output: String = "output"
    let output_true: String = "output_true"
    
    var trained = false
    var data = HousingData()
    var defaultModel: MLModel?
    var retrainedModel: MLModel?
    
    mutating func randomizeData(trainPercentage: Float = 0.8) {
        data = HousingData(trainPercentage: trainPercentage)
    }

    public func inference(retrained: Bool = false, testSample: Int) -> Float {
        guard testSample >= 0 && testSample < data.numTestRecords else { return -1 }
        
        let xNumerical: [Float] = data.xNumericalTest[testSample]
        let xCategorical1: Float = Float(data.xCategoricalTest[0][testSample])
        let xCategorical2: Float = Float(data.xCategoricalTest[1][testSample])
        
        return inference(retrained: retrained, xNumerical: xNumerical, xCategorical1: xCategorical1, xCategorical2: xCategorical2)
    }
    
    public func inference(retrained: Bool = false, xNumerical: [Float], xCategorical1: Float, xCategorical2: Float) -> Float {
        let numericalInputMultiArr = try! MLMultiArray(shape: [NSNumber(value: data.numNumericalFeatures)], dataType: .float32)
        let categoricalInput1MultiArr = try! MLMultiArray(shape: [NSNumber(value: 1)], dataType: .float32)
        let categoricalInput2MultiArr = try! MLMultiArray(shape: [NSNumber(value: 1)], dataType: .float32)
        
        for c in 0..<data.numNumericalFeatures {
            numericalInputMultiArr[c] = NSNumber(value: xNumerical[c])
        }

        categoricalInput1MultiArr[0] = NSNumber(value: xCategorical1)
        categoricalInput2MultiArr[0] = NSNumber(value: xCategorical2)

        let numericalInputValue = MLFeatureValue(multiArray: numericalInputMultiArr)
        let categorical1InputValue = MLFeatureValue(multiArray: categoricalInput1MultiArr)
        let categorical2InputValue = MLFeatureValue(multiArray: categoricalInput2MultiArr)

        let dataPointFeatures: [String: MLFeatureValue] = [numericalInput: numericalInputValue,
                                                           categoricalInput1: categorical1InputValue,
                                                           categoricalInput2: categorical2InputValue]

        let provider = try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)

        let model = retrained ? retrainedModel : defaultModel
        
        guard let prediction = try! model?.prediction(from: provider) else { return -1 }

        return Float(prediction.featureValue(for: output)!.multiArrayValue![0].floatValue)
    }

    func prepareTrainingBatch() -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()

        for r in 0..<data.numTrainRecords {
            let numericalInputMultiArr = try! MLMultiArray(shape: [NSNumber(value: data.numNumericalFeatures)], dataType: .float32)
            let categoricalInput1MultiArr = try! MLMultiArray(shape: [NSNumber(value: 1)], dataType: .float32)
            let categoricalInput2MultiArr = try! MLMultiArray(shape: [NSNumber(value: 1)], dataType: .float32)
            let outputMultiArr = try! MLMultiArray(shape: [NSNumber(value: data.numLabels)], dataType: .float32)

            for c in 0..<data.numNumericalFeatures {
                numericalInputMultiArr[c] = NSNumber(value: data.xNumericalTrain[r][c])
            }

            categoricalInput1MultiArr[0] = NSNumber(value: data.xCategoricalTrain[0][r])
            categoricalInput2MultiArr[0] = NSNumber(value: data.xCategoricalTrain[1][r])
            outputMultiArr[0] = NSNumber(value: data.yTrain[r][0])

            let numericalInputValue = MLFeatureValue(multiArray: numericalInputMultiArr)
            let categorical1InputValue = MLFeatureValue(multiArray: categoricalInput1MultiArr)
            let categorical2InputValue = MLFeatureValue(multiArray: categoricalInput2MultiArr)
            let outputValue = MLFeatureValue(multiArray: outputMultiArr)

            let dataPointFeatures: [String: MLFeatureValue] = [numericalInput: numericalInputValue,
                                                               categoricalInput1: categorical1InputValue,
                                                               categoricalInput2: categorical2InputValue,
                                                               output_true: outputValue]

            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }

        return MLArrayBatchProvider(array: featureProviders)
    }
    
    public func compileCoreML(path: String) -> (MLModel, URL) {
        let modelUrl = URL(fileURLWithPath: path)
        let compiledUrl = try! MLModel.compileModel(at: modelUrl)
        
        print("Compiled Model Path: \(compiledUrl)")
        return try! (MLModel(contentsOf: compiledUrl), compiledUrl)
    }

//    public func train(url: URL, retrainedCoreMLFilePath: String) {
//        let configuration = MLModelConfiguration()
//        configuration.computeUnits = .all
//        //configuration.parameters = [.epochs : 100]
//
//        let progressHandler = { (context: MLUpdateContext) in
//            switch context.event {
//            case .trainingBegin:
//                print("Training begin")
//
//            case .miniBatchEnd:
//                break
////                let batchIndex = context.metrics[.miniBatchIndex] as! Int
////                let batchLoss = context.metrics[.lossValue] as! Double
////                print("Mini batch \(batchIndex), loss: \(batchLoss)")
//
//            case .epochEnd:
//                let epochIndex = context.metrics[.epochIndex] as! Int
//                let trainLoss = context.metrics[.lossValue] as! Double
//                print("Epoch \(epochIndex) end with loss \(trainLoss)")
//
//            default:
//                print("Unknown event")
//            }
//
//    //        print(context.model.modelDescription.parameterDescriptionsByKey)
//
//    //        do {
//    //            let multiArray = try context.model.parameterValue(for: MLParameterKey.weights.scoped(to: "dense_1")) as! MLMultiArray
//    //            print(multiArray.shape)
//    //        } catch {
//    //            print(error)
//    //        }
//        }
//
//        let completionHandler = { (context: MLUpdateContext) in
//            print("Training completed with state \(context.task.state.rawValue)")
//            print("CoreML Error: \(context.task.error.debugDescription)")
//
//            if context.task.state != .completed {
//                print("Failed")
//                return
//            }
//
//            let trainLoss = context.metrics[.lossValue] as! Double
//            print("Final loss: \(trainLoss)")
//
//
//            let updatedModel = context.model
//            let updatedModelURL = URL(fileURLWithPath: retrainedCoreMLFilePath)
//            try! updatedModel.write(to: updatedModelURL)
//
//            print("Model Trained!")
//            print("Press return to continue..")
//        }
//
//        let handlers = MLUpdateProgressHandlers(
//                            forEvents: [.trainingBegin, .miniBatchEnd, .epochEnd],
//                            progressHandler: progressHandler,
//                            completionHandler: completionHandler)
//
//
//        let updateTask = try! MLUpdateTask(forModelAt: url,
//                                           trainingData: prepareTrainingBatch(),
//                                           configuration: configuration,
//                                           progressHandlers: handlers)
//
//        updateTask.resume()
//    }
}
