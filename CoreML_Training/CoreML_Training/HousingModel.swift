//
//  HousingModel.swift
//  CoreMLTraining
//
//  Created by Jacopo Mangiavacchi on 3/19/20.
//

import Foundation
import CoreML

public struct HousingModel {
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
//
//    public func inferenceCoreML(model: MLModel, x: [Float]) -> Float {
//        let inputName = "input"
//
//        let multiArr = try! MLMultiArray(shape: [13], dataType: .float32)
//        for c in 0..<(numColumns-1) {
//            multiArr[c] = NSNumber(value: x[c])
//        }
//
//        let inputValue = MLFeatureValue(multiArray: multiArr)
//        let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue]
//        let provider = try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)
//
//        let prediction = try! model.prediction(from: provider)
//
//        return Float(prediction.featureValue(for: "output")!.multiArrayValue![0].floatValue)
//    }
}
