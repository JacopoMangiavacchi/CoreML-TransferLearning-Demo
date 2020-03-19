//
//  BostonHousing.swift
//  CoreMLTraining
//
//  Created by Jacopo Mangiavacchi on 2/17/20.
//

import Foundation
import CoreML

public struct BostonHousingModel {
    let trainPercentage:Float = 0.8
    public let numRecords: Int
    public let numColumns: Int
    public let numTrainRecords: Int
    public let numTestRecords: Int
    public let xTrain: [[Float]]
    public let yTrain: [[Float]]
    public let xTest: [[Float]]
    public let yTest: [[Float]]

    public init() {
        let filePath = Bundle.main.url(forResource: "housing", withExtension: "csv")
        let data = try! String(contentsOf: filePath!, encoding: String.Encoding.utf8)

        // Convert Space Separated CSV with no Header
        let dataRecords: [[Float]] = data.split(separator: "\n").map{ String($0).split(separator: " ").compactMap{ Float(String($0)) } }

        let numRecords = dataRecords.count
        let numColumns = dataRecords[0].count

        let dataFeatures = dataRecords.map{ Array($0[0..<numColumns-1]) }
        let dataLabels = dataRecords.map{ Array($0[(numColumns-1)...]) }

        self.numRecords = numRecords
        self.numColumns = numColumns
        self.numTrainRecords = Int(ceil(Float(numRecords) * trainPercentage))
        self.numTestRecords = numRecords - numTrainRecords

        self.yTrain = Array(dataLabels[0..<numTrainRecords])
        self.yTest = Array(dataLabels[numTrainRecords...])
        var xTrainNormalized = Array(dataFeatures[0..<numTrainRecords])
        var xTestNormalized = Array(dataFeatures[numTrainRecords...])

        // Normalize
        
        var mean = Array(repeating: Float(0), count: numColumns - 1)
        for r in xTrainNormalized {
            for c in 0..<mean.count {
                mean[c] = mean[c] + r[c]
            }
        }
        for c in 0..<mean.count {
            mean[c] = mean[c] / Float(numTrainRecords)
        }

        var std = Array(repeating: Float(0), count: numColumns - 1)
        for r in xTrainNormalized {
            for c in 0..<mean.count {
                std[c] = std[c] + pow(r[c] - mean[c], 2.0)
            }
        }
        for c in 0..<mean.count {
            std[c] = std[c] / Float(numTrainRecords - 1)
        }

        for r in 0..<xTrainNormalized.count {
            for c in 0..<(numColumns - 1) {
                xTrainNormalized[r][c] = (xTrainNormalized[r][c] - mean[c]) / std[c]
            }
        }

        for r in 0..<xTestNormalized.count {
            for c in 0..<(numColumns - 1) {
                xTestNormalized[r][c] = (xTestNormalized[r][c] - mean[c]) / std[c]
            }
        }
        
//        print(mean)
//        print(std)

        self.xTrain = xTrainNormalized
        self.xTest = xTestNormalized
    }
    
    public func compileCoreML(path: String) -> (MLModel, URL) {
        let modelUrl = URL(fileURLWithPath: path)
        let compiledUrl = try! MLModel.compileModel(at: modelUrl)
        
        print("Compiled Model Path: \(compiledUrl)")
        return try! (MLModel(contentsOf: compiledUrl), compiledUrl)
    }

    func prepareTrainingBatch() -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()

        let inputName = "input"
        let outputName = "output_true"
        
        for r in 0..<numTrainRecords {
            let inputMultiArr = try! MLMultiArray(shape: [13], dataType: .float32)
            let outputMultiArr = try! MLMultiArray(shape: [1], dataType: .float32)

            for c in 0..<(numColumns-1) {
                inputMultiArr[c] = NSNumber(value: xTrain[r][c])
            }

            outputMultiArr[0] = NSNumber(value: yTrain[r][0])

            let inputValue = MLFeatureValue(multiArray: inputMultiArr)
            let outputValue = MLFeatureValue(multiArray: outputMultiArr)
             
            let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue,
                                                               outputName: outputValue]
             
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }
         
        return MLArrayBatchProvider(array: featureProviders)
    }
    
    public func train(url: URL, retrainedCoreMLFilePath: String) {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        //configuration.parameters = [.epochs : 100]

        let progressHandler = { (context: MLUpdateContext) in
            switch context.event {
            case .trainingBegin:
                print("Training begin")

            case .miniBatchEnd:
                break
//                let batchIndex = context.metrics[.miniBatchIndex] as! Int
//                let batchLoss = context.metrics[.lossValue] as! Double
//                print("Mini batch \(batchIndex), loss: \(batchLoss)")

            case .epochEnd:
                let epochIndex = context.metrics[.epochIndex] as! Int
                let trainLoss = context.metrics[.lossValue] as! Double
                print("Epoch \(epochIndex) end with loss \(trainLoss)")

            default:
                print("Unknown event")
            }

    //        print(context.model.modelDescription.parameterDescriptionsByKey)

    //        do {
    //            let multiArray = try context.model.parameterValue(for: MLParameterKey.weights.scoped(to: "dense_1")) as! MLMultiArray
    //            print(multiArray.shape)
    //        } catch {
    //            print(error)
    //        }
        }

        let completionHandler = { (context: MLUpdateContext) in
            print("Training completed with state \(context.task.state.rawValue)")
            print("CoreML Error: \(context.task.error.debugDescription)")

            if context.task.state != .completed {
                print("Failed")
                return
            }

            let trainLoss = context.metrics[.lossValue] as! Double
            print("Final loss: \(trainLoss)")

            
            let updatedModel = context.model
            let updatedModelURL = URL(fileURLWithPath: retrainedCoreMLFilePath)
            try! updatedModel.write(to: updatedModelURL)
            
            print("Model Trained!")
            print("Press return to continue..")
        }

        let handlers = MLUpdateProgressHandlers(
                            forEvents: [.trainingBegin, .miniBatchEnd, .epochEnd],
                            progressHandler: progressHandler,
                            completionHandler: completionHandler)
        
        
        let updateTask = try! MLUpdateTask(forModelAt: url,
                                           trainingData: prepareTrainingBatch(),
                                           configuration: configuration,
                                           progressHandlers: handlers)

        updateTask.resume()
    }
    
    public func inferenceCoreML(model: MLModel, x: [Float]) -> Float {
        let inputName = "input"
        
        let multiArr = try! MLMultiArray(shape: [13], dataType: .float32)
        for c in 0..<(numColumns-1) {
            multiArr[c] = NSNumber(value: x[c])
        }

        let inputValue = MLFeatureValue(multiArray: multiArr)
        let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue]
        let provider = try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)
        
        let prediction = try! model.prediction(from: provider)

        return Float(prediction.featureValue(for: "output")!.multiArrayValue![0].floatValue)
    }
}
