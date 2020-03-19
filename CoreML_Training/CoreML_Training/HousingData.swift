//
//  HousingData.swift
//  CoreML_Training
//
//  Created by Jacopo Mangiavacchi on 3/19/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import CoreML

public struct HousingData {
    let trainPercentage:Float = 0.8
    
    public let numRecords: Int
    public let numColumns: Int
    public let numCategoricalFeatures: Int
    public let numNumericalFeatures: Int
    public let numLabels: Int
    public let numTrainRecords: Int
    public let numTestRecords: Int
    
    public let allCategoriesValues: [[Int32]]
    public let mean: [Float]
    public let std: [Float]
    
    public let xNumericalTrain: [[Float]]
    public let xCategoricalTrain: [[Int32]]
    public let yTrain: [[Float]]

    public let xNumericalTest: [[Float]]
    public let xCategoricalTest: [[Int32]]
    public let yTest: [[Float]]

    static func matrixTranspose<T>(_ matrix: [[T]]) -> [[T]] {
        if matrix.isEmpty {return matrix}
        var result = [[T]]()
        for index in 0..<matrix.first!.count {
            result.append(matrix.map{$0[index]})
        }
        return result
    }
    
    public init() {
        // Load Data
        let filePath = Bundle.main.url(forResource: "housing", withExtension: "csv")
        let data = try! String(contentsOf: filePath!, encoding: String.Encoding.utf8)

        // Convert Space Separated CSV with no Header
        let dataRecords: [[Float]] = data.split(separator: "\n").map{ String($0).split(separator: " ").compactMap{ Float(String($0)) } }

        // Data Ingestion
        let numRecords = dataRecords.count
        let numColumns = dataRecords[0].count

        // Randomize Data
        var index = Set<Int>()
        while index.count < numRecords {
            index.insert(Int.random(in: 0..<numRecords))
        }
        let randomDataRecords = index.map{ dataRecords[$0] }
        let dataFeatures = randomDataRecords.map{ Array($0[0..<numColumns-1]) }
        let dataLabels = randomDataRecords.map{ Array($0[(numColumns-1)...]) }
        
        // Split Numerical Categorical Features
        let categoricalColumns = [3, 8]
        let numericalColumns = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12]
        let numCategoricalFeatures = categoricalColumns.count
        let numNumericalFeatures = numericalColumns.count
        let numLabels = 1
        assert(numColumns == numCategoricalFeatures + numNumericalFeatures + numLabels)
        
        // Get Categorical Features
        let allCategoriesValues = dataFeatures.map{ row in categoricalColumns.map{ Int32(row[$0]) } }
                                        .reduce(into: Array(repeating: [Int32](), count: 2)){ total, value in
                                            total[0].append(value[0])
                                            total[1].append(value[1]) }
                                        .map{ Set($0).sorted() }
        let categoricalFeatures = dataFeatures.map{ row in categoricalColumns.map{ Int32(row[$0]) } }

        // Get Numerical Features
        let numericalFeatures = dataFeatures.map{ row in numericalColumns.map{ row[$0] } }
        
        // Categorize Categorical Features with Ordinal values
        var categoricalValues = Array(repeating: Set<Int32>(), count: 2)

        for record in categoricalFeatures {
            categoricalValues[0].insert(record[0])
            categoricalValues[1].insert(record[1])
        }

        let sortedCategoricalValues = [categoricalValues[0].sorted(), categoricalValues[1].sorted()]

        let ordinalCategoricalFeatures = categoricalFeatures.map{ [Int32(sortedCategoricalValues[0].firstIndex(of:$0[0])!),
                                                                   Int32(sortedCategoricalValues[1].firstIndex(of:$0[1])!)] }
        
        // Split Train and Test
        let numTrainRecords = Int(ceil(Float(numRecords) * trainPercentage))
        let numTestRecords = numRecords - numTrainRecords
        let xCategoricalAllTrain = HousingData.matrixTranspose(Array(ordinalCategoricalFeatures[0..<numTrainRecords]))
        let xCategoricalAllTest = HousingData.matrixTranspose(Array(ordinalCategoricalFeatures[numTrainRecords...]))
        let xNumericalAllTrain = Array(numericalFeatures[0..<numTrainRecords])
        let xNumericalAllTest = Array(numericalFeatures[numTrainRecords...])
        let yAllTrain = Array(dataLabels[0..<numTrainRecords])
        let yAllTest = Array(dataLabels[numTrainRecords...])
        
        // Normalize Numerical Features
        var xTrainNormalized = xNumericalAllTrain
        var xTestNormalized = xNumericalAllTest

        var mean = Array(repeating: Float(0), count: numNumericalFeatures)
        for r in xTrainNormalized {
            for c in 0..<mean.count {
                mean[c] = mean[c] + r[c]
            }
        }
        for c in 0..<mean.count {
            mean[c] = mean[c] / Float(numTrainRecords)
        }

        var std = Array(repeating: Float(0), count: numNumericalFeatures)
        for r in xTrainNormalized {
            for c in 0..<mean.count {
                std[c] = std[c] + pow(r[c] - mean[c], 2.0)
            }
        }
        for c in 0..<mean.count {
            std[c] = std[c] / Float(numTrainRecords - 1)
        }

        for r in 0..<xTrainNormalized.count {
            for c in 0..<numNumericalFeatures {
                xTrainNormalized[r][c] = (xTrainNormalized[r][c] - mean[c]) / std[c]
            }
        }

        for r in 0..<xTestNormalized.count {
            for c in 0..<numNumericalFeatures {
                xTestNormalized[r][c] = (xTestNormalized[r][c] - mean[c]) / std[c]
            }
        }
        
        // Initialize class properties
        self.numRecords = numRecords
        self.numColumns = numColumns
        self.numCategoricalFeatures = numCategoricalFeatures
        self.numNumericalFeatures = numNumericalFeatures
        self.numLabels = numLabels
        self.numTrainRecords = numTrainRecords
        self.numTestRecords = numTestRecords
        self.allCategoriesValues = allCategoriesValues
        self.mean = mean
        self.std = std

        self.xNumericalTrain = xTrainNormalized
        self.xCategoricalTrain = xCategoricalAllTrain
        self.yTrain = yAllTrain

        self.xNumericalTest = xTestNormalized
        self.xCategoricalTest = xCategoricalAllTest
        self.yTest = yAllTest
    }
    
    func prepareTrainingBatch(numericalInput: String = "numericalInput", categoricalInput1: String = "categoricalInput1", categoricalInput2: String = "categoricalInput2", output_true: String = "output_true") -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()

        for r in 0..<numTrainRecords {
            let numericalInputMultiArr = try! MLMultiArray(shape: [NSNumber(value: numNumericalFeatures)], dataType: .float32)
            let categoricalInput1MultiArr = try! MLMultiArray(shape: [NSNumber(value: 1)], dataType: .int32)
            let categoricalInput2MultiArr = try! MLMultiArray(shape: [NSNumber(value: 1)], dataType: .int32)
            let outputMultiArr = try! MLMultiArray(shape: [NSNumber(value: numLabels)], dataType: .float32)

            for c in 0..<numNumericalFeatures {
                numericalInputMultiArr[c] = NSNumber(value: xNumericalTrain[r][c])
            }

            categoricalInput1MultiArr[0] = NSNumber(value: xCategoricalTrain[0][r])
            categoricalInput2MultiArr[0] = NSNumber(value: xCategoricalTrain[1][r])
            outputMultiArr[0] = NSNumber(value: yTrain[r][0])

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
}
