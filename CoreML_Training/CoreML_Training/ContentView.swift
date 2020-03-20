//
//  ContentView.swift
//  CoreML_Training
//
//  Created by Jacopo Mangiavacchi on 3/18/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import SwiftUI
import CoreML

struct ContentView: View {
    @State var model = HousingModel()
    
    @State var changeRatio = false
    @State var newRatio = 0.8

    @State var testSampleNumber = 0.0

    var body: some View {
        Form {
            Section(header: Text("Dataset")) {
                VStack {
                    HStack {
                        VStack {
                            if changeRatio {
                                Text("\(Int(Double(model.data.numRecords) * self.newRatio) + 1) Train Samples")
                                Text("\(model.data.numRecords - Int(Double(model.data.numRecords) * self.newRatio) - 1) Test Samples")
                            }
                            else {
                                Text("\(model.data.numTrainRecords) Train Samples")
                                Text("\(model.data.numTestRecords) Test Samples")
                            }
                        }
                        Spacer()
                        if changeRatio {
                            Text("\(Int(newRatio*100)) %")
                        }
                        else {
                            Text("\(Int(model.data.trainPercentage * 100)) %")
                        }
                        Spacer()
                        HStack {
                            if self.changeRatio {
                                Button(action: {}) {
                                    Text("Cancel")
                                }.onTapGesture {
                                    self.changeRatio = false
                                    self.newRatio = Double(self.model.data.trainPercentage)
                                }
                            }
                            Button(action: {}) {
                                if changeRatio {
                                    Text("Confirm").foregroundColor(.red)
                                }
                                else {
                                    Text("Change")
                                }
                            }.onTapGesture {
                                if self.changeRatio {
                                    self.model.randomizeData(trainPercentage: Float(self.newRatio))
                                    self.changeRatio = false
                                    self.testSampleNumber = 0.0
                                }
                                else {
                                    self.changeRatio = true
                                }
                            }
                        }
                    }
                    if changeRatio {
                        Slider(value: $newRatio, in: 0.05...0.99, step: 0.01)
                    }
                }
            }
            Section(header: Text("Training")) {
                Text("TODO")
            }
            Section(header: Text("Inferencing")) {
                HStack {
                    Text("Expected:")
                    Spacer()
                    Text(String(format: "%.2f", self.model.data.yTest[Int(self.testSampleNumber)][0]))
                }
                HStack {
                    Text("Pre-Trained Model Prediction:")
                    Spacer()
                    Text(String(format: "%.2f", self.model.inference(testSample: Int(self.testSampleNumber))))
                }
                HStack {
                    Text("Re-Trained Model Prediction:")
                    Spacer()
                    Text(String(format: "%.2f", self.model.inference(retrained: true, testSample: Int(self.testSampleNumber))))
                }
                HStack {
                    Text("Sample Number:")
                    Spacer()
                    Text("\(Int(self.testSampleNumber) + 1)")
                }
                Slider(value: $testSampleNumber, in: 0.0...Double(self.model.data.numTestRecords - 1), step: 1.0)
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
