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
    @State var data = HousingData()

    var body: some View {
        Text("\(data.prepareTrainingBatch().count)")
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
