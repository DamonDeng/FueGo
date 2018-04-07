/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Xin Li yakumolx@gmail.com
 */

#include <chrono>
#include "SgSystem.h"

#include "mxnet-cpp/MxNetCpp.h"
#include "SpRandomPlayer.h"
#include "GoInit.h"
#include "SgDebug.h"
#include "SgException.h"
#include "SgInit.h"
#include "GoBoard.h"
#include "GoUctBoard.h"
#include "GoUctPureRandomGenerator.h"
#include "SgTimeRecord.h"

#include "SgTimer.h"

#include <boost/utility.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>



using namespace std;
using namespace mxnet::cpp;

Symbol mlp(const vector<int> &layers) {
  auto x = Symbol::Variable("X");
  auto label = Symbol::Variable("label");

  vector<Symbol> weights(layers.size());
  vector<Symbol> biases(layers.size());
  vector<Symbol> outputs(layers.size());

  for (size_t i = 0; i < layers.size(); ++i) {
    weights[i] = Symbol::Variable("w" + to_string(i));
    biases[i] = Symbol::Variable("b" + to_string(i));
    Symbol fc = FullyConnected(
      i == 0? x : outputs[i-1],  // data
      weights[i],
      biases[i],
      layers[i]);
    outputs[i] = i == layers.size()-1 ? fc : Activation(fc, ActivationActType::kRelu);
  }

  return SoftmaxOutput(outputs.back(), label);
}

void printGoUctBoard(GoUctBoard& board){
    int boardSize = 19;
    SgBoardColor boardColor;
    string result = "";

    for (int i = 0; i< boardSize; i++){
        for (int j = 0; j< boardSize; j++){
            SgPoint curPoint = SgPointUtil::Pt(i, j);
            boardColor = board.GetColor(curPoint);
            if (boardColor == SG_BLACK){
                result = result + "*";
            } else if (boardColor == SG_WHITE){
                result = result + "O";
            } else{
                result = result + ".";
            }
            
        }
        result = result + "\n";
    }

    SgDebug() << result ;

}

void printGoBoard(GoBoard& board){
    int boardSize = 19;
    SgBoardColor boardColor;
    string result = "";

    for (int i = 1; i<= boardSize; i++){
        for (int j = 1; j<= boardSize; j++){
            SgDebug() << "i:" << i << "    j:" << j << ". \n";

            SgPoint curPoint = SgPointUtil::Pt(i, j);
            boardColor = board.GetColor(curPoint);
            if (boardColor == SG_BLACK){
                result = result + "*";
            } else if (boardColor == SG_WHITE){
                result = result + "O";
            } else{
                result = result + ".";
            }
            
        }
        result = result + "\n";
    }

    SgDebug() << result ;

}


int main(int argc, char** argv) {
  
    SgDebug() << "Starting the testing. \n";

    SgInit();
    GoInit();

    // SgPoint mv = SG_NULLMOVE;

    // SgDebug() << "NULLMOVE: " << mv << "\n";

    // GoBoard basicBoard;
    // GoUctBoard goBoard(basicBoard);

    // GoUctPlayoutPolicyParam param;

    // param.

    // GoUctPlayoutPolicy<GoUctBoard> curPolicy(goBoard, param);

    SgPoint testingPoint;

    // SgRandom sgRandom;
    // GoUctPureRandomGenerator<GoUctBoard> randomGenerator(goBoard, sgRandom);

    

    SgTimer timer;

    double start_time;
    start_time = timer.GetTime();

    for (int game_time = 0; game_time < 100; game_time ++){

        GoBoard basicBoard;
        // GoUctBoard goBoard(basicBoard);

        SpRandomPlayer testPlayer(basicBoard);

        // SgRandom sgRandom;
    
        // GoUctPureRandomGenerator<GoUctBoard> randomGenerator(goBoard, sgRandom);
        // randomGenerator.Start();

        SgBlackWhite colorBlack = SG_BLACK;
        SgBlackWhite colorWhite = SG_WHITE;
        
    
        for (int i =0; i< 900; i++){
            // testingPoint = curPolicy.GenerateMove();
            // testingPoint = randomGenerator.Generate();

            // SgDebug() << "move round: " << i << ". \n";

            SgTimeRecord timeRecord;

            timeRecord = SgTimeRecord(true, 1);

            auto testingPoint1 = testPlayer.GenMove(timeRecord, colorBlack);

            if (testingPoint1 != SG_NULLMOVE && testingPoint1 != SG_PASS){
                basicBoard.Play(testingPoint1);
            }

            auto testingPoint2 = testPlayer.GenMove(timeRecord, colorWhite);

            if (testingPoint2 != SG_NULLMOVE && testingPoint2 != SG_PASS){
                basicBoard.Play(testingPoint2);
            }

            if ((testingPoint1 == SG_NULLMOVE || testingPoint1 == SG_PASS)
               && (testingPoint2 == SG_NULLMOVE || testingPoint2 == SG_PASS)){
                break;
            }
            


            // SgDebug() << "testingPoint " << i << ": " << testingPoint << " . \n";

            // SgPoint testingPoint = SgPointUtil::Pt(10, 10);
            // SgBlackWhite color = SG_BLACK;

            

        }

        // printGoBoard(basicBoard);
    }
    

    double end_time;
    end_time = timer.GetTime();

    double time_used = end_time - start_time;

    SgDebug() << "Time used: " << time_used << ". \n";
    
    // // testingPoint = SgPointUtil::Pt(9, 9);

    // testingPoint = curPolicy.GenerateMove();

    // goBoard.Play(testingPoint);

    // printGoUctBoard(goBoard);
    
    SgDebug() << "End of the testing \n";


  // SpRandomPlayer testingPlayer();



  return 0;
}
