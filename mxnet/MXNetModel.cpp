
//----------------------------------------------------------------------------
/** @file MXNetModel.cpp
    See MXNetModel.h */
//----------------------------------------------------------------------------

#include "SgSystem.h"
#include "MXNetModel.h"

#include <boost/preprocessor/stringize.hpp>
#include <boost/algorithm/string.hpp>
#include "GoGtpCommandUtil.h"
#include "GoGtpExtraCommands.h"


using std::string;
using boost::trim_copy;

//----------------------------------------------------------------------------

// using namespace std;
// using namespace mxnet::cpp;

MXNetModel::MXNetModel():
global_ctx(kCPU, 0)
{
    // global_ctx = Context(kCPU, 0);
    SgDebug() << "Creating MXNetModel with default cpu: cpu 0. \n";
    // LoadSymbol();
    // LoadParameters();

    SgDebug() << "Successfully loaded the model. \n";
    
}


MXNetModel::MXNetModel(unsigned int threadId):
global_ctx(kCPU, threadId)
{
    // global_ctx = Context(kCPU, 0);
    SgDebug() << "Creating MXNetModel with cpu number: " << threadId << ". \n";
    LoadSymbol();
    LoadParameters();

    SgDebug() << "Successfully loaded the model. \n";
    
}

MXNetModel::~MXNetModel()
{ }

void MXNetModel::LoadSymbol() {

    // net = Symbol::Load("./model/zero_super_simple_cnn-symbol.json")
    //           .GetInternals()["softmax_output"];
    
    // net = Symbol::Load("./model/zero_resnet-symbol.json")
    //           .GetInternals()["softmax_output"];

    net = Symbol::Load("./model/new_zero_resnet-symbol.json")
    .GetInternals()["softmax_output"];


              
              
  }

void MXNetModel::LoadParameters() {
    map<string, NDArray> paramters;
    // NDArray::Load("./model/Inception-BN-0126.params", 0, &paramters);
    NDArray::Load("./model/new_zero_resnet-0005.params", 0, &paramters);
    // NDArray::Load("./model/zero_resnet-0003.params", 0, &paramters);
    
    // NDArray::Load("./model/zero_super_simple_cnn-0010.params", 0, &paramters);

    for (const auto &k : paramters) {
      if (k.first.substr(0, 4) == "aux:") {
        auto name = k.first.substr(4, k.first.size() - 4);
        aux_map[name] = k.second.Copy(global_ctx);
      }
      if (k.first.substr(0, 4) == "arg:") {
        auto name = k.first.substr(4, k.first.size() - 4);
        args_map[name] = k.second.Copy(global_ctx);
      }
    }
    /*WaitAll is need when we copy data between GPU and the main memory*/
    NDArray::WaitAll();
  }




void MXNetModel::ApplyPrioProbability(std::vector<SgUctMoveInfo>& moves, const GoBoard& bd)
{
    SgDebug() << "Trying to call CNN to generate prio probability. \n";

    int historyLength = 2;
    int arrayLength = historyLength*2 + 1;

    int boardSize = 19;

    int dataLength = 1*arrayLength*boardSize*boardSize;


    NDArray ret(Shape(1, arrayLength, boardSize, boardSize), global_ctx, false);

    std::vector<float> inputData(dataLength);

    bd.GetHistoryData(inputData, dataLength);
  

    ret.SyncCopyFromCPU(inputData.data(), dataLength);
    
    
    NDArray::WaitAll();
    
    // args_map["data"] = generateSampleData();

    // SgDebug() << "input data: \n";
    // for (int i=0; i<arrayLength; i++){
    //   for (int j=0; j<19; j++){
    //     for (int k=0; k<19; k++){
    //       int value_index = i*361 + j*19 + k;
    //       SgDebug() << ret.At(0,value_index);
    //     }
    //     SgDebug() << " \n";
    //   }
    //   SgDebug() << "\n \n";

    // }
    
    // SgDebug() << ". \n";
    
    args_map["data"] = ret;
    
    
    /*bind the excutor*/

    NDArray array;

    executor = net.SimpleBind(global_ctx, args_map, map<string, NDArray>(),
                              map<string, OpReqType>(), aux_map);

    executor->Forward(false);
      
    array = executor->outputs[0].Copy(global_ctx);
    NDArray::WaitAll();


    delete executor;
    
    // for (int i = 0; i < 362; ++i) {
    //   curValue = array.At(0, i);

    //   if (curValue > maxValue) {
    //     maxValue = curValue;
    //     maxPosition = i;
    //   }
    //   cout << array.At(0, i) << ",";
    // }
    
    // SgMove moveValue;
    SgGrid col;
    SgGrid row;

    int stoneNumber = boardSize*boardSize;

    for (size_t j = 0; j < moves.size(); ++j)
        {
            if(moves[j].m_move == SG_PASS){
                moves[j].m_prioProbability = array.At(0, stoneNumber);
            } else{
                row = SgPointUtil::Row(moves[j].m_move)-1;
                col = SgPointUtil::Col(moves[j].m_move)-1;

                moves[j].m_prioProbability = array.At(0, row*boardSize + col);

            }

            
        }

}

//----------------------------------------------------------------------------
