
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

int MXNetModel::m_gpuNumber = 0;

MXNetModel::MXNetModel():
global_ctx(kCPU, 0),
current_ctx(kCPU, 0)
{
    if (MXNetModel::m_gpuNumber > 0){
        SgDebug() << "Using the gpu(0) as device. \n";
        global_ctx = Context(kGPU, 0);
    } else {

        SgDebug() << "Creating MXNetModel with default cpu: cpu 0. \n";
    }
    
    LoadSymbol();
    LoadParameters();

    SgDebug() << "Successfully loaded the model. \n";
    
}


MXNetModel::MXNetModel( unsigned int threadId):
global_ctx(kCPU, threadId),
current_ctx(kCPU, threadId)
{
    if (MXNetModel::m_gpuNumber > 0){
        int gpu_id = threadId%MXNetModel::m_gpuNumber;
        SgDebug() << "Using the gpu(" << gpu_id << ") as device. \n";
        global_ctx = Context(kGPU, gpu_id);
    } else {

        SgDebug() << "Creating MXNetModel with cpu number: " << threadId << ". \n";
    }
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

    // net = Symbol::Load("./model/new_zero_resnet-symbol.json")
    // .GetInternals()["softmax_output"];

    // net = Symbol::Load("./model/zeronew_zero_19resnet_l17-symbol.json")
    // .GetInternals()["softmax_output"];


    // Symbol temp_net = Symbol::Load("./model/thetago_dual_res_17L_19res-symbol.json");

    // temp_net.list_outputs();


    // net = Symbol::Load("./model/thetago_dual_res_17L_19res-symbol.json");
    // .GetInternals()["move_output"];
    
    // net = Symbol::Load("./model/thetago_dual_res_17L_19res_rate_0p02-symbol.json");
    
    // net = Symbol::Load("./model/thetago_dual_res_17L_19res_scalevalue_rate_0p02-symbol.json");
    

    // net = Symbol::Load("./model/thetago_dual_res_17L_19res_rate_0p002-symbol.json");

    // net = Symbol::Load("./model/thetago_dual_res_17L_19res_rate_0p001_advance-symbol.json");

    net = Symbol::Load("./model/thetago_dual_res_17L_19res_0p001_advanced_no_color-symbol.json");

    
    
            
              
  }

void MXNetModel::LoadParameters() {
    map<string, NDArray> paramters;
    // NDArray::Load("./model/Inception-BN-0126.params", 0, &paramters);
    // NDArray::Load("./model/new_zero_resnet-0003.params", 0, &paramters);
    // NDArray::Load("./model/new_zero_resnet-0005.params", 0, &paramters);
    
    // NDArray::Load("./model/zero_resnet-0003.params", 0, &paramters);
    
    // NDArray::Load("./model/zero_super_simple_cnn-0010.params", 0, &paramters);

    // NDArray::Load("./model/zeronew_zero_19resnet_l17-0003.params", 0, &paramters);

    // NDArray::Load("./model/zeronew_zero_19resnet_l17-0007.params", 0, &paramters);

    // NDArray::Load("./model/thetago_dual_res_17L_19res-0001.params", 0, &paramters);

    // NDArray::Load("./model/thetago_dual_res_17L_19res_rate_0p02-0001.params", 0, &paramters);

    // NDArray::Load("./model/thetago_dual_res_17L_19res_scalevalue_rate_0p02-0001.params", 0, &paramters);
 
    // NDArray::Load("./model/thetago_dual_res_17L_19res_rate_0p002-0003.params", 0, &paramters);

    // NDArray::Load("./model/thetago_dual_res_17L_19res_rate_0p001_advance-0004.params", 0, &paramters);


    // NDArray::Load("./model/thetago_dual_res_17L_19res_0p001_advanced_no_color-0001.params", 0, &paramters);
    
    // NDArray::Load("./model/thetago_dual_res_17L_19res_0p001_advanced_no_color-0006.params", 0, &paramters);
    
    NDArray::Load("./model/thetago_dual_res_17L_19res_0p001_advanced_no_color-0009.params", 0, &paramters);
    

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




void MXNetModel::ApplyPrioProbability(std::vector<SgUctMoveInfo>& moves, SgUctValue& value, const std::vector<float>& inputData)
{
    // SgDebug() << "Trying to call CNN to apply prio probability. \n";

    int historyLength = 8;
    int arrayLength = historyLength*2 + 1;

    int boardSize = 19;

    int dataLength = arrayLength*boardSize*boardSize;

    int batchSize = 1;

    // std::vector<float> batchHistoryData(batchSize * dataLength);

    // for (int i=0; i<batchSize; i++){
    //     for (int j=0; j<dataLength; j++){
    //         batchHistoryData[i*j] = inputData[j];
    //     }
    // }


    NDArray ret(Shape(batchSize, arrayLength, boardSize, boardSize), global_ctx, false);

    // std::vector<float> inputData(dataLength);

    // bd.GetHistoryData(inputData, dataLength);
  

    // ret.SyncCopyFromCPU(inputData.data(), dataLength);
    ret.SyncCopyFromCPU(inputData.data(), batchSize*dataLength);
    
    
    NDArray::WaitAll();
    
    
    args_map["data"] = ret;
    
    
    /*bind the excutor*/

    NDArray array;

    NDArray move_value;

    executor = net.SimpleBind(global_ctx, args_map, map<string, NDArray>(),
                              map<string, OpReqType>(), aux_map);

   


    executor->Forward(false);

   
    // Context current_cpu(kCPU, 0);
      
    array = executor->outputs[0].Copy(current_ctx);
    move_value = executor->outputs[1].Copy(current_ctx);

    NDArray::WaitAll();

    // SgDebug() << "----------------------.\n";
    // SgDebug() << move_value.At(0,0) << "\n";
    // SgDebug() << "-----------------------------.\n";


    value = move_value.At(0,0);
   

    // trying to get top list of the array value
    // topList array to record to top topMemberNumber of value.

    int topMemberNumber = 20;

    std::vector<SgUctValue> topList(topMemberNumber);
    SgUctValue swapValue = 0;


    
    // SgMove moveValue;
    SgGrid col;
    SgGrid row;

    int stoneNumber = boardSize*boardSize;

    for (size_t j = 0; j < moves.size(); ++j)
        {
            if(moves[j].m_move == SG_PASS){
                moves[j].m_prioProbability = array.At(0, stoneNumber);
                moves[j].m_hasPrioProbability = true;
                
                // moves[j].m_value = value;
                // moves[j].m_count = 1;
            } else{
                row = SgPointUtil::Row(moves[j].m_move)-1;
                col = SgPointUtil::Col(moves[j].m_move)-1;

                moves[j].m_prioProbability = array.At(0, row*boardSize + col);
                moves[j].m_hasPrioProbability = true;

                // moves[j].m_value = value;
                // moves[j].m_count = 1;

            }

            // trying to get top list of the array value

            if (moves[j].m_prioProbability > topList[0]){
                topList[0] = moves[j].m_prioProbability;
                for (int j=0; j<(topMemberNumber-1); j++){
                    if (topList[j] > topList[j+1]){
                        swapValue = topList[j+1];
                        topList[j+1] = topList[j];
                        topList[j] = swapValue;
                    } else {
                        break;
                    }
                }
            }

            
        }

    delete executor;

    //@dangerous, passing reference of local variable out of function.

    std::vector<SgUctMoveInfo> filteredMoves;
    for (std::vector<SgUctMoveInfo>::const_iterator it = moves.begin();it != moves.end(); ++it){
        if (it->m_prioProbability >= topList[0]){
            filteredMoves.push_back(*it);
        }

    }

    
    moves = filteredMoves;


}

void MXNetModel::GetPrioProbability(SgArray<SgUctValue, SG_MAX_MOVE_VALUE>& outputArray, SgUctValue& outputValue, const std::vector<float>& inputData){

    // SgDebug() << "Trying to call CNN to get prio probability. \n";

    int historyLength = 8;
    int arrayLength = historyLength*2 + 1;

    int boardSize = 19;

    int dataLength = arrayLength*boardSize*boardSize;

    int batchSize = 1;

    // std::vector<float> batchHistoryData(batchSize * dataLength);

    // for (int i=0; i<batchSize; i++){
    //     for (int j=0; j<dataLength; j++){
    //         batchHistoryData[i*j] = inputData[j];
    //     }
    // }


    NDArray ret(Shape(batchSize, arrayLength, boardSize, boardSize), global_ctx, false);

    // std::vector<float> inputData(dataLength);

    // bd.GetHistoryData(inputData, dataLength);
  

    // ret.SyncCopyFromCPU(inputData.data(), dataLength);
    ret.SyncCopyFromCPU(inputData.data(), batchSize*dataLength);
    
    
    NDArray::WaitAll();
    
    
    args_map["data"] = ret;
    
    
    /*bind the excutor*/

    NDArray array;

    NDArray move_value;

    executor = net.SimpleBind(global_ctx, args_map, map<string, NDArray>(),
                              map<string, OpReqType>(), aux_map);

   


    executor->Forward(false);

   
    // Context current_cpu(kCPU, 0);
      
    array = executor->outputs[0].Copy(current_ctx);
    move_value = executor->outputs[1].Copy(current_ctx);

    NDArray::WaitAll();

    // SgDebug() << "----------------------.\n";
    // SgDebug() << move_value.At(0,0) << "\n";
    // SgDebug() << "-----------------------------.\n";


    outputValue = move_value.At(0,0);
   

    int stoneNumber = boardSize*boardSize;
   


    outputArray[SG_MAX_MOVE_VALUE-1] = array.At(0, stoneNumber);

    SgPoint point;

    for (int row = 1; row <= boardSize; row++){

        for (int col = 1; col <= boardSize; col++){

            point = SgPointUtil::Pt(col, row);

            outputArray[point] = array.At(0, (row-1)*boardSize + (col-1));
        }

        
    }

    delete executor;



   

    // Debug code

    // int boardSize = 19;

    // SgPoint point;

    // for (int row = 1; row <= boardSize; row++){

    //     for (int col = 1; col <= boardSize; col++){

    //         point = SgPointUtil::Pt(col, row);

    //         outputArray[point] = 10;
    //     }

        
    // }

    // outputArray[10] = 10;




}


//----------------------------------------------------------------------------
