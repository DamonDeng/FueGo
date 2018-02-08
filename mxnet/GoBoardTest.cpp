#include "SgSystem.h"

#include "GoBoard.h"

#include "SgDebug.h"
#include "mxnet-cpp/MxNetCpp.h"

#include "GoInit.h"
#include "SgInit.h"


using namespace std;
using namespace mxnet::cpp;


Context global_ctx(kCPU, 0);

void printNDArray(NDArray& ret);

int main(int argc, char** argv){

    SgDebug() << "GoBard Test Started. \n";

    SgInit();
    GoInit();

    GoBoard testingBoard(19);

    NDArray ret(Shape(1, 5, 19, 19), global_ctx, false);

    std::vector<float> historyData(1*5*19*19);

    SgPoint targetPoint = SgPointUtil::Pt(1,1);

    testingBoard.Play(targetPoint, SG_BLACK);


    testingBoard.GetHistoryData(historyData, 1*5*19*19);

    // SgDebug() << "Checking the data reference:\n";

    // for (int i=0; i<100; i++){
    //     SgDebug() << *historyArray;
    //     historyArray++;
    // }

    // SgDebug() << "end of the debug data.\n";
  
    ret.SyncCopyFromCPU(historyData.data(), 1 * 5 * 19 * 19);


    
    
    printNDArray(ret);


     targetPoint = SgPointUtil::Pt(16,16);

    testingBoard.Play(targetPoint, SG_WHITE);

    

    testingBoard.GetHistoryData(historyData, 1*5*19*19);
  
    ret.SyncCopyFromCPU(historyData.data(), 1 * 5 * 19 * 19);


    
    
    printNDArray(ret);

    
    
}


void printNDArray(NDArray& ret){
    SgDebug() << "input data: \n";
    for (int i=0; i<5; i++){
      for (int j=0; j<19; j++){
        for (int k=0; k<19; k++){
          int value_index = i*361 + j*19 + k;
          SgDebug() << ret.At(0,value_index) << " ";
        }
        SgDebug() << " \n";
      }
      SgDebug() << "\n \n";

    }

}
