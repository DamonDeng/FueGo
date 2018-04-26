#include "SelfplayMain.h"

#include "SgSystem.h"

#include "GoBoard.h"

#include "SgDebug.h"
#include "mxnet-cpp/MxNetCpp.h"

#include "GoInit.h"
#include "SgInit.h"


using namespace std;
using namespace mxnet::cpp;



int main(int argc, char** argv){

    SgDebug() << "SelfPlay testing. \n";

    SgInit();
    GoInit();

    GoBoard bd1(19);
    GoBoard bd2(19);

    PlayerType player1(bd1);
    PlayerType player2(bd2);

    player1.SetForcedOpeningMoves(false);
    player2.SetForcedOpeningMoves(false);
    

    SgTimeRecord time;

    for(int i=0; i<500; i++){

        SgDebug() << "Round : " << i << ". \n";

        time = SgTimeRecord(true, 5);

        SgPoint point1 = player1.GenMove(time, SG_BLACK);
        if(point1 == SG_RESIGN || point1 == SG_PASS){
            break;
        }
        bd1.Play(point1, SG_BLACK);
        bd2.Play(point1, SG_BLACK);

        time = SgTimeRecord(true, 5);
        SgPoint point2 = player2.GenMove(time, SG_WHITE);
        if(point2 == SG_RESIGN || point2 == SG_PASS){
            break;
        }

        bd1.Play(point2, SG_WHITE);
        bd2.Play(point2, SG_WHITE);
        
    }

    
    
}



