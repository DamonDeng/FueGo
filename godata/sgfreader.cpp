#include "sgfreader.h"

#include <fstream>

#include "SgInit.h"
#include "GoInit.h"

#include "SgSystem.h"

#include "GoBoard.h"

#include "SgDebug.h"
#include "mxnet-cpp/MxNetCpp.h"

#include "GoInit.h"
#include "SgInit.h"

#include "SgGameReader.h"
#include "SgGameWriter.h"

#include "GoGame.h"


using namespace std;
using namespace mxnet::cpp;



int main(int argc, char** argv){

    SgInit();
    GoInit();

    SgDebug() << "Starting SGF reader. \n";

    std::string inFileName("./godata/kgs_sgf/2000-7-19-2.sgf");
    std::string outFileName("./godata/kgs_sgf/convert_2000-7-19-2.sgf");

    GoGame testingGame(19);

    std::ifstream in(inFileName);

    if (! in){

        SgDebug() << "could not open file. \n";
        return 1;
    }

    SgGameReader reader(in);
    SgNode* root = reader.ReadGame();

    if (root == 0){

        SgDebug() << "no games in file. \n";
        return 1;
    }

    testingGame.Init(root);

    GoBoard testingBoard(19);

    SgNode* current = root->NodeInDirection(SgNode::NEXT);

    while (!current->IsTerminal()){
        testingBoard.Play(current->NodeMove());
        SgDebugGotoXY(0, 0);
        SgDebug() << testingBoard;
        current = current->NodeInDirection(SgNode::NEXT);
    }

    // std::ofstream out(outFileName);
    // SgGameWriter writer(out);
    // writer.WriteGame(testingGame.Root(), true, 0, 1, 19);




    
}