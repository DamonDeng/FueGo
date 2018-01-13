//----------------------------------------------------------------------------
/** @file FuegoTestMain.cpp
    Main function for FuegoTest. */
//----------------------------------------------------------------------------

#include "SgSystem.h"

#include <iostream>
#include "FuegoTestEngine.h"
#include "GoInit.h"
#include "SgDebug.h"
#include "SgException.h"
#include "SgInit.h"
#include "GoBoard.h"
#include "GoUctBoard.h"
#include "GoUctPlayoutPolicy.h"

#include <boost/utility.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

using std::string;
namespace po = boost::program_options;

//----------------------------------------------------------------------------

namespace {

/** @name Settings from command line options */
// @{

bool g_quiet;

string g_config;

/** Player string as in FuegoTestEngine::SetPlayer */
string g_player;

const char* g_programPath;

// @} // @name

void MainLoop()
{
    FuegoTestEngine engine(0, g_programPath, g_player);
    GoGtpAssertionHandler assertionHandler(engine);
    if (g_config != "")
        engine.ExecuteFile(g_config);
    GtpInputStream in(std::cin);
    GtpOutputStream out(std::cout);
    engine.MainLoop(in, out);
}

void Help(po::options_description& desc)
{
    std::cout << "Options:\n" << desc << '\n';
    exit(1);
}

void ParseOptions(int argc, char** argv)
{
    int srand;
    po::options_description desc;
    desc.add_options()
        ("config", 
         po::value<std::string>(&g_config)->default_value(""),
         "execuate GTP commands from file before starting main command loop")
        ("help", "displays this help and exit")
        ("player", 
         po::value<std::string>(&g_player)->default_value(""),
         "player (average|ladder|liberty|maxeye|minlib|no-search|random|safe")
        ("quiet", "don't print debug messages")
        ("srand", 
         po::value<int>(&srand)->default_value(0),
         "set random seed (-1:none, 0:time(0))");
    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (...)
    {
        Help(desc);
    }
    if (vm.count("help"))
        Help(desc);
    if (vm.count("quiet"))
        g_quiet = true;
    if (vm.count("srand"))
        SgRandom::SetSeed(srand);
}

} // namespace

//----------------------------------------------------------------------------

// int main(int argc, char** argv)
// {
//     if (argc > 0 && argv != 0)
//     {
//         g_programPath = argv[0];
//         try
//         {
//             ParseOptions(argc, argv);
//         }
//         catch (const SgException& e)
//         {
//             SgDebug() << e.what() << "\n";
//             return 1;
//         }
//     }
//     if (g_quiet)
//         SgDebugToNull();
//     try
//     {
//         SgInit();
//         GoInit();
//         MainLoop();
//         GoFini();
//         SgFini();
//     }
//     catch (const GtpFailure& e)
//     {
//         SgDebug() << e.Response() << '\n';
//         return 1;
//     }
//     catch (const std::exception& e)
//     {
//         SgDebug() << e.what() << '\n';
//         return 1;
//     }
//     return 0;
// }

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

int main(int argc, char** argv)
{
    SgDebug() << "Starting the testing new. \n";

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

    for (int game_time = 0; game_time < 1000; game_time ++){

        GoBoard basicBoard;
        GoUctBoard goBoard(basicBoard);

        SgRandom sgRandom;
    
        GoUctPureRandomGenerator<GoUctBoard> randomGenerator(goBoard, sgRandom);
        randomGenerator.Start();
    
        for (int i =0; i< 1000; i++){
            // testingPoint = curPolicy.GenerateMove();
            testingPoint = randomGenerator.Generate();

            if (testingPoint == SG_NULLMOVE){
                break;
            }

            // SgDebug() << "testingPoint " << i << ": " << testingPoint << " . \n";

            // SgPoint testingPoint = SgPointUtil::Pt(10, 10);
            // SgBlackWhite color = SG_BLACK;

            goBoard.Play(testingPoint);

        }
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
    return 0;
}



//----------------------------------------------------------------------------

