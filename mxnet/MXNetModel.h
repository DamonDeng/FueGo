//----------------------------------------------------------------------------
/** @file MXNetModel.h */
//----------------------------------------------------------------------------

#ifndef MXNETMODEL_H
#define MXNETMODEL_H

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"

#include "SgDebug.h"
#include "SgTimer.h"
#include "SgSystem.h"
#include "SgRandom.h"

#include "GoBoard.h"
#include "GoUctBoard.h"

#include "SgUctValue.h"
#include "SgUctTree.h"
#include "GoUctSearch.h"

#include <boost/utility.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

using namespace std;
using namespace mxnet::cpp;

//----------------------------------------------------------------------------

/** GTP interface with commands for testing functionality of the Fuego
    libraries.
    @see @ref fuegotestoverview */
class MXNetModel
    
{
public:
    /** Constructor.
         */
    MXNetModel();
    
    MXNetModel(unsigned int threadId);

    

    ~MXNetModel();

    void ApplyPrioProbability(std::vector<SgUctMoveInfo>& moves, const GoBoard& bd);

    void GetPrioProbability(SgArray<SgUctValue, SG_MAX_MOVE_VALUE>& outputArray, SgUctValue& outputValue, const std::vector<float>& inputData);

private:

    map<string, NDArray> args_map;
    map<string, NDArray> aux_map;
    Symbol net;
    Executor *executor;

    // Context global_ctx(kCPU, 0);

    Context global_ctx;
    

    void LoadSymbol();
    void LoadParameters();
   
    
};

//----------------------------------------------------------------------------

#endif // MXNETMODEL_H

