#ifndef SELFPLAY_MAIN_H
#define SELFPLAY_MAIN_H


#include "GoGtpEngine.h"
#include "GoSafetyCommands.h"
#include "GoUctBookBuilderCommands.h"
#include "GoUctCommands.h"
#include "GoUctFeatureCommands.h"

#include "SgSystem.h"

#include "GoBoard.h"

#include "SgDebug.h"
#include "mxnet-cpp/MxNetCpp.h"

#include "GoInit.h"
#include "SgInit.h"


typedef GoUctPlayer<GoUctGlobalSearch<GoUctPlayoutPolicy<GoUctBoard>,
                    GoUctPlayoutPolicyFactory<GoUctBoard> >,
                    GoUctGlobalSearchState<GoUctPlayoutPolicy<GoUctBoard> > >
    PlayerType;

#endif