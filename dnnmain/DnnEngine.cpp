//----------------------------------------------------------------------------
/** @file DnnEngine.cpp
    See DnnEngine.h */
//----------------------------------------------------------------------------

#include "SgSystem.h"
#include "DnnEngine.h"

#include <boost/preprocessor/stringize.hpp>
#include <boost/algorithm/string.hpp>
#include "GoGtpCommandUtil.h"
#include "GoGtpExtraCommands.h"
#include "SpAveragePlayer.h"
#include "SpCapturePlayer.h"
#include "SpDumbTacticalPlayer.h"
#include "SpGreedyPlayer.h"
#include "SpInfluencePlayer.h"
#include "SpLadderPlayer.h"
#include "SpLibertyPlayer.h"
#include "SpMaxEyePlayer.h"
#include "SpMinLibPlayer.h"
#include "SpRandomPlayer.h"
#include "SpSafePlayer.h"

using std::string;
using boost::trim_copy;

//----------------------------------------------------------------------------

DnnEngine::DnnEngine(int fixedBoardSize, const char* programPath)
    : GoGtpEngine(fixedBoardSize, programPath),
      m_extraCommands(Board()),
      m_safetyCommands(Board())
{
    Register("fuegotest_param", &DnnEngine::CmdParam, this);
    m_extraCommands.Register(*this);
    m_safetyCommands.Register(*this);
    SetPlayer();
}

DnnEngine::~DnnEngine()
{ }

void DnnEngine::CmdAnalyzeCommands(GtpCommand& cmd)
{
    GoGtpEngine::CmdAnalyzeCommands(cmd);
    m_extraCommands.AddGoGuiAnalyzeCommands(cmd);
    m_safetyCommands.AddGoGuiAnalyzeCommands(cmd);
    cmd <<
        "param/FuegoTest Param/fuegotest_param\n";
    string response = cmd.Response();
    cmd.SetResponse(GoGtpCommandUtil::SortResponseAnalyzeCommands(response));
}

void DnnEngine::CmdName(GtpCommand& cmd)
{
    if (m_playerId == "")
        cmd << "FuegoTest";
    else
        GoGtpEngine::CmdName(cmd);
}

/** Player selection.
    This command is compatible with the GoGui analyze command type "param".

    Parameters:
    @arg @c player Player id as in DnnEngine::SetPlayer */
void DnnEngine::CmdParam(GtpCommand& cmd)
{
    cmd.CheckNuArgLessEqual(2);
    if (cmd.NuArg() == 0)
    {
        cmd <<
            "[list/<none>/average/capture/dumbtactic/greedy/influence/"
            "ladder/liberty/maxeye/minlib/no-search/random/safe] player "
            << (m_playerId == "" ? "<none>" : m_playerId) << '\n';
    }
    else if (cmd.NuArg() >= 1 && cmd.NuArg() <= 2)
    {
        string name = cmd.Arg(0);
        if (name == "player")
        {
            try
            {
                // string id = trim_copy(cmd.RemainingLine(0));
                // if (id == "<none>")
                //     id = "";
                SetPlayer();
            }
            catch (const SgException& e)
            {
                throw GtpFailure(e.what());
            }
        }
        else
            throw GtpFailure() << "unknown parameter: " << name;
    }
    else
        throw GtpFailure() << "need 0 or 2 arguments";
}

void DnnEngine::CmdVersion(GtpCommand& cmd)
{
#ifdef VERSION
    cmd << BOOST_PP_STRINGIZE(VERSION);
#else
    cmd << "(" __DATE__ ")";
#endif
#ifndef NDEBUG
    cmd << " (dbg)";
#endif
}

GoPlayer* DnnEngine::CreatePlayer(const string& playerId)
{
    const GoBoard& bd = Board();
    if (playerId == "")
        return 0;
    if (playerId == "average")
        return new SpAveragePlayer(bd);
    if (playerId == "capture")
        return new SpCapturePlayer(bd);
    if (playerId == "dumbtactic")
        return new SpDumbTacticalPlayer(bd);
    if (playerId == "greedy")
        return new SpGreedyPlayer(bd);
    if (playerId == "influence")
        return new SpInfluencePlayer(bd);
    if (playerId == "ladder")
        return new SpLadderPlayer(bd);
    if (playerId == "liberty")
        return new SpLibertyPlayer(bd);
    if (playerId == "maxeye")
        return new SpMaxEyePlayer(bd, true);
    if (playerId == "minlib")
        return new SpMinLibPlayer(bd);
    if (playerId == "random")
        return new SpRandomPlayer(bd);
    if (playerId == "safe")
        return new SpSafePlayer(bd);
    throw SgException("unknown player " + playerId);
}

void DnnEngine::SetPlayer()
{
    // SgDebug() << "Going to create player:, player id: " << playerId << "\n";
    // GoPlayer* player = CreatePlayer(playerId);
    // GoGtpEngine::SetPlayer(player);
    // m_playerId = playerId;

    // Going to use UctPlayer directly.

    // SgDebug() << "Got player id: " << playerId << ". But we wouldn't use it. \n";

    GoUctPlayer<GoUctGlobalSearch<GoUctPlayoutPolicy<GoUctBoard>,
                    GoUctPlayoutPolicyFactory<GoUctBoard> >,
                    GoUctGlobalSearchState<GoUctPlayoutPolicy<GoUctBoard> > > *uctPlayer = new GoUctPlayer<GoUctGlobalSearch<GoUctPlayoutPolicy<GoUctBoard>,
                    GoUctPlayoutPolicyFactory<GoUctBoard> >,
                    GoUctGlobalSearchState<GoUctPlayoutPolicy<GoUctBoard> > >(Board());

    GoGtpEngine::SetPlayer(uctPlayer);
}

//----------------------------------------------------------------------------
