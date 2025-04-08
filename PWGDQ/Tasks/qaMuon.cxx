// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
/// \file muonDCA.cxx
/// \brief Task to compute and evaluate DCA quantities
/// \author Nicolas Bizé <nicolas.bize@cern.ch>, SUBATECH
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/ASoAHelpers.h"
#include "GlobalTracking/MatchGlobalFwd.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "Common/DataModel/EventSelection.h"
#include "MFTTracking/Constants.h"
#include "PWGDQ/DataModel/ReducedInfoTables.h"
#include "PWGDQ/Core/VarManager.h"

#include <string>
#include <unordered_map>
#include <map>
#include <limits>

using namespace o2;
using namespace o2::framework;
using namespace o2::aod;

using MyReducedMuons = soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtra, aod::ReducedMuonsCov>;
using MyReducedEvents = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended>;
using MyReducedEventsVtxCov = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>;

using MyCollisions = aod::Collisions;
using MyBCs = soa::Join<aod::BCs, aod::Timestamps>;
using MyEvents = soa::Join<aod::Collisions, aod::EvSels>;
using MyMuonsWithCov = soa::Join<aod::FwdTracks, aod::FwdTracksCov>;
//using MyMuonsWithCov = aod::FwdTracks;
using MyMFTs = aod::MFTTracks;

using MyCollision = MyCollisions::iterator;
using MyBC = MyBCs::iterator;
using MyMUON = MyMuonsWithCov::iterator;
using MyMFT = MyMFTs::iterator;

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

std::unordered_map<int, std::vector<int64_t>> map_mfttracks;
std::unordered_map<int, std::vector<int64_t>> map_muontracks;
std::unordered_map<int, bool> map_collisions;
std::unordered_map<int, bool> map_has_mfttracks_collisions;
std::unordered_map<int, bool> map_has_muontracks_collisions;
std::unordered_map<int, float> map_vtxz;
std::unordered_map<int, int> map_nmfttrack;

constexpr double muonMass = 0.1056584;
constexpr double muonMass2 = muonMass * muonMass;

// constexpr static uint32_t gkMuonDCAFillMapWithCov = VarManager::ObjTypes::ReducedMuon | VarManager::ObjTypes::ReducedMuonExtra | VarManager::ObjTypes::ReducedMuonCov | VarManager::ObjTypes::MuonDCA;

constexpr static int toVertex = 0; //VarManager::kToVertex;
constexpr static int toDCA = 1; //VarManager::kToDCA;
constexpr static int toRabs = 2; //VarManager::kToRabs;
constexpr static int toMFT = 3;

constexpr double firstMFTPlaneZ = o2::mft::constants::mft::LayerZCoordinate()[0];
constexpr double lastMFTPlaneZ = o2::mft::constants::mft::LayerZCoordinate()[9];

static o2::globaltracking::MatchGlobalFwd sExtrap;

template <typename T>
bool isSelected(const T& muon);

static double getDeltaPhi(double phi1, double phi2)
{
  double dPhi1 = phi1 - phi2;
  double dPhi2 = phi1 - phi2 + 360.;
  double dPhi3 = phi1 - phi2 - 360.;
  double dPhi = (std::fabs(dPhi1) < std::fabs(dPhi2)) ?
      ((std::fabs(dPhi1) < std::fabs(dPhi3)) ? dPhi1 : dPhi3) :
      ((std::fabs(dPhi2) < std::fabs(dPhi3)) ? dPhi2 : dPhi3);
  //std::cout << std::format("phi1={}  phi2={}  dphi={},{},{} ==> {}",
  //    phi1, phi2, dPhi1, dPhi2, dPhi3, dPhi) << std::endl;
  return dPhi;

}

struct qaMuon {
  ////   Variables for selecting muon tracks
  Configurable<float> fPMchLow{"cfgPMchLow", 0.0f, ""};
  Configurable<float> fPtMchLow{"cfgPtMchLow", 0.7f, ""};
  Configurable<float> fEtaMchLow{"cfgEtaMchLow", -4.0f, ""};
  Configurable<float> fEtaMchUp{"cfgEtaMchUp", -2.5f, ""};
  Configurable<float> fRabsLow{"cfgRabsLow", 17.6f, ""};
  Configurable<float> fRabsUp{"cfgRabsUp", 89.5f, ""};
  Configurable<float> fSigmaPdcaUp{"cfgPdcaUp", 6.f, ""};
  Configurable<float> fTrackChi2MchUp{"cfgTrackChi2MchUp", 5.f, ""};
  Configurable<float> fMatchingChi2MchMidUp{"cfgMatchingChi2MchMidUp", 999.f, ""};

  ////   Variables for selecting mft tracks
  Configurable<float> fEtaMftLow{"cfgEtaMftlow", -3.6f, ""};
  Configurable<float> fEtaMftUp{"cfgEtaMftup", -2.5f, ""};
  Configurable<int> fTrackNClustMftLow{"cfgTrackNClustMftLow", 7, ""};
  Configurable<float> fTrackChi2MftUp{"cfgTrackChi2MftUp", 999.f, ""};

  ////   Variables for selecting global tracks
  Configurable<float> fMatchingChi2MftMchUp{"cfgMatchingChi2MftMchUp", 50.f, ""};

  ///    Variables to event mixing criteria
  Configurable<float> fSaveMixedMatchingParamsRate{"cfgSaveMixedMatchingParamsRate", 0.002f, ""};
  Configurable<int> fEventMaxDeltaNMFT{"cfgEventMaxDeltaNMFT", 1, ""};
  Configurable<float> fEventMaxDeltaVtxZ{"cfgEventMaxDeltaVtxZ", 1.f, ""};
  Configurable<int> fEventMinDeltaBc{"cfgEventMinDeltaBc", 500, ""};

  ////   Variables for ccdb
  Configurable<std::string> ccdburl{"ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
  Configurable<std::string> grpPath{"grpPath", "GLO/GRP/GRP", "Path of the grp file"};
  Configurable<std::string> grpmagPath{"grpmagPath", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
  Configurable<std::string> geoPath{"geoPath", "GLO/Config/GeometryAligned", "Path of the geometry file"};

  ///    Variables for histograms configuration
  Configurable<int> fNCandidatesMax{"nCandidatesMax", 5, ""};

  o2::parameters::GRPMagField* grpmag = nullptr; // for run 3, we access GRPMagField from GLO/Config/GRPMagField
  int mRunNumber{0};                               // needed to detect if the run changed and trigger update of magnetic field

  Service<o2::ccdb::BasicCCDBManager> ccdbManager;
  o2::field::MagneticField* fieldB;
  o2::ccdb::CcdbApi ccdbApi;

  HistogramRegistry registry{"registry", {}};

  std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 4>, 2> dcaHistos;
  std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 4>, 2> dcaHistosMixedEvents;

  std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 4>, 4> alignmentHistos;
  std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 4>, 4> alignmentHistosMixedEvents;

  // vector of all MFT-MCH(-MID) matching candidates associated to the same MCH(-MID) track,
  // to be sorted in descending order with respect to the matching quality
  // the map key is the MCH(-MID) track global index
  using MatchingCandidates = std::map<uint64_t, std::vector<uint64_t>>;

  struct CollisionInfo
  {
    uint64_t bc{0};
    // z position of the collision
    double zVertex{0};
    // number of MFT tracks associated to the collision
    int mftTracksMultiplicity{0};
    // vector of MFT track indexes
    std::vector<uint64_t> mftTracks;
    // vector of MCH(-MID) track indexes
    std::vector<uint64_t> mchTracks;
    // matching candidates
    std::map<uint64_t, std::vector<uint64_t>> globalMuonTracks;
  };

  void InitCollisions(MyEvents const& collisions,
                      aod::BCsWithTimestamps const& bcs,
                      MyMuonsWithCov const& muonTracks,
                      std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    // fill collision information for global muon tracks (MFT-MCH-MID matches)
    for (auto muonTrack : muonTracks) {
      if (!muonTrack.has_collision())
        continue;

      auto collision = collisions.rawIteratorAt(muonTrack.collisionId());
      uint64_t collisionIndex = collision.globalIndex();

      auto bc = bcs.rawIteratorAt(collision.bcId());

      auto& collisionInfo = collisionInfos[collisionIndex];
      collisionInfo.bc = bc.globalBC();
      collisionInfo.zVertex = collision.posZ();

      if (static_cast<int>(muonTrack.trackType()) > 2) {
        // standalone MCH or MCH-MID tracks
        uint64_t mchTrackIndex = muonTrack.globalIndex();
        collisionInfo.mchTracks.push_back(mchTrackIndex);
      } else {
        // global muon tracks (MFT-MCH or MFT-MCH-MID)
        uint64_t muonTrackIndex = muonTrack.globalIndex();
        auto const& mchTrack = muonTrack.template matchMCHTrack_as<MyMuonsWithCov>();
        uint64_t mchTrackIndex = mchTrack.globalIndex();

        // check if a vector of global muon candidates is already available for the current MCH index
        // if not, initialize a new one and add the current global muon track
        //bool globalMuonTrackFound = false;
        auto matchingCandidateIterator = collisionInfo.globalMuonTracks.find(mchTrackIndex);
        if (matchingCandidateIterator != collisionInfo.globalMuonTracks.end()) {
          matchingCandidateIterator->second.push_back(muonTrackIndex);
          //globalMuonTrackFound = true;
        } else {
          collisionInfo.globalMuonTracks[mchTrackIndex].push_back(muonTrackIndex);
        }
      }
    }

    // sort the vectors of matching candidates in ascending order based on the matching chi2 value
    auto compareChi2 = [&muonTracks](uint64_t trackIndex1, uint64_t trackIndex2) -> bool {
      auto const& track1 = muonTracks.rawIteratorAt(trackIndex1);
      auto const& track2 = muonTracks.rawIteratorAt(trackIndex2);

      return (track1.chi2MatchMCHMFT() < track2.chi2MatchMCHMFT());
    };

    for (auto& [collisionIndex, collisionInfo] : collisionInfos) {
      for (auto& [mchIndex, globalTracksVector] : collisionInfo.globalMuonTracks) {
        std::sort(globalTracksVector.begin(), globalTracksVector.end(), compareChi2);
      }
    }
  }

  void InitCollisions(MyEvents const& collisions,
                      aod::BCsWithTimestamps const& bcs,
                      MyMuonsWithCov const& muonTracks,
                      MyMFTs const& mftTracks,
                      std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    InitCollisions(collisions, bcs, muonTracks, collisionInfos);

    // fill collision information for MFT standalone tracks
    for (auto mftTrack : mftTracks) {
      if (!mftTrack.has_collision())
        continue;

      auto collision = collisions.rawIteratorAt(mftTrack.collisionId());
      uint64_t collisionIndex = collision.globalIndex();

      auto bc = bcs.rawIteratorAt(collision.bcId());

      uint64_t mftTrackIndex = mftTrack.globalIndex();

      auto& collisionInfo = collisionInfos[collisionIndex];
      collisionInfo.bc = bc.globalBC();
      collisionInfo.zVertex = collision.posZ();

      collisionInfo.mftTracks.push_back(mftTrackIndex);
    }
  }

  template <typename BC>
  void initCCDB(BC const& bc)
  {
    if (mRunNumber == bc.runNumber())
      return;

    mRunNumber = bc.runNumber();
    std::map<string, string> metadata;
    auto soreor = o2::ccdb::BasicCCDBManager::getRunDuration(ccdbApi, mRunNumber);
    auto ts = soreor.first;
    auto grpmag = ccdbApi.retrieveFromTFileAny<o2::parameters::GRPMagField>(grpmagPath, metadata, ts);
    o2::base::Propagator::initFieldFromGRP(grpmag);
    if (!o2::base::GeometryManager::isGeometryLoaded()) {
      ccdbManager->get<TGeoManager>(geoPath);
    }
    o2::mch::TrackExtrap::setField();
    fieldB = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    //std::cout << "fieldB: " << (void*)fieldB << std::endl;
  }

  void CreateAlignementHistos()
  {
    std::vector<std::pair<std::string, double>> referencePlanes = {
        {"MFT-begin", 15.0},
        {"MFT-end", 20.0},
        {"absorber-mid", 75.0},
        {"absorber-end", 100.0}
    };

    std::array<std::string, 4> quadrants = {"Q0", "Q1", "Q2", "Q3"};

    AxisSpec dcaxMFTAxis = {400, -1.0, 1.0, "DCA_{x}"};
    AxisSpec dcayMFTAxis = {400, -1.0, 1.0, "DCA_{y}"};
    AxisSpec dcaxMCHAxis = {400, -10.0, 10.0, "DCA_{x}"};
    AxisSpec dcayMCHAxis = {400, -10.0, 10.0, "DCA_{y}"};
    AxisSpec dxAxis = {600, -30.0, 30.0, "#Delta x (cm)"};
    AxisSpec dyAxis = {600, -30.0, 30.0, "#Delta y (cm)"};
    AxisSpec thetaxAxis = {10, 0.0, 20.0, "#theta_{x} (degrees)"};
    AxisSpec dThetaxAxis = {500, -5.0, 5.0, "#Delta#theta_{x} (degrees)"};
    AxisSpec thetayAxis = {10, 0.0, 20.0, "#theta_{y} (degrees)"};
    AxisSpec dThetayAxis = {500, -5.0, 5.0, "#Delta#theta_{y} (degrees)"};
    AxisSpec phiAxis = {360, -180.0, 180.0, "#phi (degrees)"};
    AxisSpec dPhiAxis = {200, -20.0, 20.0, "#Delta#phi (degrees)"};

    for (size_t j = 0; j < quadrants.size(); j++) {
      const auto& quadrant = quadrants[j];
      std::string histPath = std::string("alignment/DCA/MFT/") + quadrant + "/";
      dcaHistos[0][j]["DCA_x"] = registry.add((histPath + "DCA_x").c_str(), std::format("DCA(x) - {}", quadrant).c_str(), {HistType::kTH1F, {dcaxMFTAxis}});
      dcaHistos[0][j]["DCA_y"] = registry.add((histPath + "DCA_y").c_str(), std::format("DCA(y) - {}", quadrant).c_str(), {HistType::kTH1F, {dcayMFTAxis}});

      histPath = std::string("alignment/DCA/MCH/") + quadrant + "/";
      dcaHistos[1][j]["DCA_x"] = registry.add((histPath + "DCA_x").c_str(), std::format("DCA(x) - {}", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistos[1][j]["DCA_y"] = registry.add((histPath + "DCA_y").c_str(), std::format("DCA(y) - {}", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});

      histPath = std::string("alignment/mixed-events/DCA/MFT/") + quadrant + "/";
      dcaHistosMixedEvents[0][j]["DCA_x"] = registry.add((histPath + "DCA_x").c_str(), std::format("DCA(x) - {}", quadrant).c_str(), {HistType::kTH1F, {dcaxMFTAxis}});
      dcaHistosMixedEvents[0][j]["DCA_y"] = registry.add((histPath + "DCA_y").c_str(), std::format("DCA(y) - {}", quadrant).c_str(), {HistType::kTH1F, {dcayMFTAxis}});

      histPath = std::string("alignment/mixed-events/DCA/MCH/") + quadrant + "/";
      dcaHistosMixedEvents[1][j]["DCA_x"] = registry.add((histPath + "DCA_x").c_str(), std::format("DCA(x) - {}", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistosMixedEvents[1][j]["DCA_y"] = registry.add((histPath + "DCA_y").c_str(), std::format("DCA(y) - {}", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});
    }

    for (size_t i = 0; i < referencePlanes.size(); i++) {
      const auto& refPLane = referencePlanes[i];
      AxisSpec xAxis = {10, 0, refPLane.second, "|x| (cm)"};
      AxisSpec yAxis = {10, 0, refPLane.second, "|y| (cm)"};
      for (size_t j = 0; j < quadrants.size(); j++) {
        const auto& quadrant = quadrants[j];
        std::string histPath = std::string("alignment/") + refPLane.first + "/" + quadrant + "/";
        alignmentHistos[i][j]["dx_vs_x"] = registry.add((histPath + "dx_vs_x").c_str(), std::format("#Delta x vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dxAxis}});
        alignmentHistos[i][j]["dx_vs_y"] = registry.add((histPath + "dx_vs_y").c_str(), std::format("#Delta x vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dxAxis}});
        alignmentHistos[i][j]["dy_vs_x"] = registry.add((histPath + "dy_vs_x").c_str(), std::format("#Delta y vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dyAxis}});
        alignmentHistos[i][j]["dy_vs_y"] = registry.add((histPath + "dy_vs_y").c_str(), std::format("#Delta y vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dyAxis}});

        alignmentHistos[i][j]["dthetax_vs_x"] = registry.add((histPath + "dthetax_vs_x").c_str(), std::format("#Delta #theta_x vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dThetaxAxis}});
        alignmentHistos[i][j]["dthetax_vs_y"] = registry.add((histPath + "dthetax_vs_y").c_str(), std::format("#Delta #theta_x vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dThetaxAxis}});
        alignmentHistos[i][j]["dthetax_vs_thetax"] = registry.add((histPath + "dthetax_vs_thetax").c_str(), std::format("#Delta #theta_x vs. |#theta_x| - {}", quadrant).c_str(), {HistType::kTH2F, {thetaxAxis, dThetaxAxis}});

        alignmentHistos[i][j]["dthetay_vs_x"] = registry.add((histPath + "dthetay_vs_x").c_str(), std::format("#Delta #theta_y vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dThetayAxis}});
        alignmentHistos[i][j]["dthetay_vs_y"] = registry.add((histPath + "dthetay_vs_y").c_str(), std::format("#Delta #theta_y vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dThetayAxis}});
        alignmentHistos[i][j]["dthetay_vs_thetay"] = registry.add((histPath + "dthetay_vs_thetay").c_str(), std::format("#Delta #theta_y vs. |#theta_y| - {}", quadrant).c_str(), {HistType::kTH2F, {thetayAxis, dThetayAxis}});

        // mixed events
        histPath = std::string("alignment/mixed-events/") + refPLane.first + "/" + quadrant + "/";
        alignmentHistosMixedEvents[i][j]["dx_vs_x"] = registry.add((histPath + "dx_vs_x").c_str(), std::format("#Delta x vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dxAxis}});
        alignmentHistosMixedEvents[i][j]["dx_vs_y"] = registry.add((histPath + "dx_vs_y").c_str(), std::format("#Delta x vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dxAxis}});
        alignmentHistosMixedEvents[i][j]["dy_vs_x"] = registry.add((histPath + "dy_vs_x").c_str(), std::format("#Delta y vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dyAxis}});
        alignmentHistosMixedEvents[i][j]["dy_vs_y"] = registry.add((histPath + "dy_vs_y").c_str(), std::format("#Delta y vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dyAxis}});

        alignmentHistosMixedEvents[i][j]["dthetax_vs_x"] = registry.add((histPath + "dthetax_vs_x").c_str(), std::format("#Delta #theta_x vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dThetaxAxis}});
        alignmentHistosMixedEvents[i][j]["dthetax_vs_y"] = registry.add((histPath + "dthetax_vs_y").c_str(), std::format("#Delta #theta_x vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dThetaxAxis}});
        alignmentHistosMixedEvents[i][j]["dthetax_vs_thetax"] = registry.add((histPath + "dthetax_vs_thetax").c_str(), std::format("#Delta #theta_x vs. |#theta_x| - {}", quadrant).c_str(), {HistType::kTH2F, {thetaxAxis, dThetaxAxis}});

        alignmentHistosMixedEvents[i][j]["dthetay_vs_x"] = registry.add((histPath + "dthetay_vs_x").c_str(), std::format("#Delta #theta_y vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dThetayAxis}});
        alignmentHistosMixedEvents[i][j]["dthetay_vs_y"] = registry.add((histPath + "dthetay_vs_y").c_str(), std::format("#Delta #theta_y vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dThetayAxis}});
        alignmentHistosMixedEvents[i][j]["dthetay_vs_thetay"] = registry.add((histPath + "dthetay_vs_thetay").c_str(), std::format("#Delta #theta_y vs. |#theta_y| - {}", quadrant).c_str(), {HistType::kTH2F, {thetayAxis, dThetayAxis}});
      }
    }
  }

  void init(o2::framework::InitContext&)
  {
    // Load geometry
    ccdbManager->setURL(ccdburl);
    ccdbManager->setCaching(true);
    ccdbManager->setLocalObjectValidityChecking();
    ccdbApi.init(ccdburl);
    mRunNumber = 0;

    if (!o2::base::GeometryManager::isGeometryLoaded()) {
      LOGF(info, "Load geometry from CCDB");
      ccdbManager->get<TGeoManager>(geoPath);
    }

    int nTrackTypes = static_cast<int>(o2::aod::fwdtrack::ForwardTrackTypeEnum::MCHStandaloneTrack) + 1;
    AxisSpec trackTypeAxis = {nTrackTypes, 0, nTrackTypes, "track type"};
    registry.add("nTracksPerType", "Number of tracks per type", {HistType::kTH1F, {trackTypeAxis}});

    // ======================
    // Muons plots
    // ======================

    AxisSpec chi2Axis = {1000, 0, 1000, "chi2"};
    AxisSpec momentumAxis = {1000, 0, 1000, "p (GeV/c)"};
    AxisSpec transverseMomentumAxis = {1000, 0, 100, "p_{T} (GeV/c)"};
    AxisSpec etaAxis = {80, -5, -1, "#eta"};
    AxisSpec rAbsAxis = {100, 0., 100.0, "R_{abs} (cm)"};
    AxisSpec dcaAxis = {400, 0.0, 20.0, "DCA"};
    AxisSpec pdcaAxis = {5000, 0.0, 5000.0, "p #times DCA"};
    AxisSpec phiAxis = {360, -180.0, 180.0, "#phi (degrees)"};

    registry.add("muons/TrackChi2", "MCH track #chi^{2}", {HistType::kTH1F, {chi2Axis}});
    registry.add("muons/TrackP", "MCH track momentum", {HistType::kTH1F, {momentumAxis}});
    registry.add("muons/TrackPt", "MCH track transverse momentum", {HistType::kTH1F, {transverseMomentumAxis}});
    registry.add("muons/TrackEta", "MCH track #eta", {HistType::kTH1F, {etaAxis}});
    registry.add("muons/TrackRabs", "MCH track R_{abs}", {HistType::kTH1F, {rAbsAxis}});
    registry.add("muons/TrackDCA", "MCH track DCA", {HistType::kTH1F, {dcaAxis}});
    registry.add("muons/TrackPDCA", "MCH track p #times DCA", {HistType::kTH1F, {pdcaAxis}});
    registry.add("muons/TrackPhi", "MCH track #phi", {HistType::kTH1F, {phiAxis}});

    // ======================
    // Global muons plots
    // ======================

    AxisSpec nCandidatesAxis = {fNCandidatesMax, 0, fNCandidatesMax, "match candidate rank"};
    registry.add("global-muons/NCandidates", "Number of MFT-MCH match candidates", {HistType::kTH1F, {nCandidatesAxis}});
    registry.add("global-muons/MatchChi2", "MFT-MCH match chi2", {HistType::kTH2F, {chi2Axis, nCandidatesAxis}});

    registry.add("global-muons/TrackChi2", "Muon track #chi^{2}", {HistType::kTH1F, {chi2Axis}});
    registry.add("global-muons/TrackP", "Muon track momentum", {HistType::kTH1F, {momentumAxis}});
    registry.add("global-muons/TrackPt", "Muon track transverse momentum", {HistType::kTH1F, {transverseMomentumAxis}});
    registry.add("global-muons/TrackEta", "Muon track #eta", {HistType::kTH1F, {etaAxis}});
    registry.add("global-muons/TrackRabs", "Muon track R_{abs}", {HistType::kTH1F, {rAbsAxis}});
    registry.add("global-muons/TrackDCA", "Muon track DCA", {HistType::kTH1F, {dcaAxis}});
    registry.add("global-muons/TrackPDCA", "Muon track p #times DCA", {HistType::kTH1F, {pdcaAxis}});
    registry.add("global-muons/TrackPhi", "Muon track #phi", {HistType::kTH1F, {phiAxis}});

    // ======================
    // Global muon plots with matching cuts
    // ======================

    AxisSpec dbcAxis = {1000, -500, 500, "#Delta_{BC}"};
    registry.add("global-matches/BCdifference", "MCH-MFT BC difference", {HistType::kTH1F, {dbcAxis}});

    AxisSpec nClustersAxis = {20, 0, 20, "# of MFT clusters per track"};
    registry.add("global-matches/NCandidates", "Number of MFT-MCH match candidates", {HistType::kTH1F, {nCandidatesAxis}});
    registry.add("global-matches/MatchChi2", "MFT-MCH match chi2", {HistType::kTH1F, {chi2Axis}});

    registry.add("global-matches/TrackChi2_MFT", "MFT track #chi^{2}", {HistType::kTH1F, {chi2Axis}});
    registry.add("global-matches/TrackNclusters_MFT", "MFT track Nclusters", {HistType::kTH1F, {nClustersAxis}});

    registry.add("global-matches/TrackChi2", "Muon track #chi^{2}", {HistType::kTH1F, {chi2Axis}});
    registry.add("global-matches/TrackP", "Muon track momentum", {HistType::kTH1F, {momentumAxis}});
    registry.add("global-matches/TrackPt", "Muon track transverse momentum", {HistType::kTH1F, {transverseMomentumAxis}});
    registry.add("global-matches/TrackEta", "Muon track #eta", {HistType::kTH1F, {etaAxis}});
    registry.add("global-matches/TrackRabs", "Muon track R_{abs}", {HistType::kTH1F, {rAbsAxis}});
    registry.add("global-matches/TrackDCA", "Muon track DCA", {HistType::kTH1F, {dcaAxis}});
    registry.add("global-matches/TrackPDCA", "Muon track p #times DCA", {HistType::kTH1F, {pdcaAxis}});
    registry.add("global-matches/TrackPhi", "Muon track #phi", {HistType::kTH1F, {phiAxis}});

    registry.add("global-matches/TrackP_glo", "Global muon track momentum", {HistType::kTH1F, {momentumAxis}});
    registry.add("global-matches/TrackPt_glo", "Global muon track transverse momentum", {HistType::kTH1F, {transverseMomentumAxis}});
    registry.add("global-matches/TrackEta_glo", "Global muon track #eta", {HistType::kTH1F, {etaAxis}});
    registry.add("global-matches/TrackDCA_glo", "Global muon track DCA", {HistType::kTH1F, {dcaAxis}});
    registry.add("global-matches/TrackPhi_glo", "Global muon track #phi", {HistType::kTH1F, {phiAxis}});

    AxisSpec momentumCorrelationAxis = {100, 0, 100, "momentum (GeV/c)"};
    AxisSpec momentumDeltaAxis = {100, -1, 1, "#DeltaP (GeV/c)"};
    // Momentum correlations
    registry.add("global-muons/MomentumCorrelation_Global_vs_Muon",
                 "P_{global} vs. P_{MCH}",
                 {HistType::kTH2F, {momentumCorrelationAxis, momentumCorrelationAxis}});
    registry.add("global-muons/MomentumDifference_Global_vs_Muon",
                 "(P_{global} - P_{MCH}) / P_{MCH} vs. P_{MCH}",
                 {HistType::kTH2F, {momentumCorrelationAxis, momentumDeltaAxis}});
    registry.add("global-muons/MomentumCorrelation_subleading_vs_leading",
                 "P_{subleading_match} vs. P_{leading_match}",
                 {HistType::kTH2F, {momentumCorrelationAxis, momentumCorrelationAxis}});
    registry.add("global-muons/MomentumDifference_subleading_vs_leading",
                 "(P_{subleading_match} - P_{leading_match}) / P_{leading_match} vs. P_{leading_match}",
                 {HistType::kTH2F, {momentumCorrelationAxis, momentumDeltaAxis}});

    //AxisSpec etaAxis = {100, -5.0, -2.0, "#eta"};
    AxisSpec etaCorrelationAxis = {80, -5.0, -1.0, "#eta"};
    AxisSpec etaDeltaAxis = {100, -0.2, 0.2, "#Delta#eta"};
    // Eta correlations
    registry.add("global-muons/EtaCorrelation_Global_vs_Muon",
                 "#eta_{global} vs. #eta_{MCH}",
                 {HistType::kTH2F, {etaCorrelationAxis, etaCorrelationAxis}});
    registry.add("global-muons/EtaDifference_Global_vs_Muon",
                 "(#eta_{global} - #eta_{MCH}) / #eta_{MCH} vs. #eta_{MCH}",
                 {HistType::kTH2F, {etaCorrelationAxis, etaDeltaAxis}});
    registry.add("global-muons/EtaCorrelation_subleading_vs_leading",
                 "#eta_{subleading_match} vs. #eta_{leading_match}",
                 {HistType::kTH2F, {etaCorrelationAxis, etaCorrelationAxis}});
    registry.add("global-muons/EtaDifference_subleading_vs_leading",
                 "(#eta_{subleading_match} - #eta_{leading_match}) / #eta_{leading_match} vs. #eta_{leading_match}",
                 {HistType::kTH2F, {etaCorrelationAxis, etaDeltaAxis}});


    // ======================
    // Di-muon plots
    // ======================

    AxisSpec invMassAxis = {400, 1, 5, "M_{#mu^{+}#mu^{-}} (GeV/c^{2})"};
    AxisSpec invMassCorrelationAxis = {400, 0, 8, "M_{#mu^{+}#mu^{-}} (GeV/c^{2})"};
    AxisSpec invMassAxisFull = {5000, 0, 100, "M_{#mu^{+}#mu^{-}} (GeV/c^{2})"};
    // MCH-MID tracks with MCH acceptance cuts
    registry.add("dimuon/invariantMass_MuonKine_MuonCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMassFull_MuonKine_MuonCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    registry.add("dimuon/mixed-events/invariantMass_MuonKine_MuonCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMassFull_MuonKine_MuonCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    // MCH-MID tracks with MFT acceptance cuts
    registry.add("dimuon/invariantMass_MuonKine_GlobalMuonCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMassFull_MuonKine_GlobalMuonCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    registry.add("dimuon/mixed-events/invariantMass_MuonKine_GlobalMuonCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMassFull_MuonKine_GlobalMuonCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    // Good MFT-MCH-MID tracks with MCH parameters and MFT acceptance cuts
    registry.add("dimuon/invariantMass_MuonKine_GlobalMatchesCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMassFull_MuonKine_GlobalMatchesCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    registry.add("dimuon/mixed-events/invariantMass_MuonKine_GlobalMatchesCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMassFull_MuonKine_GlobalMatchesCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    // Good MFT-MCH-MID tracks with global parameters MFT acceptance cuts
    registry.add("dimuon/invariantMass_GlobalMuonKine_GlobalMatchesCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    registry.add("dimuon/mixed-events/invariantMass_GlobalMuonKine_GlobalMatchesCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    // Good MFT-MCH-MID tracks with global parameters MFT acceptance cuts
    registry.add("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMassFull_ScaledMftKine_GlobalMatchesCuts", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum", {HistType::kTH1F, {invMassAxisFull}});
    registry.add("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMassFull_ScaledMftKine_GlobalMatchesCuts", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum", {HistType::kTH1F, {invMassAxisFull}});
    // combinations with sub-leading matches
    registry.add("dimuon/invariantMass_GlobalMuonKine_GlobalMatchesCuts_leading_subleading", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts_leading_subleading", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    registry.add("dimuon/invariantMass_GlobalMuonKine_GlobalMatchesCuts_subleading_leading", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts_subleading_leading", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});
    registry.add("dimuon/invariantMass_GlobalMuonKine_GlobalMatchesCuts_subleading_subleading", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts_subleading_subleading", "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxisFull}});

    // invariant mass correlations
    registry.add("dimuon/invariantMass_MuonKine_vs_GlobalMuonKine", "M_{#mu^{+}#mu^{-}} - muon tracks vs. global tracks", {HistType::kTH2F, {invMassCorrelationAxis, invMassCorrelationAxis}});
    registry.add("dimuon/invariantMass_ScaledMftKine_vs_GlobalMuonKine", "M_{#mu^{+}#mu^{-}} - rescaled MFT tracks vs. global tracks", {HistType::kTH2F, {invMassCorrelationAxis, invMassCorrelationAxis}});
    registry.add("dimuon/invariantMass_GlobalMuonKine_subleading_vs_leading", "M_{#mu^{+}#mu^{-}} - subleading vs. leading matches", {HistType::kTH2F, {invMassCorrelationAxis, invMassCorrelationAxis}});

    CreateAlignementHistos();
    /*
     * Plots to add:
     * - Global vs. MCH momentum
     * - Relative (Global - MCH) momentum difference vs. MCH momentum
     * - sub-leading vs. leading momentum
     * - Relative (sub-leading - leading) momentum difference vs. leading momentum
     * - leading vs sub-leading matching chi2
     * - invariant mass for sub-leading matches that pass the chi2 cut
     * - distributions of the variables used for the tracks selection
     * - DCA and pDCA plots for MCH, MFT and global muon tracks
     */
  }

  template<class T>
  double GetQuadrant(const T& track)
  {
    double phi = track.phi() * 180 / TMath::Pi();
    if (phi >= 0 && phi < 90) {
      return 0;
    }
    if (phi >= 90 && phi <= 180) {
      return 1;
    }
    if (phi >= -180 && phi < -90) {
      return 2;
    }
    if (phi >= -90 && phi < 0) {
      return 3;
    }
    return -1;
  }

  template<class T, class C>
  double GetDCA(const T& track, const C& collision)
  {
    // propagate muon track to DCA
    auto trackAtDCA = VarManager::PropagateMuon(track, collision, toDCA);
    // Calculate DCA quantities (preferable to do it with VarManager)
    double dcax = trackAtDCA.getX() - collision.posX();
    double dcay = trackAtDCA.getY() - collision.posY();
    return std::sqrt(dcax * dcax + dcay * dcay);
  }

  template<class T, class C>
  bool pDCACut(const T& mchTrack, const C& collision, double nSigmaPDCA)
  {
    static const double sigmaPDCA23 = 80.;
    static const double sigmaPDCA310 = 54.;
    static const double relPRes = 0.0004;
    static const double slopeRes = 0.0005;

    double thetaAbs = TMath::ATan(mchTrack.rAtAbsorberEnd() / 505.) * TMath::RadToDeg();

    // propagate muon track to vertex
    auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, toVertex);

    //double pUncorr = mchTrack.p();
    double p = mchTrackAtVertex.getP();

    double pDCA = mchTrack.pDca();
    double sigmaPDCA = (thetaAbs < 3) ? sigmaPDCA23 : sigmaPDCA310;
    double nrp = nSigmaPDCA * relPRes * p;
    double pResEffect = sigmaPDCA / (1. - nrp / (1. + nrp));
    double slopeResEffect = 535. * slopeRes * p;
    double sigmaPDCAWithRes = TMath::Sqrt(pResEffect * pResEffect + slopeResEffect * slopeResEffect);
    if (pDCA > nSigmaPDCA * sigmaPDCAWithRes) {
      return false;
    }

    return true;
  }

  template<class T, class C>
  bool IsGoodMuon(const T& muonTrack, const C& collision,
                  double chi2Cut,
                  double pCut,
                  double pTCut,
                  std::array<double, 2> etaCut,
                  std::array<double, 2> rAbsCut,
                  double nSigmaPdcaCut)
  {
    auto const& mchTrack = (static_cast<int>(muonTrack.trackType()) <= 2) ?
        muonTrack.template matchMCHTrack_as<MyMuonsWithCov>() :
        muonTrack;

    // chi2 cut
    if (mchTrack.chi2() > chi2Cut)
      return false;

    // momentum cut
    if (mchTrack.p() < pCut) {
      return false; // skip low-momentum tracks
    }

    // transverse momentum cut
    if (mchTrack.pt() < pTCut) {
      return false; // skip low-momentum tracks
    }

    // Eta cut
    double eta = mchTrack.eta();
    if ((eta < etaCut[0] || eta > etaCut[1])) {
      return false;
    }

    // RAbs cut
    double rAbs = mchTrack.rAtAbsorberEnd();
    if ((rAbs < rAbsCut[0] || rAbs > rAbsCut[1])) {
      return false;
    }

    // pDCA cut
    if (!pDCACut(mchTrack, collision, nSigmaPdcaCut)) {
      return false;
    }

    return true;
  }

  template<class T, class C>
  bool IsGoodMuon(const T& muonTrack, const C& collision)
  {
    return IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMchLow, fEtaMchUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp);
  }

  template<class T, class C>
  bool IsGoodGlobalMuon(const T& muonTrack, const C& collision)
  {
    return IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp);
  }

  template<class T>
  bool IsGoodMFT(const T& mftTrack,
                 double chi2Cut,
                 int nClustersCut)
  {
    // chi2 cut
    if (mftTrack.chi2() > chi2Cut)
      return false;

    // number of clusters cut
    if (mftTrack.nClusters() < nClustersCut)
      return false;

    return true;
  }

  template<class T>
  bool IsGoodMFT(const T& mftTrack)
  {
    return IsGoodMFT(mftTrack, fTrackChi2MftUp, fTrackNClustMftLow);
  }

  template<class T, class C>
  bool IsGoodGlobalMatching(const T& muonTrack, const C& collision,
                            double chi2CutMFT,
                            int nClustersCutMFT,
                            double matchingChi2Cut)
  {
    if (static_cast<int>(muonTrack.trackType()) >= 2)
      return false;

    auto const& mftTrack = muonTrack.template matchMFTTrack_as<MyMFTs>();

    if (!IsGoodMFT(mftTrack,
                   chi2CutMFT,
                   nClustersCutMFT))
      return false;

    // MFT-MCH matching chi2 cut
    if (muonTrack.chi2MatchMCHMFT() > matchingChi2Cut)
      return false;

    return true;
  }

  template<class T, class C>
  bool IsGoodGlobalMatching(const T& muonTrack, const C& collision)
  {
    return IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp);
  }

  template<class C>
  bool IsSameEvent(const C& c1, const C& c2)
  {
    return (c1.bc == c2.bc);
  }

  template<class C>
  bool IsMixedEvent(const C& c1, const C& c2)
  {
    if (IsSameEvent(c1, c2))
      return false;

    uint64_t bcDiff = (c2.bc > c1.bc) ? (c2.bc - c1.bc) : (c2.bc - c1.bc);
    // in the event mixing case, we require a minimum BC gap between the collisions
    if (bcDiff < fEventMinDeltaBc)
      return false;
    // we also require that the collisions have similar Z positions and multiplicity of MFT tracks
    if (std::fabs(c2.zVertex - c1.zVertex) > fEventMaxDeltaVtxZ)
      return false;
    if (std::abs(c2.mftTracksMultiplicity - c1.mftTracksMultiplicity) > fEventMaxDeltaNMFT)
      return false;

    return true;
  }

  double GetMuMuInvariantMass(const o2::dataformats::GlobalFwdTrack& track1, const o2::dataformats::GlobalFwdTrack& track2)
  {
    ROOT::Math::PxPyPzMVector muon1{
      track1.getPx(),
      track1.getPy(),
      track1.getPz(),
      o2::constants::physics::MassMuon};

    ROOT::Math::PxPyPzMVector muon2{
      track2.getPx(),
      track2.getPy(),
      track2.getPz(),
      o2::constants::physics::MassMuon};

    auto dimuon = muon1 + muon2;

    //std::cout << std::format("[TOTO] P1=({:0.2f} [{:0.2f},{:0.2f},{:0.2f}])  P2=({:0.2f} [{:0.2f},{:0.2f},{:0.2f}])  M={:0.2f}",
    //    track1.getP(), track1.getPx(), track1.getPy(), track1.getPz(),
    //    track2.getP(), track2.getPx(), track2.getPy(), track2.getPz(),
    //    dimuon.M()) << std::endl;

    return dimuon.M();
  }

  template<class T, class C>
  double GetMuMuInvariantMass(const T& track1, const T& track2, const C& collision)
  {
    // propagate muon tracks to vertex
    auto const& muonTrack1AtVertex = VarManager::PropagateMuon(track1, collision, toVertex);
    auto const& muonTrack2AtVertex = VarManager::PropagateMuon(track2, collision, toVertex);

    return GetMuMuInvariantMass(muonTrack1AtVertex, muonTrack2AtVertex);
  }

  template<class TMFT, class C>
  o2::dataformats::GlobalFwdTrack PropagateMftToDCA(const TMFT& mftTrack, const C& collision)
  {
    double chi2 = mftTrack.chi2();
    SMatrix5 tpars = {mftTrack.x(), mftTrack.y(), mftTrack.phi(), mftTrack.tgl(), mftTrack.signed1Pt()};
    std::vector<double> v1;
    SMatrix55 tcovs{v1.begin(), v1.end()};
    o2::track::TrackParCovFwd fwdtrack{mftTrack.z(), tpars, tcovs, chi2};
    o2::dataformats::GlobalFwdTrack propmuon;

    double propVec[3] = {};
    propVec[0] = collision.posX() - mftTrack.x();
    propVec[1] = collision.posY() - mftTrack.y();
    propVec[2] = collision.posZ() - mftTrack.z();

    double centerZ[3] = {mftTrack.x() + propVec[0] / 2.,
                         mftTrack.y() + propVec[1] / 2.,
                         mftTrack.z() + propVec[2] / 2.};
    o2::field::MagneticField* field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    auto Bz = field->getBz(centerZ);
    fwdtrack.propagateToZ(collision.posZ(), Bz);

    propmuon.setParameters(fwdtrack.getParameters());
    propmuon.setZ(fwdtrack.getZ());
    propmuon.setCovariances(fwdtrack.getCovariances());

    return propmuon;
  }

  template<class TMFT, class TMCH, class C>
  o2::dataformats::GlobalFwdTrack PropagateMftToVertex(const TMFT& mftTrack, const TMCH& mchTrack, const C& collision)
  {
    // propagate MCH track to the vertex to get the updated momentum
    auto const& mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToDCA);
    int sign = mchTrack.sign();

    // get scaling factor for MFT momentum
    double pScale = mchTrackAtVertex.getP() / mftTrack.p();
    double signed1Pt = mftTrack.signed1Pt() * sign / pScale;

    //std::cout << std::format("[TOTO]   P(MCH)=({:0.2f},{:0.2f})  P(MFT)=({:0.2f},{:0.2f})  Pt(scaled)={:0.2f}",
    //    mchTrackAtVertex.getP(), mchTrackAtVertex.getPt(),
    //    mftTrack.p(), mftTrack.signed1Pt(), signed1Pt) << std::endl;

    double chi2 = mftTrack.chi2();
    SMatrix5 tpars = {mftTrack.x(), mftTrack.y(), mftTrack.phi(), mftTrack.tgl(), signed1Pt};
    std::vector<double> v1;
    SMatrix55 tcovs{v1.begin(), v1.end()};
    o2::track::TrackParCovFwd fwdtrack{mftTrack.z(), tpars, tcovs, chi2};
    o2::dataformats::GlobalFwdTrack propmuon;

    double centerMFT[3] = {0, 0, -61.4};
    o2::field::MagneticField* field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    auto Bz = field->getBz(centerMFT); // Get field at centre of MFT
    auto geoMan = o2::base::GeometryManager::meanMaterialBudget(mftTrack.x(), mftTrack.y(), mftTrack.z(), collision.posX(), collision.posY(), collision.posZ());
    auto x2x0 = static_cast<float>(geoMan.meanX2X0);
    fwdtrack.propagateToVtxhelixWithMCS(collision.posZ(), {collision.posX(), collision.posY()}, {collision.covXX(), collision.covYY()}, Bz, x2x0);
    propmuon.setParameters(fwdtrack.getParameters());
    propmuon.setZ(fwdtrack.getZ());
    propmuon.setCovariances(fwdtrack.getCovariances());

    return propmuon;
  }

  template<class TMFT, class TMCH, class C>
  double GetMuMuInvariantMass(const TMFT& mftTrack1, const TMCH& mchTrack1, const TMFT& mftTrack2, const TMCH& mchTrack2, const C& collision)
  {
    auto mftTrack1AtVertex = PropagateMftToVertex(mftTrack1, mchTrack1, collision);
    auto mftTrack2AtVertex = PropagateMftToVertex(mftTrack2, mchTrack2, collision);

    return GetMuMuInvariantMass(mftTrack1AtVertex, mftTrack2AtVertex);
  }

  using MuonPair = std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>>;
  using GlobalMuonPair = std::pair<std::pair<uint64_t, std::vector<uint64_t>>, std::pair<uint64_t, std::vector<uint64_t>>>;

  void GetMuonPairs(const std::map<uint64_t, CollisionInfo>& collisionInfos,
                    std::vector<MuonPair>& muonPairs,
                    std::vector<GlobalMuonPair>& globalMuonPairs)
  {
    // muon tracks - outer loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {

      // outer loop over muon tracks
      for (auto mchIndex1 : collisionInfo1.mchTracks) {

        // inner loop over collisions
        for (auto& [collisionIndex2, collisionInfo2] : collisionInfos) {
          // avoid double-counting of collisions
          if (collisionIndex2 < collisionIndex1) continue;

          bool sameEvent = (collisionIndex1 == collisionIndex2);
          bool mixedEvent = IsMixedEvent(collisionInfo1, collisionInfo2);

          if (!sameEvent && !mixedEvent)
            continue;

          // inner loop over muon tracks
          for (auto mchIndex2 : collisionInfo2.mchTracks) {
            // avoid double-counting of muon pairs if we are not mixing events
            if (sameEvent && mchIndex2 <= mchIndex1) continue;

            MuonPair muonPair{{collisionIndex1, mchIndex1}, {collisionIndex2, mchIndex2}};
            muonPairs.emplace_back(muonPair);
          }
        }
      }
    }

    // global muon tracks - outer loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {

      // outer loop over global muon tracks
      for (auto& [mchIndex1, globalTracksVector1] : collisionInfo1.globalMuonTracks) {

        // inner loop over collisions
        for (auto& [collisionIndex2, collisionInfo2] : collisionInfos) {
          // avoid double-counting of collisions
          if (collisionIndex2 < collisionIndex1) continue;

          bool sameEvent = (collisionIndex1 == collisionIndex2);
          bool mixedEvent = IsMixedEvent(collisionInfo1, collisionInfo2);

          if (!sameEvent && !mixedEvent)
            continue;

          // outer loop over global muon tracks
          for (auto& [mchIndex2, globalTracksVector2] : collisionInfo2.globalMuonTracks) {
            // avoid double-counting of muon pairs if we are not mixing events
            if (sameEvent && mchIndex2 <= mchIndex1) continue;

            GlobalMuonPair muonPair{{collisionIndex1, globalTracksVector1}, {collisionIndex2, globalTracksVector2}};
            globalMuonPairs.emplace_back(muonPair);
          }
        }
      }
    }
  }

  void FillMuonPlots(MyEvents const& collisions,
                     aod::BCsWithTimestamps const& bcs,
                     MyMuonsWithCov const& muonTracks,
                     //MyMFTs const& mftTracks,
                     const std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    constexpr double doubleMin = std::numeric_limits<double>::min();
    constexpr double doubleMax = std::numeric_limits<double>::max();

    // loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {
      auto const& collision = collisions.rawIteratorAt(collisionIndex1);

      // loop over muon tracks
      for (auto mchIndex : collisionInfo1.mchTracks) {
        auto const& muonTrack = muonTracks.rawIteratorAt(mchIndex);

        // Kinematic plots
        // Quality cuts are applied to all variables except the one being plotted

        // track chi2 distribution
        if (IsGoodMuon(muonTrack, collision, doubleMax, fPMchLow, fPtMchLow, {fEtaMchLow, fEtaMchUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("muons/TrackChi2"))->Fill(muonTrack.chi2());
        }

        // track momentum distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, 0, fPtMchLow, {fEtaMchLow, fEtaMchUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("muons/TrackP"))->Fill(muonTrack.p());
        }

        // track transverse momentum distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, 0, {fEtaMchLow, fEtaMchUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("muons/TrackPt"))->Fill(muonTrack.pt());
        }

        // track eta distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {-doubleMax, doubleMax}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("muons/TrackEta"))->Fill(muonTrack.eta());
        }

        // track Rabs distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMchLow, fEtaMchUp}, {0, doubleMax}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("muons/TrackRabs"))->Fill(muonTrack.rAtAbsorberEnd());
        }

        // track pDCA distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMchLow, fEtaMchUp}, {fRabsLow, fRabsUp}, doubleMax)) {
          registry.get<TH1>(HIST("muons/TrackPDCA"))->Fill(muonTrack.pDca());
        }

        // track phi distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMchLow, fEtaMchUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("muons/TrackPhi"))->Fill(muonTrack.phi() * 180.0 / TMath::Pi());
          registry.get<TH1>(HIST("muons/TrackDCA"))->Fill(GetDCA(muonTrack, collision));
        }
      }

      // loop over global muon tracks
      for (auto& [mchIndex1, globalTracksVector1] : collisionInfo1.globalMuonTracks) {
        // get the best global match for this muon track
        auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector1[0]);
        auto const& mchTrack = muonTrack.template matchMCHTrack_as<MyMuonsWithCov>();
        auto const& mftTrack = muonTrack.template matchMFTTrack_as<MyMFTs>();

        // Global muons plots
        registry.get<TH1>(HIST("global-muons/NCandidates"))->Fill(globalTracksVector1.size());
        // loop over global matching candidates
        for (size_t candidateIndex = 0; candidateIndex < globalTracksVector1.size(); candidateIndex++) {
          auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector1[candidateIndex]);
          registry.get<TH2>(HIST("global-muons/MatchChi2"))->Fill(muonTrack.chi2MatchMCHMFT(), candidateIndex);
        }

        // Kinematic plots
        // Quality cuts are applied to all variables except the one being plotted

        if (muonTrack.has_collision() && mftTrack.has_collision()) {
          auto collisionMuon = collisions.rawIteratorAt(muonTrack.collisionId());
          int64_t bcMuon = bcs.rawIteratorAt(collisionMuon.bcId()).globalBC();
          auto collisionMFT = collisions.rawIteratorAt(mftTrack.collisionId());
          int64_t bcMFT = bcs.rawIteratorAt(collisionMFT.bcId()).globalBC();

          int64_t dbc = bcMuon - bcMFT;
          registry.get<TH1>(HIST("global-matches/BCdifference"))->Fill(dbc);
        }

        // track chi2 distribution
        if (IsGoodMuon(muonTrack, collision, doubleMax, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("global-muons/TrackChi2"))->Fill(mchTrack.chi2());
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackChi2"))->Fill(mchTrack.chi2());
          }
        }

        // track momentum distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, 0, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("global-muons/TrackP"))->Fill(mchTrack.p());
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackP"))->Fill(mchTrack.p());
          }
        }

        // track transverse momentum distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, 0, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("global-muons/TrackPt"))->Fill(mchTrack.pt());
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackPt"))->Fill(mchTrack.pt());
          }
        }

        // track eta distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {-doubleMax, doubleMax}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("global-muons/TrackEta"))->Fill(mchTrack.eta());
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackEta"))->Fill(mchTrack.eta());
          }
        }

        // track Rabs distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {0, doubleMax}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("global-muons/TrackRabs"))->Fill(mchTrack.rAtAbsorberEnd());
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackRabs"))->Fill(mchTrack.rAtAbsorberEnd());
          }
        }

        // track pDCA distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, doubleMax)) {
          registry.get<TH1>(HIST("global-muons/TrackPDCA"))->Fill(mchTrack.pDca());
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackPDCA"))->Fill(mchTrack.pDca());
          }
        }

        // track DCA and phi distributions
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          double dca = GetDCA(mchTrack, collision);
          registry.get<TH1>(HIST("global-muons/TrackDCA"))->Fill(dca);
          registry.get<TH1>(HIST("global-muons/TrackPhi"))->Fill(mchTrack.phi() * 180.0 / TMath::Pi());
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackDCA"))->Fill(dca);
            registry.get<TH1>(HIST("global-matches/TrackPhi"))->Fill(mchTrack.phi() * 180.0 / TMath::Pi());
          }
        }

        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          if (IsGoodGlobalMatching(muonTrack, collision, doubleMax, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackChi2_MFT"))->Fill(mftTrack.chi2());
          }
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, 0, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackNclusters_MFT"))->Fill(mftTrack.nClusters());
          }
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, doubleMax)) {
            registry.get<TH1>(HIST("global-matches/MatchChi2"))->Fill(muonTrack.chi2MatchMCHMFT());
          }

          // variables not used in the quality cuts
          if (IsGoodGlobalMatching(muonTrack, collision, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackP_glo"))->Fill(muonTrack.p());
            registry.get<TH1>(HIST("global-matches/TrackPt_glo"))->Fill(muonTrack.pt());
            registry.get<TH1>(HIST("global-matches/TrackEta_glo"))->Fill(muonTrack.eta());
            registry.get<TH1>(HIST("global-matches/TrackPhi_glo"))->Fill(muonTrack.phi() * 180.0 / TMath::Pi());
            registry.get<TH1>(HIST("global-matches/TrackDCA_glo"))->Fill(GetDCA(muonTrack, collision));
          }
        }
      }
    }
  }

  void FillDimuonPlots(MyEvents const& collisions,
                       MyMuonsWithCov const& muonTracks,
                       //MyMFTs const& mftTracks,
                       const std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    std::vector<MuonPair> muonPairs;
    std::vector<GlobalMuonPair> globalMuonPairs;

    GetMuonPairs(collisionInfos, muonPairs, globalMuonPairs);

    for (auto& [muon1, muon2] : muonPairs) {
      auto collisionIndex1 = muon1.first;
      auto collisionIndex2 = muon2.first;
      auto mchIndex1 = muon1.second;
      auto mchIndex2 = muon2.second;

      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);
      auto const& collision2 = collisions.rawIteratorAt(collisionIndex2);

      auto const& muonTrack1 = muonTracks.rawIteratorAt(mchIndex1);
      auto const& muonTrack2 = muonTracks.rawIteratorAt(mchIndex2);
      int sign1 = muonTrack1.sign();
      int sign2 = muonTrack2.sign();

      // only consider opposite-sign pairs
      if ((sign1 * sign2) >= 0) continue;

      bool goodMuonTracks = (IsGoodMuon(muonTrack1, collision1) && IsGoodMuon(muonTrack2, collision1));
      bool goodGlobalMuonTracks = (IsGoodGlobalMuon(muonTrack1, collision1) && IsGoodGlobalMuon(muonTrack2, collision1));

      bool sameEvent = (collisionIndex1 == collisionIndex2);

      if (goodMuonTracks) {
        //std::cout << "[TOTO] muon tracks with muon cuts" << std::endl;
        double mass = GetMuMuInvariantMass(muonTrack1, muonTrack2, collision1);
        if (sameEvent) {
          // same-event case
          registry.get<TH1>(HIST("dimuon/invariantMass_MuonKine_MuonCuts"))->Fill(mass);
          registry.get<TH1>(HIST("dimuon/invariantMassFull_MuonKine_MuonCuts"))->Fill(mass);
        } else {
          // event-mixing case
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMass_MuonKine_MuonCuts"))->Fill(mass);
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMassFull_MuonKine_MuonCuts"))->Fill(mass);
        }
      }

      if (goodGlobalMuonTracks) {
        double mass = GetMuMuInvariantMass(muonTrack1, muonTrack2, collision1);
        if (sameEvent) {
          // same-event case
          registry.get<TH1>(HIST("dimuon/invariantMass_MuonKine_GlobalMuonCuts"))->Fill(mass);
          registry.get<TH1>(HIST("dimuon/invariantMassFull_MuonKine_GlobalMuonCuts"))->Fill(mass);
        } else {
          // event-mixing case
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMass_MuonKine_GlobalMuonCuts"))->Fill(mass);
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMassFull_MuonKine_GlobalMuonCuts"))->Fill(mass);
        }
      }
    }

    for (auto& [muon1, muon2] : globalMuonPairs) {
      auto collisionIndex1 = muon1.first;
      auto collisionIndex2 = muon2.first;
      auto& globalTracksVector1 = muon1.second;
      auto& globalTracksVector2 = muon2.second;

      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);
      auto const& collision2 = collisions.rawIteratorAt(collisionIndex2);

      auto const& muonTrack1 = muonTracks.rawIteratorAt(globalTracksVector1[0]);
      auto const& muonTrack2 = muonTracks.rawIteratorAt(globalTracksVector2[0]);
      auto const& mftTrack1 = muonTrack1.template matchMFTTrack_as<MyMFTs>();
      auto const& mftTrack2 = muonTrack2.template matchMFTTrack_as<MyMFTs>();
      auto const& mchTrack1 = muonTrack1.template matchMCHTrack_as<MyMuonsWithCov>();
      auto const& mchTrack2 = muonTrack2.template matchMCHTrack_as<MyMuonsWithCov>();
      int sign1 = mchTrack1.sign();
      int sign2 = mchTrack2.sign();

      // only consider opposite-sign pairs
      if ((sign1 * sign2) >= 0) continue;

      bool goodGlobalMuonTracks = (IsGoodGlobalMuon(muonTrack1, collision1) && IsGoodGlobalMuon(muonTrack2, collision1));
      bool goodGlobalMuonMatches = (IsGoodGlobalMatching(muonTrack1, collision1) && IsGoodGlobalMatching(muonTrack2, collision1));

      bool sameEvent = (collisionIndex1 == collisionIndex2);

      if (goodGlobalMuonTracks && goodGlobalMuonMatches) {
        //std::cout << "[TOTO] muon tracks with global muon cuts" << std::endl;
        double massMCH = GetMuMuInvariantMass(mchTrack1, mchTrack2, collision1);
        //std::cout << "[TOTO] global muon tracks with global muon cuts" << std::endl;
        double mass = GetMuMuInvariantMass(muonTrack1, muonTrack2, collision1);
        //std::cout << "[TOTO] scaled MFT tracks with global muon cuts" << std::endl;
        double massScaled = GetMuMuInvariantMass(mftTrack1, mchTrack1, mftTrack2, mchTrack2, collision1);
        if (sameEvent) {
          // same-event case
          registry.get<TH1>(HIST("dimuon/invariantMass_MuonKine_GlobalMatchesCuts"))->Fill(massMCH);
          registry.get<TH1>(HIST("dimuon/invariantMassFull_MuonKine_GlobalMatchesCuts"))->Fill(massMCH);
          registry.get<TH1>(HIST("dimuon/invariantMass_GlobalMuonKine_GlobalMatchesCuts"))->Fill(mass);
          registry.get<TH1>(HIST("dimuon/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts"))->Fill(mass);
          registry.get<TH1>(HIST("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts"))->Fill(massScaled);
          registry.get<TH1>(HIST("dimuon/invariantMassFull_ScaledMftKine_GlobalMatchesCuts"))->Fill(massScaled);

          // mass correlation
          registry.get<TH2>(HIST("dimuon/invariantMass_MuonKine_vs_GlobalMuonKine"))->Fill(mass, massMCH);
          registry.get<TH2>(HIST("dimuon/invariantMass_ScaledMftKine_vs_GlobalMuonKine"))->Fill(mass, massScaled);
        } else {
          // event-mixing case
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMass_MuonKine_GlobalMatchesCuts"))->Fill(massMCH);
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMassFull_MuonKine_GlobalMatchesCuts"))->Fill(massMCH);
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMass_GlobalMuonKine_GlobalMatchesCuts"))->Fill(mass);
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts"))->Fill(mass);
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts"))->Fill(massScaled);
          registry.get<TH1>(HIST("dimuon/mixed-events/invariantMassFull_ScaledMftKine_GlobalMatchesCuts"))->Fill(massScaled);
        }
      }

      // plots for sub-leading matches are only filled in the same-event case
      if (sameEvent) {
        if (globalTracksVector1.size() > 1) {
          auto const& muonTrack1b = muonTracks.rawIteratorAt(globalTracksVector1[1]);
          goodGlobalMuonTracks = (IsGoodGlobalMuon(muonTrack1b, collision1) && IsGoodGlobalMuon(muonTrack2, collision1));
          goodGlobalMuonMatches = (IsGoodGlobalMatching(muonTrack1b, collision1) && IsGoodGlobalMatching(muonTrack2, collision1));
          double mass = GetMuMuInvariantMass(muonTrack1b, muonTrack2, collision1);
          if (goodGlobalMuonTracks && goodGlobalMuonMatches) {
            registry.get<TH1>(HIST("dimuon/invariantMass_GlobalMuonKine_GlobalMatchesCuts_subleading_leading"))->Fill(mass);
            registry.get<TH1>(HIST("dimuon/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts_subleading_leading"))->Fill(mass);
          }
        }

        if (globalTracksVector2.size() > 1) {
          auto const& muonTrack2b = muonTracks.rawIteratorAt(globalTracksVector2[1]);
          goodGlobalMuonTracks = (IsGoodGlobalMuon(muonTrack1, collision1) && IsGoodGlobalMuon(muonTrack2b, collision1));
          goodGlobalMuonMatches = (IsGoodGlobalMatching(muonTrack1, collision1) && IsGoodGlobalMatching(muonTrack2b, collision1));
          double mass = GetMuMuInvariantMass(muonTrack1, muonTrack2b, collision1);
          if (goodGlobalMuonTracks && goodGlobalMuonMatches) {
            registry.get<TH1>(HIST("dimuon/invariantMass_GlobalMuonKine_GlobalMatchesCuts_leading_subleading"))->Fill(mass);
            registry.get<TH1>(HIST("dimuon/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts_leading_subleading"))->Fill(mass);
          }
        }

        if (globalTracksVector1.size() > 1 && globalTracksVector2.size() > 1) {
          auto const& muonTrack1b = muonTracks.rawIteratorAt(globalTracksVector1[1]);
          auto const& muonTrack2b = muonTracks.rawIteratorAt(globalTracksVector2[1]);
          goodGlobalMuonTracks = (IsGoodGlobalMuon(muonTrack1b, collision1) && IsGoodGlobalMuon(muonTrack2b, collision1));
          goodGlobalMuonMatches = (IsGoodGlobalMatching(muonTrack1b, collision1) && IsGoodGlobalMatching(muonTrack2b, collision1));
          double mass = GetMuMuInvariantMass(muonTrack1b, muonTrack2b, collision1);
          double massLeading = GetMuMuInvariantMass(muonTrack1, muonTrack2, collision1);
          if (goodGlobalMuonTracks && goodGlobalMuonMatches) {
            registry.get<TH1>(HIST("dimuon/invariantMass_GlobalMuonKine_GlobalMatchesCuts_subleading_subleading"))->Fill(mass);
            registry.get<TH1>(HIST("dimuon/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts_subleading_subleading"))->Fill(mass);

            // mass correlation
            registry.get<TH2>(HIST("dimuon/invariantMass_GlobalMuonKine_subleading_vs_leading"))->Fill(massLeading, mass);
          }
        }
      }
    }
  }

  template <typename T>
  o2::dataformats::GlobalFwdTrack PropagateToZMCH(const T& muon, const double z)
  {
    double chi2 = muon.chi2();
    SMatrix5 tpars(muon.x(), muon.y(), muon.phi(), muon.tgl(), muon.signed1Pt());
    std::vector<double> v1{muon.cXX(), muon.cXY(), muon.cYY(), muon.cPhiX(), muon.cPhiY(),
                           muon.cPhiPhi(), muon.cTglX(), muon.cTglY(), muon.cTglPhi(), muon.cTglTgl(),
                           muon.c1PtX(), muon.c1PtY(), muon.c1PtPhi(), muon.c1PtTgl(), muon.c1Pt21Pt2()};
    SMatrix55 tcovs(v1.begin(), v1.end());
    o2::track::TrackParCovFwd fwdtrack{muon.z(), tpars, tcovs, chi2};
    o2::dataformats::GlobalFwdTrack track;
    track.setParameters(tpars);
    track.setZ(fwdtrack.getZ());
    track.setCovariances(tcovs);
    auto mchTrack = sExtrap.FwdtoMCH(track);

    if (z > -505.) {
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrack, z);
    } else {
      o2::mch::TrackExtrap::extrapToZ(mchTrack, z);
    }

    auto proptrack = sExtrap.MCHtoFwd(mchTrack);
    o2::dataformats::GlobalFwdTrack propmuon;
    propmuon.setParameters(proptrack.getParameters());
    propmuon.setZ(proptrack.getZ());
    propmuon.setCovariances(proptrack.getCovariances());

    return propmuon;
  }

  template <typename TMFT>
  o2::dataformats::GlobalFwdTrack PropagateToZMFT(const TMFT& mftTrack, const double pMCH, const double z)
  {
    double px = pMCH * sin(M_PI / 2 - atan(mftTrack.tgl())) * cos(mftTrack.phi());
    double py = pMCH * sin(M_PI / 2 - atan(mftTrack.tgl())) * sin(mftTrack.phi());
    double pz = pMCH * cos(M_PI / 2 - atan(mftTrack.tgl()));
    double pt = std::sqrt(std::pow(px, 2) + std::pow(py, 2));
    double sign = mftTrack.sign();

    SMatrix5 tpars = {mftTrack.x(), mftTrack.y(), mftTrack.phi(), mftTrack.tgl(), sign / pt};
    std::vector<double> v1;
    SMatrix55 tcovs{v1.begin(), v1.end()};
    //o2::track::TrackParCovFwd mchTrack = {mftTrack.z(), mftpars, mftcovs, mftTrack.chi2()};

    o2::dataformats::GlobalFwdTrack track;
    track.setParameters(tpars);
    track.setZ(mftTrack.z());
    track.setCovariances(tcovs);

    auto mchTrackExt = sExtrap.FwdtoMCH(track);

    o2::mch::TrackExtrap::extrapToZ(mchTrackExt, z);

    o2::dataformats::GlobalFwdTrack propmuon;
    auto proptrack = sExtrap.MCHtoFwd(mchTrackExt);
    propmuon.setParameters(proptrack.getParameters());
    propmuon.setZ(proptrack.getZ());
    propmuon.setCovariances(proptrack.getCovariances());

    return propmuon;
  }

  void FillAlignmentPlots(MyEvents const& collisions,
                          aod::BCsWithTimestamps const& bcs,
                          MyMuonsWithCov const& muonTracks,
                          MyMFTs const& mftTracks,
                          const std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    constexpr double doubleMin = std::numeric_limits<double>::min();
    constexpr double doubleMax = std::numeric_limits<double>::max();

    std::array<double, 4> zRefPlane = {firstMFTPlaneZ, lastMFTPlaneZ, -300.0, -505.0};

    // outer loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {
      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);
      int64_t bc = bcs.rawIteratorAt(collision1.bcId()).globalBC();

      // outer loop over global muon tracks
      for (auto& [mchIndex, globalTracksVector] : collisionInfo1.globalMuonTracks) {
        auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[0]);
        const auto& mchTrack = muonTrack.template matchMCHTrack_as<MyMuonsWithCov>();
        const auto& mftMatchedTrack = muonTrack.template matchMFTTrack_as<MyMFTs>();

        if (!mftMatchedTrack.has_collision())
          continue;

        auto collisionMFTmatched = collisions.rawIteratorAt(mftMatchedTrack.collisionId());
        int64_t bcMFTmatched = bcs.rawIteratorAt(collisionMFTmatched.bcId()).globalBC();

        int quadrant = GetQuadrant(mchTrack);

        bool isGoodMuon = IsGoodMuon(mchTrack, collision1, fTrackChi2MchUp, 20.0, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp);
        if (!isGoodMuon) continue;

        std::array<o2::dataformats::GlobalFwdTrack, 4> mchTrackExtrap = {
            PropagateToZMCH(mchTrack, zRefPlane[0]),
            PropagateToZMCH(mchTrack, zRefPlane[1]),
            PropagateToZMCH(mchTrack, zRefPlane[2]),
            PropagateToZMCH(mchTrack, zRefPlane[3])
        };

        // inner loop over collisions
        for (auto& [collisionIndex2, collisionInfo2] : collisionInfos) {
          auto const& collision2 = collisions.rawIteratorAt(collisionIndex2);

          // inner loop over MFT tracks
          for (auto mftIndex : collisionInfo2.mftTracks) {
            auto const& mftTrack = mftTracks.rawIteratorAt(mftIndex);

            if (!mftTrack.has_collision())
              continue;

            auto collisionMFT = collisions.rawIteratorAt(mftTrack.collisionId());
            int64_t bcMFT = bcs.rawIteratorAt(collisionMFT.bcId()).globalBC();

            bool sameEvent = (bcMFT == bcMFTmatched);
            bool mixedEvent = IsMixedEvent(collisionInfo1, collisionInfo2);

            if (!sameEvent && !mixedEvent)
              continue;

            bool isGoodMFT = IsGoodMFT(mftTrack, fTrackChi2MftUp, fTrackNClustMftLow);
            if (!isGoodMFT) continue;

            std::array<o2::dataformats::GlobalFwdTrack, 4> mftTrackExtrap = {
                PropagateToZMFT(mftTrack, mchTrackExtrap[1].getP(), zRefPlane[0]),
                PropagateToZMFT(mftTrack, mchTrackExtrap[1].getP(), zRefPlane[1]),
                PropagateToZMFT(mftTrack, mchTrackExtrap[1].getP(), zRefPlane[2]),
                PropagateToZMFT(mftTrack, mchTrackExtrap[1].getP(), zRefPlane[3])
            };

            std::array<std::array<double, 2>, 4> xPos = {{
                {{mchTrackExtrap[0].getX(), mftTrackExtrap[0].getX()}},
                {{mchTrackExtrap[1].getX(), mftTrackExtrap[1].getX()}},
                {{mchTrackExtrap[2].getX(), mftTrackExtrap[2].getX()}},
                {{mchTrackExtrap[3].getX(), mftTrackExtrap[3].getX()}}
            }};

            std::array<std::array<double, 2>, 4> yPos = {{
                {{mchTrackExtrap[0].getY(), mftTrackExtrap[0].getY()}},
                {{mchTrackExtrap[1].getY(), mftTrackExtrap[1].getY()}},
                {{mchTrackExtrap[2].getY(), mftTrackExtrap[2].getY()}},
                {{mchTrackExtrap[3].getY(), mftTrackExtrap[3].getY()}}
            }};

            std::array<std::array<double, 2>, 4> thetax = {{
                {{std::atan2(mchTrackExtrap[0].getPx(), -1.0 * mchTrackExtrap[0].getPz()) * 180 / TMath::Pi(),
                  std::atan2(mftTrackExtrap[0].getPx(), -1.0 * mftTrackExtrap[0].getPz()) * 180 / TMath::Pi()}},
                {{std::atan2(mchTrackExtrap[1].getPx(), -1.0 * mchTrackExtrap[1].getPz()) * 180 / TMath::Pi(),
                  std::atan2(mftTrackExtrap[1].getPx(), -1.0 * mftTrackExtrap[1].getPz()) * 180 / TMath::Pi()}},
                {{std::atan2(mchTrackExtrap[2].getPx(), -1.0 * mchTrackExtrap[2].getPz()) * 180 / TMath::Pi(),
                  std::atan2(mftTrackExtrap[2].getPx(), -1.0 * mftTrackExtrap[2].getPz()) * 180 / TMath::Pi()}},
                {{std::atan2(mchTrackExtrap[3].getPx(), -1.0 * mchTrackExtrap[3].getPz()) * 180 / TMath::Pi(),
                  std::atan2(mftTrackExtrap[3].getPx(), -1.0 * mftTrackExtrap[3].getPz()) * 180 / TMath::Pi()}}
            }};

            std::array<std::array<double, 2>, 4> thetay = {{
                {{std::atan2(mchTrackExtrap[0].getPy(), -1.0 * mchTrackExtrap[0].getPz()) * 180 / TMath::Pi(),
                  std::atan2(mftTrackExtrap[0].getPy(), -1.0 * mftTrackExtrap[0].getPz()) * 180 / TMath::Pi()}},
                {{std::atan2(mchTrackExtrap[1].getPy(), -1.0 * mchTrackExtrap[1].getPz()) * 180 / TMath::Pi(),
                  std::atan2(mftTrackExtrap[1].getPy(), -1.0 * mftTrackExtrap[1].getPz()) * 180 / TMath::Pi()}},
                {{std::atan2(mchTrackExtrap[2].getPy(), -1.0 * mchTrackExtrap[2].getPz()) * 180 / TMath::Pi(),
                  std::atan2(mftTrackExtrap[2].getPy(), -1.0 * mftTrackExtrap[2].getPz()) * 180 / TMath::Pi()}},
                {{std::atan2(mchTrackExtrap[3].getPy(), -1.0 * mchTrackExtrap[3].getPz()) * 180 / TMath::Pi(),
                  std::atan2(mftTrackExtrap[3].getPy(), -1.0 * mftTrackExtrap[3].getPz()) * 180 / TMath::Pi()}}
            }};

            for (int i = 0; i < 4; i++) {
              if (sameEvent) {
                //std::cout << std::format("[TOTO1] X at abosrber end: {:0.2f} {:0.2f}", xPos[3][0], xPos[3][1]) << std::endl;
                //std::cout << std::format("[TOTO2] Fill({:0.2f}, {:0.2f})", std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]) << std::endl;
                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dx_vs_x"])->Fill(std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dx_vs_y"])->Fill(std::fabs(yPos[i][1]), xPos[i][0] - xPos[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dy_vs_x"])->Fill(std::fabs(xPos[i][1]), yPos[i][0] - yPos[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dy_vs_y"])->Fill(std::fabs(yPos[i][1]), yPos[i][0] - yPos[i][1]);

                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dthetax_vs_x"])->Fill(std::fabs(xPos[i][1]), thetax[i][0] - thetax[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dthetax_vs_y"])->Fill(std::fabs(yPos[i][1]), thetax[i][0] - thetax[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dthetax_vs_thetax"])->Fill(std::fabs(thetax[i][1]), thetax[i][0] - thetax[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dthetay_vs_x"])->Fill(std::fabs(xPos[i][1]), thetay[i][0] - thetay[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dthetay_vs_y"])->Fill(std::fabs(yPos[i][1]), thetay[i][0] - thetay[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistos[i][quadrant]["dthetay_vs_thetay"])->Fill(std::fabs(thetay[i][1]), thetay[i][0] - thetay[i][1]);
              }
              if (mixedEvent) {
                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dx_vs_x"])->Fill(std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dx_vs_y"])->Fill(std::fabs(yPos[i][1]), xPos[i][0] - xPos[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dy_vs_x"])->Fill(std::fabs(xPos[i][1]), yPos[i][0] - yPos[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dy_vs_y"])->Fill(std::fabs(yPos[i][1]), yPos[i][0] - yPos[i][1]);

                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dthetax_vs_x"])->Fill(std::fabs(xPos[i][1]), thetax[i][0] - thetax[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dthetax_vs_y"])->Fill(std::fabs(yPos[i][1]), thetax[i][0] - thetax[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dthetax_vs_thetax"])->Fill(std::fabs(thetax[i][1]), thetax[i][0] - thetax[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dthetay_vs_x"])->Fill(std::fabs(xPos[i][1]), thetay[i][0] - thetay[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dthetay_vs_y"])->Fill(std::fabs(yPos[i][1]), thetay[i][0] - thetay[i][1]);
                std::get<std::shared_ptr<TH2>>(alignmentHistosMixedEvents[i][quadrant]["dthetay_vs_thetay"])->Fill(std::fabs(thetay[i][1]), thetay[i][0] - thetay[i][1]);
              }
            }
          }
        }
      }
    }
  }


  void FillDCAPlots(MyEvents const& collisions,
                          aod::BCsWithTimestamps const& bcs,
                          MyMuonsWithCov const& muonTracks,
                          MyMFTs const& mftTracks,
                          const std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    constexpr double doubleMin = std::numeric_limits<double>::min();
    constexpr double doubleMax = std::numeric_limits<double>::max();

    // outer loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {
      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);
      int64_t bc1 = bcs.rawIteratorAt(collision1.bcId()).globalBC();

      // loop over muon tracks
      for (auto mchIndex : collisionInfo1.mchTracks) {
        auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

        int quadrant = GetQuadrant(mchTrack);

        bool isGoodMuon = IsGoodMuon(mchTrack, collision1, fTrackChi2MchUp, 20.0, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp);
        if (!isGoodMuon) continue;

        // inner loop over collisions
        for (auto& [collisionIndex2, collisionInfo2] : collisionInfos) {
          auto const& collision2 = collisions.rawIteratorAt(collisionIndex2);

          int64_t bc2 = bcs.rawIteratorAt(collision2.bcId()).globalBC();

          bool sameEvent = (bc1 == bc2);
          bool mixedEvent = IsMixedEvent(collisionInfo1, collisionInfo2);

          if (!sameEvent && !mixedEvent)
            continue;

          auto mchTrackAtDCA = VarManager::PropagateMuon(mchTrack, collision2, toDCA);
          double dcax = mchTrackAtDCA.getX() - collision2.posX();
          double dcay = mchTrackAtDCA.getY() - collision2.posY();

          if (sameEvent) {
            //std::cout << std::format("[TOTO1] X at abosrber end: {:0.2f} {:0.2f}", xPos[3][0], xPos[3][1]) << std::endl;
            //std::cout << std::format("[TOTO2] Fill({:0.2f}, {:0.2f})", std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]) << std::endl;
            std::get<std::shared_ptr<TH1>>(dcaHistos[1][quadrant]["DCA_x"])->Fill(dcax);
            std::get<std::shared_ptr<TH1>>(dcaHistos[1][quadrant]["DCA_y"])->Fill(dcay);
          }
          if (mixedEvent) {
            //std::cout << std::format("[TOTO1] X at abosrber end: {:0.2f} {:0.2f}", xPos[3][0], xPos[3][1]) << std::endl;
            //std::cout << std::format("[TOTO2] Fill({:0.2f}, {:0.2f})", std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]) << std::endl;
            std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[1][quadrant]["DCA_x"])->Fill(dcax);
            std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[1][quadrant]["DCA_y"])->Fill(dcay);
          }
        }
      }

      // outer loop over global muon tracks
      for (auto& [mchIndex, globalTracksVector] : collisionInfo1.globalMuonTracks) {
        auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[0]);
        const auto& mchTrack = muonTrack.template matchMCHTrack_as<MyMuonsWithCov>();
        const auto& mftMatchedTrack = muonTrack.template matchMFTTrack_as<MyMFTs>();

        // loop over MFT tracks
        for (auto mftIndex : collisionInfo1.mftTracks) {
          auto const& mftTrack = mftTracks.rawIteratorAt(mftIndex);

          if (mftTrack.trackTime() != mftMatchedTrack.trackTime())
            continue;

          int quadrant = GetQuadrant(mftTrack);
          if (quadrant < 0) continue;

          bool isGoodMFT = IsGoodMFT(mftTrack, fTrackChi2MftUp, fTrackNClustMftLow);
          if (!isGoodMFT) continue;

          // inner loop over collisions
          for (auto& [collisionIndex2, collisionInfo2] : collisionInfos) {
            auto const& collision2 = collisions.rawIteratorAt(collisionIndex2);

            int64_t bc2 = bcs.rawIteratorAt(collision2.bcId()).globalBC();

            bool sameEvent = (bc1 == bc2);
            bool mixedEvent = IsMixedEvent(collisionInfo1, collisionInfo2);

            if (!sameEvent && !mixedEvent)
              continue;


            auto mftTrackAtDCA = PropagateMftToDCA(mftTrack, collision2);
            double dcax = mftTrackAtDCA.getX() - collision2.posX();
            double dcay = mftTrackAtDCA.getY() - collision2.posY();

            if (sameEvent) {
              //std::cout << std::format("[TOTO1] X at abosrber end: {:0.2f} {:0.2f}", xPos[3][0], xPos[3][1]) << std::endl;
              //std::cout << std::format("[TOTO2] Fill({:0.2f}, {:0.2f})", std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]) << std::endl;
              //std::cout << "[TOTO] DCA_x: " << std::get<std::shared_ptr<TH1>>(dcaHistos[0][quadrant]["DCA_x"]).get() << std::endl;
              std::get<std::shared_ptr<TH1>>(dcaHistos[0][quadrant]["DCA_x"])->Fill(dcax);
              std::get<std::shared_ptr<TH1>>(dcaHistos[0][quadrant]["DCA_y"])->Fill(dcay);
            }

            if (mixedEvent) {
              //std::cout << std::format("[TOTO1] X at abosrber end: {:0.2f} {:0.2f}", xPos[3][0], xPos[3][1]) << std::endl;
              //std::cout << std::format("[TOTO2] Fill({:0.2f}, {:0.2f})", std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]) << std::endl;
              std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[0][quadrant]["DCA_x"])->Fill(dcax);
              std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[0][quadrant]["DCA_y"])->Fill(dcay);
            }
          }
        }
      }
    }
  }

  void processQA(MyEvents const& collisions,
                 aod::BCsWithTimestamps const& bcs,
                 MyMuonsWithCov const& muonTracks,
                 MyMFTs const& mftTracks)
  {
    auto bc = bcs.begin();
    if (mRunNumber != bc.runNumber()) {
      initCCDB(bc);
       grpmag = ccdbManager->getForTimeStamp<o2::parameters::GRPMagField>(grpmagPath, bc.timestamp());
      if (grpmag != nullptr) {
        LOGF(info, "Init field from GRP");
        o2::base::Propagator::initFieldFromGRP(grpmag);
      }
      LOGF(info, "Set field for muons");
      VarManager::SetupMuonMagField();
      mRunNumber = bc.runNumber();
    }

    std::map<uint64_t, CollisionInfo> collisionInfos;
    InitCollisions(collisions, bcs, muonTracks, mftTracks, collisionInfos);

    FillMuonPlots(collisions, bcs, muonTracks, collisionInfos);

    FillDimuonPlots(collisions, muonTracks, collisionInfos);

    FillDCAPlots(collisions, bcs, muonTracks, mftTracks, collisionInfos);
    FillAlignmentPlots(collisions, bcs, muonTracks, mftTracks, collisionInfos);
  }

  PROCESS_SWITCH(qaMuon, processQA, "process qa", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<qaMuon>(cfgc)};
};
