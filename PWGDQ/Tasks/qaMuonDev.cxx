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
#include <algorithm>
#include <random>

using namespace o2;
using namespace o2::framework;
using namespace o2::aod;

using namespace o2::aod::rctsel;

using MyReducedMuons = soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtra, aod::ReducedMuonsCov>;
using MyReducedEvents = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended>;
using MyReducedEventsVtxCov = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>;

using MyCollisions = aod::Collisions;
using MyBCs = soa::Join<aod::BCs, aod::Timestamps>;
using MyEvents = soa::Join<aod::Collisions, aod::EvSels>;
using MyMuonsWithCov = soa::Join<aod::FwdTracks, aod::FwdTracksCov>;
//using MyMuonsWithCov = aod::FwdTracks;
using MyMFTs = aod::MFTTracks;
using MyMFTCovariances = aod::MFTTracksCov;

using MyCollision = MyCollisions::iterator;
using MyBC = MyBCs::iterator;
using MyMUON = MyMuonsWithCov::iterator;
using MyMFT = MyMFTs::iterator;
using MyMFTCovariance = MyMFTCovariances::iterator;

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

std::unordered_map<int, std::vector<int64_t>> map_mfttracks;
std::unordered_map<int, std::vector<int64_t>> map_muontracks;
std::unordered_map<int, bool> map_collisions;
std::unordered_map<int, bool> map_has_mfttracks_collisions;
std::unordered_map<int, bool> map_has_muontracks_collisions;
std::unordered_map<int, float> map_vtxz;
std::unordered_map<int, int> map_nmfttrack;

const int fgNCh = 10;
const int fgNDetElemCh[fgNCh] = {4, 4, 4, 4, 18, 18, 26, 26, 26, 26};
const int fgSNDetElemCh[fgNCh + 1] = {0, 4, 8, 12, 16, 34, 52, 78, 104, 130, 156};

constexpr double muonMass = 0.1056584;
constexpr double muonMass2 = muonMass * muonMass;

// constexpr static uint32_t gkMuonDCAFillMapWithCov = VarManager::ObjTypes::ReducedMuon | VarManager::ObjTypes::ReducedMuonExtra | VarManager::ObjTypes::ReducedMuonCov | VarManager::ObjTypes::MuonDCA;

constexpr static int toVertex = 0; //VarManager::kToVertex;
constexpr static int toDCA = 1; //VarManager::kToDCA;
constexpr static int toRabs = 2; //VarManager::kToRabs;
constexpr static int toMFT = 3;

constexpr double firstMFTPlaneZ = o2::mft::constants::mft::LayerZCoordinate()[0];
constexpr double lastMFTPlaneZ = o2::mft::constants::mft::LayerZCoordinate()[9];
constexpr double firstMCHPlaneZ = -526.16;

static o2::globaltracking::MatchGlobalFwd sExtrap;

using o2::dataformats::GlobalFwdTrack;
using o2::track::TrackParCovFwd;
typedef std::function<double(const GlobalFwdTrack& mchtrack, const TrackParCovFwd& mfttrack)> MatchingFunc_t;
std::map<std::string, MatchingFunc_t> mMatchingFunctionMap; ///< MFT-MCH Matching function

using SMatrix55Std = ROOT::Math::SMatrix<double, 5>;
using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;

using SVector2 = ROOT::Math::SVector<double, 2>;
using SVector4 = ROOT::Math::SVector<double, 4>;
using SVector5 = ROOT::Math::SVector<double, 5>;

using SMatrix44 = ROOT::Math::SMatrix<double, 4>;
using SMatrix45 = ROOT::Math::SMatrix<double, 4, 5>;
using SMatrix54 = ROOT::Math::SMatrix<double, 5, 4>;
using SMatrix22 = ROOT::Math::SMatrix<double, 2>;
using SMatrix25 = ROOT::Math::SMatrix<double, 2, 5>;
using SMatrix52 = ROOT::Math::SMatrix<double, 5, 2>;

//_________________________________________________________________________________________

int getChamberIndex(int deId)
{
  return (deId / 100) - 1;
}

int getNumDEinChamber(int chIndex)
{
  int nDE = 0;
  switch (chIndex) {
    case 0:
    case 1:
    case 2:
    case 3:
      nDE = 4;
      break;
    case 4:
    case 5:
      nDE = 18;
      break;
    case 6:
    case 7:
    case 8:
    case 9:
      nDE = 26;
      break;
    default:
      break;
  }
  return nDE;
}

int getNumDE()
{
  static int nDE = -1;
  if (nDE < 0) {
    for (int c = 0; c < 10; c++) {
      nDE += getNumDEinChamber(c);
    }
  }

  return nDE;
}


std::pair<int, int> getDEindexInChamber(int deId)
{
  std::pair<int, int> result = std::make_pair(int(-1), int(-1));
  int nDE = getNumDEinChamber(getChamberIndex(deId));
  if (nDE == 0) {
    return result;
  }
  // number of detectors in one half chamber
  int nDEhc = nDE / 2;

  // detector index within the chamber
  int idx = (deId - 100) % 100;
  // minimum and maximum detector indexes for the L side
  int lMin = (nDEhc + 1) / 2;
  int lMax = lMin + nDEhc - 1;
  if (idx >= lMin && idx <= lMax) {
    // detector on the L side, simply sibtract lMin
    result.second = 0;
    result.first = idx - lMin;
  } else {
    // detector on the R side, compute index separately above and below middle horizontal axis
    result.second = 1;
    if (idx > lMax) {
      idx -= nDE;
    }
    result.first = lMin - idx - 1;
  }
  return result;
}

int getChamberOffset(int chIndex)
{
  int offset = 0;
  for (int c = 0; c < chIndex; c++) {
    offset += getNumDEinChamber(c);
  }
  return offset;
}

int getDEindex(int deId)
{
  auto idx = getDEindexInChamber(deId);
  if (idx.first < 0 || idx.second < 0) {
    return -1;
  }
  int offset = getChamberOffset(getChamberIndex(deId));

  // number of detectors in one half chamber
  int nDE = getNumDEinChamber(getChamberIndex(deId));
  if (idx.second > 0) {
    idx.first += nDE / 2;
  }

  return idx.first + offset;
}

//_________________________________________________________________________________________

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

  ////   Variables for alignment corrections
  Configurable<bool> fEnableMFTAlignmentCorrections{"cfgEnableMFTAlignmentCorrections", false, ""};
  Configurable<bool> fEnableMCHAlignmentCorrections{"cfgEnableMCHAlignmentCorrections", false, ""};
  //Configurable<float> fVertexZshift{"cfgVertexZshift", 0.f, "Correction to the vertex z position"};
  Configurable<float> fVertexZshift{"cfgVertexZshift", 0.201f, "Correction to the vertex z position"};
  //Configurable<float> fVertexZshift{"cfgVertexZshift", 0.02f, "Correction to the vertex z position"};

  ///    Variables to event mixing criteria
  //Configurable<float> fSaveMixedMatchingParamsRate{"cfgSaveMixedMatchingParamsRate", 0.002f, ""};
  Configurable<int> fEventMaxDeltaNMFT{"cfgEventMaxDeltaNMFT", 10, ""};
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

  o2::aod::rctsel::RCTFlagsChecker rctChecker{"CBT_muon_glo", false, false, true};

  double mBzAtMftCenter{ 0 };

  HistogramRegistry registry{"registry", {}};
  HistogramRegistry registryDCA{"registryDCA", {}};
  HistogramRegistry registryResiduals{"registryResiduals", {}};
  HistogramRegistry registryResidualsMFT{"registryResidualsMFT", {}};
  HistogramRegistry registryResidualsMCH{"registryResidualsMCH", {}};
  HistogramRegistry registryMatching{"registryMatching", {}};

  std::array<double, 3> zRefPlane{
      firstMFTPlaneZ,
      lastMFTPlaneZ,
      firstMCHPlaneZ
  };
  std::vector<std::pair<std::string, double>> referencePlanes{
      {"MFT-begin", 10.0},
      {"MFT-end", 15.0},
      {"MCH-begin", 100.0}
  };
  std::array<std::string, 4> quadrants{"Q0", "Q1", "Q2", "Q3"};

  double mMatchingPlaneZ{ lastMFTPlaneZ };


  std::array<std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 3>, 4>, 2> dcaHistos;
  std::array<std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 3>, 4>, 2> dcaHistosMixedEvents;

  std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 4>, 6> trackResidualsHistos;
  std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 4>, 6> trackResidualsHistosMixedEvents;

  std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 10>, 4> residualsHistos;
  std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 10>, 4> residualsHistosMixedEvents;

  std::array<std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 10>, 2>, 2> residualsHistosPerDE;
  std::array<std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 10>, 2>, 2> residualsHistosPerDEMixedEvents;

  std::array<std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 10>, 2>, 2> mchResidualsHistosPerDE;
  std::array<std::array<std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 10>, 2>, 2> mchResidualsHistosPerDEMixedEvents;

  std::unordered_map<std::string, o2::framework::HistPtr> matchingHistos;
  std::unordered_map<std::string, o2::framework::HistPtr> matchingHistosRealigned;

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
    std::map<std::string, std::string> metadata;
    auto soreor = o2::ccdb::BasicCCDBManager::getRunDuration(ccdbApi, mRunNumber);
    auto ts = soreor.first;
    auto grpmag = ccdbApi.retrieveFromTFileAny<o2::parameters::GRPMagField>(grpmagPath, metadata, ts);
    o2::base::Propagator::initFieldFromGRP(grpmag);
    if (!o2::base::GeometryManager::isGeometryLoaded()) {
      ccdbManager->get<TGeoManager>(geoPath);
    }
    o2::mch::TrackExtrap::setField();
    fieldB = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
    mBzAtMftCenter = fieldB->getBz(centerMFT);
    //std::cout << "fieldB: " << (void*)fieldB << std::endl;
  }

  void CreateDCAHistos()
  {
    float mftLadderWidth = 1.7;
    AxisSpec dcaxMFTAxis = {400, -0.5, 0.5, "DCA_{x} (cm)"};
    AxisSpec dcayMFTAxis = {400, -0.5, 0.5, "DCA_{y} (cm)"};
    AxisSpec dcaxMCHAxis = {400, -10.0, 10.0, "DCA_{x} (cm)"};
    AxisSpec dcayMCHAxis = {400, -10.0, 10.0, "DCA_{y} (cm)"};
    AxisSpec dcazAxis = {20, -10.0, 10.0, "v_{z} (cm)"};
    AxisSpec txAxis = {30, -mftLadderWidth * 15.f / 2.f, mftLadderWidth * 15.f / 2.f, "track_{x} (cm)"};
    AxisSpec tyAxis = {20, -10.f, 10.f, "track_{y} (cm)"};
    AxisSpec vxAxis = {400, -0.5, 0.5, "vtx_{x} (cm)"};
    AxisSpec vyAxis = {400, -0.5, 0.5, "vtx_{y} (cm)"};
    AxisSpec vzAxis = {1000, -10.0, 10.0, "vtx_{z} (cm)"};
    AxisSpec phiAxis = {36, -180.0, 180.0, "#phi (degrees)"};
    AxisSpec nMftClustersAxis = {6, 5, 11, "# of clusters"};
    AxisSpec mftTrackTypeAxis = {2, 0, 2, "track type"};
    AxisSpec trackChargeSignAxis = {2, 0, 0, "sign"};
    AxisSpec zshiftAxis = {21, -5.25, 5.25, "z shift (mm)"};
    AxisSpec layersPatternAxis = {1024, 0, 1024, "layers pattern"};

    dcaHistos[0][0][0]["vertex_y_vs_x"] = registryDCA.add("alignment/DCA/vertex_y_vs_x", std::format("Vertex y vs. x").c_str(), {HistType::kTH2F, {vxAxis, vyAxis}});
    dcaHistos[0][0][0]["vertex_z"] = registryDCA.add("alignment/DCA/vertex_z", std::format("Vertex z").c_str(), {HistType::kTH1F, {vzAxis}});
    dcaHistos[0][0][0]["nTracksMFT"] = registryDCA.add("alignment/DCA/nTracksMFT", std::format("Number of MFT tracks per collision").c_str(), {HistType::kTH1F, {{100, 0, 1000, "# of MFT tracks"}}});
    dcaHistos[0][0][0]["DCA_y_vs_x"] = registryDCA.add("alignment/DCA/MFT/DCA_y_vs_x", std::format("DCA y vs. x").c_str(), {HistType::kTH2F, {dcaxMFTAxis, dcayMFTAxis}});
    dcaHistos[0][0][0]["DCA_x_vs_phi"] = registryDCA.add("alignment/DCA/MFT/DCA_x_vs_phi", std::format("DCA(x) vs. #phi").c_str(), {HistType::kTH2F, {phiAxis, dcaxMFTAxis}});
    dcaHistos[0][0][0]["DCA_y_vs_phi"] = registryDCA.add("alignment/DCA/MFT/DCA_y_vs_phi", std::format("DCA(y) vs. #phi").c_str(), {HistType::kTH2F, {phiAxis, dcayMFTAxis}});
    dcaHistos[0][0][0]["DCA_x_full"] = registryDCA.add("alignment/DCA/MFT/DCA_x_vs_vz_tx_ty_nclus_ttype",
        "DCA(x) vs. vz, tx, ty, nclus, trackType",
        HistType::kTHnSparseF,
        {dcaxMFTAxis, dcazAxis, txAxis, tyAxis, nMftClustersAxis, mftTrackTypeAxis});
    dcaHistos[0][0][0]["DCA_y_full"] = registryDCA.add("alignment/DCA/MFT/DCA_y_vs_vz_tx_ty_nclus_ttype",
        "DCA(y) vs. vz, tx, ty, nclus, trackType",
        HistType::kTHnSparseF,
        {dcayMFTAxis, dcazAxis, txAxis, tyAxis, nMftClustersAxis, mftTrackTypeAxis});
    dcaHistos[0][0][0]["layers"] = registryDCA.add("alignment/DCA/MFT/layers",
            "Layers pattern vs. tx, ty, nclus, trackType",
            HistType::kTHnSparseF,
            {layersPatternAxis, txAxis, tyAxis, nMftClustersAxis, mftTrackTypeAxis});
    //dcaHistos[0][0][0]["DCA_x_vs_phi_alt"] = registryDCA.add("alignment/DCA/MFT/DCA_x_vs_phi_alt", std::format("DCA(x) vs. #phi (alt. extrap.)").c_str(), {HistType::kTH2F, {phiAxis, dcaxMFTAxis}});
    //dcaHistos[0][0][0]["DCA_y_vs_phi_alt"] = registryDCA.add("alignment/DCA/MFT/DCA_y_vs_phi_alt", std::format("DCA(y) vs. #phi (alt. extrap.)").c_str(), {HistType::kTH2F, {phiAxis, dcayMFTAxis}});
    dcaHistos[0][0][0]["DCA_x_vs_phi_vs_zshift"] = registryDCA.add("alignment/DCA/MFT/DCA_x_vs_phi_vs_zshift", std::format("DCA(x) vs. #phi vs. z shift").c_str(), {HistType::kTH3F, {zshiftAxis, phiAxis, dcaxMFTAxis}});
    dcaHistos[0][0][0]["DCA_y_vs_phi_vs_zshift"] = registryDCA.add("alignment/DCA/MFT/DCA_y_vs_phi_vs_zshift", std::format("DCA(y) vs. #phi vs. z shift").c_str(), {HistType::kTH3F, {zshiftAxis, phiAxis, dcayMFTAxis}});

    auto h3 = std::get<std::shared_ptr<TH3>>(dcaHistos[0][0][0]["DCA_x_vs_phi_vs_zshift"]);
    for (int bin = 1; bin <= h3->GetXaxis()->GetNbins(); bin++) {
      std::cout << std::format("Bin #{} -> {}", bin, h3->GetXaxis()->GetBinCenter(bin)) << std::endl;
    }

    for (size_t j = 0; j < quadrants.size(); j++) {
      const auto& quadrant = quadrants[j];
      std::string histPath = std::string("alignment/DCA/MFT/") + quadrant + "/";
      dcaHistos[0][j][0]["DCA_x"] = registryDCA.add((histPath + "DCA_x").c_str(), std::format("DCA(x) - {}", quadrant).c_str(), {HistType::kTH1F, {dcaxMFTAxis}});
      dcaHistos[0][j][1]["DCA_x"] = registryDCA.add((histPath + "DCA_x_pos").c_str(), std::format("DCA(x) - {} charge > 0", quadrant).c_str(), {HistType::kTH1F, {dcaxMFTAxis}});
      dcaHistos[0][j][2]["DCA_x"] = registryDCA.add((histPath + "DCA_x_neg").c_str(), std::format("DCA(x) - {} charge < 0", quadrant).c_str(), {HistType::kTH1F, {dcaxMFTAxis}});
      dcaHistos[0][j][0]["DCA_y"] = registryDCA.add((histPath + "DCA_y").c_str(), std::format("DCA(y) - {}", quadrant).c_str(), {HistType::kTH1F, {dcayMFTAxis}});
      dcaHistos[0][j][1]["DCA_y"] = registryDCA.add((histPath + "DCA_y_pos").c_str(), std::format("DCA(y) - {} charge > 0", quadrant).c_str(), {HistType::kTH1F, {dcayMFTAxis}});
      dcaHistos[0][j][2]["DCA_y"] = registryDCA.add((histPath + "DCA_y_neg").c_str(), std::format("DCA(y) - {} charge < 0", quadrant).c_str(), {HistType::kTH1F, {dcayMFTAxis}});
      dcaHistos[0][j][0]["DCA_x_vs_z"] = registryDCA.add((histPath + "DCA_x_vs_z").c_str(), std::format("DCA(x) vs. z - {}", quadrant).c_str(), {HistType::kTH2F, {dcazAxis, dcaxMFTAxis}});
      dcaHistos[0][j][0]["DCA_y_vs_z"] = registryDCA.add((histPath + "DCA_y_vs_z").c_str(), std::format("DCA(y) vs. z - {}", quadrant).c_str(), {HistType::kTH2F, {dcazAxis, dcayMFTAxis}});
      //dcaHistos[0][j][0]["DCA_x_vs_z_alt"] = registryDCA.add((histPath + "DCA_x_vs_z_alt").c_str(), std::format("DCA(x) vs. z - {} (alt. extrap.)", quadrant).c_str(), {HistType::kTH2F, {dcazAxis, dcaxMFTAxis}});
      //dcaHistos[0][j][0]["DCA_y_vs_z_alt"] = registryDCA.add((histPath + "DCA_y_vs_z_alt").c_str(), std::format("DCA(y) vs. z - {} (alt. extrap.)", quadrant).c_str(), {HistType::kTH2F, {dcazAxis, dcayMFTAxis}});
      dcaHistos[0][j][0]["DCA_x_vs_track_x"] = registryDCA.add((histPath + "DCA_x_vs_track_x").c_str(), std::format("DCA(x) vs. track x - {}", quadrant).c_str(), {HistType::kTH2F, {txAxis, dcaxMFTAxis}});
      dcaHistos[0][j][0]["DCA_x_vs_track_y"] = registryDCA.add((histPath + "DCA_x_vs_track_y").c_str(), std::format("DCA(x) vs. track y - {}", quadrant).c_str(), {HistType::kTH2F, {tyAxis, dcaxMFTAxis}});
      dcaHistos[0][j][0]["DCA_y_vs_track_x"] = registryDCA.add((histPath + "DCA_y_vs_track_x").c_str(), std::format("DCA(y) vs. track x - {}", quadrant).c_str(), {HistType::kTH2F, {txAxis, dcayMFTAxis}});
      dcaHistos[0][j][0]["DCA_y_vs_track_y"] = registryDCA.add((histPath + "DCA_y_vs_track_y").c_str(), std::format("DCA(y) vs. track y - {}", quadrant).c_str(), {HistType::kTH2F, {tyAxis, dcayMFTAxis}});

      histPath = std::string("alignment/DCA/MCH/") + quadrant + "/";
      dcaHistos[1][j][0]["DCA_x"] = registryDCA.add((histPath + "DCA_x").c_str(), std::format("DCA(x) - {}", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistos[1][j][1]["DCA_x"] = registryDCA.add((histPath + "DCA_x_pos").c_str(), std::format("DCA(x) - {} charge > 0", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistos[1][j][2]["DCA_x"] = registryDCA.add((histPath + "DCA_x_neg").c_str(), std::format("DCA(x) - {} charge < 0", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistos[1][j][0]["DCA_y"] = registryDCA.add((histPath + "DCA_y").c_str(), std::format("DCA(y) - {}", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});
      dcaHistos[1][j][1]["DCA_y"] = registryDCA.add((histPath + "DCA_y_pos").c_str(), std::format("DCA(y) - {} charge > 0", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});
      dcaHistos[1][j][2]["DCA_y"] = registryDCA.add((histPath + "DCA_y_neg").c_str(), std::format("DCA(y) - {} charge < 0", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});

      histPath = std::string("alignment/mixed-events/DCA/MFT/") + quadrant + "/";
      dcaHistosMixedEvents[0][j][0]["DCA_x"] = registryDCA.add((histPath + "DCA_x").c_str(), std::format("DCA(x) - {}", quadrant).c_str(), {HistType::kTH1F, {dcaxMFTAxis}});
      dcaHistosMixedEvents[0][j][1]["DCA_x"] = registryDCA.add((histPath + "DCA_x_pos").c_str(), std::format("DCA(x) - {} charge > 0", quadrant).c_str(), {HistType::kTH1F, {dcaxMFTAxis}});
      dcaHistosMixedEvents[0][j][2]["DCA_x"] = registryDCA.add((histPath + "DCA_x_neg").c_str(), std::format("DCA(x) - {} charge < 0", quadrant).c_str(), {HistType::kTH1F, {dcaxMFTAxis}});
      dcaHistosMixedEvents[0][j][0]["DCA_y"] = registryDCA.add((histPath + "DCA_y").c_str(), std::format("DCA(y) - {}", quadrant).c_str(), {HistType::kTH1F, {dcayMFTAxis}});
      dcaHistosMixedEvents[0][j][1]["DCA_y"] = registryDCA.add((histPath + "DCA_y_pos").c_str(), std::format("DCA(y) - {} charge > 0", quadrant).c_str(), {HistType::kTH1F, {dcayMFTAxis}});
      dcaHistosMixedEvents[0][j][2]["DCA_y"] = registryDCA.add((histPath + "DCA_y_neg").c_str(), std::format("DCA(y) - {} charge < 0", quadrant).c_str(), {HistType::kTH1F, {dcayMFTAxis}});
      dcaHistosMixedEvents[0][j][0]["DCA_x_vs_z"] = registryDCA.add((histPath + "DCA_x_vs_z").c_str(), std::format("DCA(x) vs. z - {}", quadrant).c_str(), {HistType::kTH2F, {dcazAxis, dcaxMFTAxis}});
      dcaHistosMixedEvents[0][j][0]["DCA_y_vs_z"] = registryDCA.add((histPath + "DCA_y_vs_z").c_str(), std::format("DCA(y) vs. z - {}", quadrant).c_str(), {HistType::kTH2F, {dcazAxis, dcayMFTAxis}});

      histPath = std::string("alignment/mixed-events/DCA/MCH/") + quadrant + "/";
      dcaHistosMixedEvents[1][j][0]["DCA_x"] = registryDCA.add((histPath + "DCA_x").c_str(), std::format("DCA(x) - {}", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistosMixedEvents[1][j][1]["DCA_x"] = registryDCA.add((histPath + "DCA_x_pos").c_str(), std::format("DCA(x) - {} charge > 0", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistosMixedEvents[1][j][2]["DCA_x"] = registryDCA.add((histPath + "DCA_x_neg").c_str(), std::format("DCA(x) - {} charge < 0", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistosMixedEvents[1][j][0]["DCA_y"] = registryDCA.add((histPath + "DCA_y").c_str(), std::format("DCA(y) - {}", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});
      dcaHistosMixedEvents[1][j][1]["DCA_y"] = registryDCA.add((histPath + "DCA_y_pos").c_str(), std::format("DCA(y) - {} charge > 0", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});
      dcaHistosMixedEvents[1][j][2]["DCA_y"] = registryDCA.add((histPath + "DCA_y_neg").c_str(), std::format("DCA(y) - {} charge < 0", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});
    }
  }

  void CreateAlignementHistos()
  {
    AxisSpec dxAxis = {600, -30.0, 30.0, "#Delta x (cm)"};
    AxisSpec dyAxis = {600, -30.0, 30.0, "#Delta y (cm)"};
    AxisSpec thetaxAxis = {10, 0.0, 20.0, "#theta_{x} (degrees)"};
    AxisSpec dThetaxAxis = {500, -5.0, 5.0, "#Delta#theta_{x} (degrees)"};
    AxisSpec thetayAxis = {10, 0.0, 20.0, "#theta_{y} (degrees)"};
    AxisSpec dThetayAxis = {500, -5.0, 5.0, "#Delta#theta_{y} (degrees)"};
    AxisSpec phiAxis = {360, -180.0, 180.0, "#phi (degrees)"};
    AxisSpec dPhiAxis = {200, -20.0, 20.0, "#Delta#phi (degrees)"};

    for (size_t i = 0; i < referencePlanes.size(); i++) {
      const auto& refPLane = referencePlanes[i];
      AxisSpec xAxis = {10, 0, refPLane.second, "|x| (cm)"};
      AxisSpec yAxis = {10, 0, refPLane.second, "|y| (cm)"};
      for (size_t j = 0; j < quadrants.size(); j++) {
        const auto& quadrant = quadrants[j];
        std::string histPath = std::string("alignment/same-event/") + refPLane.first + "/" + quadrant + "/";
        trackResidualsHistos[i][j]["dx_vs_x"] = registry.add((histPath + "dx_vs_x").c_str(), std::format("#Delta x vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dxAxis}});
        trackResidualsHistos[i][j]["dx_vs_y"] = registry.add((histPath + "dx_vs_y").c_str(), std::format("#Delta x vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dxAxis}});
        trackResidualsHistos[i][j]["dy_vs_x"] = registry.add((histPath + "dy_vs_x").c_str(), std::format("#Delta y vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dyAxis}});
        trackResidualsHistos[i][j]["dy_vs_y"] = registry.add((histPath + "dy_vs_y").c_str(), std::format("#Delta y vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dyAxis}});

        trackResidualsHistos[i][j]["dthetax_vs_x"] = registry.add((histPath + "dthetax_vs_x").c_str(), std::format("#Delta #theta_x vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dThetaxAxis}});
        trackResidualsHistos[i][j]["dthetax_vs_y"] = registry.add((histPath + "dthetax_vs_y").c_str(), std::format("#Delta #theta_x vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dThetaxAxis}});
        trackResidualsHistos[i][j]["dthetax_vs_thetax"] = registry.add((histPath + "dthetax_vs_thetax").c_str(), std::format("#Delta #theta_x vs. |#theta_x| - {}", quadrant).c_str(), {HistType::kTH2F, {thetaxAxis, dThetaxAxis}});

        trackResidualsHistos[i][j]["dthetay_vs_x"] = registry.add((histPath + "dthetay_vs_x").c_str(), std::format("#Delta #theta_y vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dThetayAxis}});
        trackResidualsHistos[i][j]["dthetay_vs_y"] = registry.add((histPath + "dthetay_vs_y").c_str(), std::format("#Delta #theta_y vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dThetayAxis}});
        trackResidualsHistos[i][j]["dthetay_vs_thetay"] = registry.add((histPath + "dthetay_vs_thetay").c_str(), std::format("#Delta #theta_y vs. |#theta_y| - {}", quadrant).c_str(), {HistType::kTH2F, {thetayAxis, dThetayAxis}});

        // mixed events
        histPath = std::string("alignment/mixed-event/") + refPLane.first + "/" + quadrant + "/";
        trackResidualsHistosMixedEvents[i][j]["dx_vs_x"] = registry.add((histPath + "dx_vs_x").c_str(), std::format("#Delta x vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dxAxis}});
        trackResidualsHistosMixedEvents[i][j]["dx_vs_y"] = registry.add((histPath + "dx_vs_y").c_str(), std::format("#Delta x vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dxAxis}});
        trackResidualsHistosMixedEvents[i][j]["dy_vs_x"] = registry.add((histPath + "dy_vs_x").c_str(), std::format("#Delta y vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dyAxis}});
        trackResidualsHistosMixedEvents[i][j]["dy_vs_y"] = registry.add((histPath + "dy_vs_y").c_str(), std::format("#Delta y vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dyAxis}});

        trackResidualsHistosMixedEvents[i][j]["dthetax_vs_x"] = registry.add((histPath + "dthetax_vs_x").c_str(), std::format("#Delta #theta_x vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dThetaxAxis}});
        trackResidualsHistosMixedEvents[i][j]["dthetax_vs_y"] = registry.add((histPath + "dthetax_vs_y").c_str(), std::format("#Delta #theta_x vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dThetaxAxis}});
        trackResidualsHistosMixedEvents[i][j]["dthetax_vs_thetax"] = registry.add((histPath + "dthetax_vs_thetax").c_str(), std::format("#Delta #theta_x vs. |#theta_x| - {}", quadrant).c_str(), {HistType::kTH2F, {thetaxAxis, dThetaxAxis}});

        trackResidualsHistosMixedEvents[i][j]["dthetay_vs_x"] = registry.add((histPath + "dthetay_vs_x").c_str(), std::format("#Delta #theta_y vs. |x| - {}", quadrant).c_str(), {HistType::kTH2F, {xAxis, dThetayAxis}});
        trackResidualsHistosMixedEvents[i][j]["dthetay_vs_y"] = registry.add((histPath + "dthetay_vs_y").c_str(), std::format("#Delta #theta_y vs. |y| - {}", quadrant).c_str(), {HistType::kTH2F, {yAxis, dThetayAxis}});
        trackResidualsHistosMixedEvents[i][j]["dthetay_vs_thetay"] = registry.add((histPath + "dthetay_vs_thetay").c_str(), std::format("#Delta #theta_y vs. |#theta_y| - {}", quadrant).c_str(), {HistType::kTH2F, {thetayAxis, dThetayAxis}});
      }
    }

    residualsHistos[0][0]["dx_vs_chamber"] = registryResiduals.add("alignment/same-event/Residuals/dx_vs_chamber",
        "Cluster x residual vs. chamber, quadrant, chargeSign",
        {HistType::kTHnSparseF, {dxAxis, {10, 1, 11, "chamber"}, {4, 0, 4, "quadrant"}, {2, 0, 2, "sign"}}});
    residualsHistos[0][0]["dy_vs_chamber"] = registryResiduals.add("alignment/same-event/Residuals/dy_vs_chamber",
        "Cluster y residual vs. chamber, quadrant, chargeSign",
        {HistType::kTHnSparseF, {dyAxis, {10, 1, 11, "chamber"}, {4, 0, 4, "quadrant"}, {2, 0, 2, "sign"}}});

    residualsHistos[0][0]["dx_vs_de"] = registryResiduals.add("alignment/same-event/Residuals/dx_vs_de",
        "Cluster x residual vs. DE, quadrant, chargeSign",
        {HistType::kTHnSparseF, {dxAxis, {getNumDE(), 0, getNumDE(), "DE"}, {4, 0, 4, "quadrant"}, {2, 0, 2, "sign"}}});
    residualsHistos[0][0]["dy_vs_de"] = registryResiduals.add("alignment/same-event/Residuals/dy_vs_de",
        "Cluster y residual vs. DE, quadrant, chargeSign",
        {HistType::kTHnSparseF, {dyAxis, {getNumDE(), 0, getNumDE(), "DE"}, {4, 0, 4, "quadrant"}, {2, 0, 2, "sign"}}});

    for (size_t j = 0; j < quadrants.size(); j++) {
      AxisSpec xAxis = {20, 0, 200, "|x| (cm)"};
      AxisSpec yAxis = {10, 0, 200, "|y| (cm)"};
      const auto& quadrant = quadrants[j];
      for (int chamber = 0; chamber < 10; chamber++) {
        std::string histPath = std::string("alignment/same-event/Residuals/") + quadrant + "/CH" + std::to_string(chamber + 1) + "/";
        // Delta x at cluster
        residualsHistos[j][chamber]["dx_vs_x"] = registryResiduals.add((histPath + "dx_vs_x").c_str(), "Cluster x residual vs. x", {HistType::kTH2F, {xAxis, dxAxis}});
        residualsHistos[j][chamber]["dx_vs_y"] = registryResiduals.add((histPath + "dx_vs_y").c_str(), "Cluster x residual vs. y", {HistType::kTH2F, {yAxis, dxAxis}});
        residualsHistos[j][chamber]["dy_vs_x"] = registryResiduals.add((histPath + "dy_vs_x").c_str(), "Cluster y residual vs. x", {HistType::kTH2F, {xAxis, dyAxis}});
        residualsHistos[j][chamber]["dy_vs_y"] = registryResiduals.add((histPath + "dy_vs_y").c_str(), "Cluster y residual vs. y", {HistType::kTH2F, {yAxis, dyAxis}});

        // mixed events
        histPath = std::string("alignment/mixed-event/Residuals/") + quadrant + "/CH" + std::to_string(chamber + 1) + "/";
        // Delta x at cluster
        residualsHistosMixedEvents[j][chamber]["dx_vs_x"] = registryResiduals.add((histPath + "dx_vs_x").c_str(), "Cluster x residual vs. x", {HistType::kTH2F, {xAxis, dxAxis}});
        residualsHistosMixedEvents[j][chamber]["dx_vs_y"] = registryResiduals.add((histPath + "dx_vs_y").c_str(), "Cluster x residual vs. y", {HistType::kTH2F, {yAxis, dxAxis}});
        residualsHistosMixedEvents[j][chamber]["dy_vs_x"] = registryResiduals.add((histPath + "dy_vs_x").c_str(), "Cluster y residual vs. x", {HistType::kTH2F, {xAxis, dyAxis}});
        residualsHistosMixedEvents[j][chamber]["dy_vs_y"] = registryResiduals.add((histPath + "dy_vs_y").c_str(), "Cluster y residual vs. y", {HistType::kTH2F, {yAxis, dyAxis}});
      }
    }

    for (size_t i = 0; i < 2; i++) {
      std::string topBottom = (i == 0) ? "top" : "bottom";
      AxisSpec deAxis = {26, 0, 26, "DE index"};
      AxisSpec phiAxis = {16, -180, 180, "#phi (degrees)"};
      for (size_t j = 0; j < 2; j++) {
        std::string sign = (j == 0) ? "positive" : "negative";
        for (int chamber = 0; chamber < 10; chamber++) {
          std::string histPath = std::string("alignment/residuals/MFT_") + topBottom + "/" + sign + "/CH" + std::to_string(chamber + 1) + "/";
          // Delta x and y at cluster
          residualsHistosPerDE[i][j][chamber]["dx_vs_de"] = registryResidualsMFT.add((histPath + "dx_vs_de").c_str(), "Cluster x residual vs. DE index", {HistType::kTH2F, {deAxis, dxAxis}});
          residualsHistosPerDE[i][j][chamber]["dy_vs_de"] = registryResidualsMFT.add((histPath + "dy_vs_de").c_str(), "Cluster y residual vs. DE index", {HistType::kTH2F, {deAxis, dxAxis}});

          residualsHistosPerDE[i][j][chamber]["dx_vs_phi"] = registryResidualsMFT.add((histPath + "dx_vs_phi").c_str(), "Cluster x residual vs. cluster #phi", {HistType::kTH2F, {phiAxis, dxAxis}});
          residualsHistosPerDE[i][j][chamber]["dy_vs_phi"] = registryResidualsMFT.add((histPath + "dy_vs_phi").c_str(), "Cluster y residual vs. cluster #phi", {HistType::kTH2F, {phiAxis, dxAxis}});

          // mixed events
          histPath = std::string("alignment/mixed-events/residuals/MFT_") + topBottom + "/" + sign + "/CH" + std::to_string(chamber + 1) + "/";
          // Delta x and y at cluster
          residualsHistosPerDEMixedEvents[i][j][chamber]["dx_vs_de"] = registryResidualsMFT.add((histPath + "dx_vs_de").c_str(), "Cluster x residual vs. DE index", {HistType::kTH2F, {deAxis, dxAxis}});
          residualsHistosPerDEMixedEvents[i][j][chamber]["dy_vs_de"] = registryResidualsMFT.add((histPath + "dy_vs_de").c_str(), "Cluster y residual vs. DE index", {HistType::kTH2F, {deAxis, dxAxis}});

          residualsHistosPerDEMixedEvents[i][j][chamber]["dx_vs_phi"] = registryResidualsMFT.add((histPath + "dx_vs_phi").c_str(), "Cluster x residual vs. cluster #phi", {HistType::kTH2F, {phiAxis, dxAxis}});
          residualsHistosPerDEMixedEvents[i][j][chamber]["dy_vs_phi"] = registryResidualsMFT.add((histPath + "dy_vs_phi").c_str(), "Cluster y residual vs. cluster #phi", {HistType::kTH2F, {phiAxis, dxAxis}});
        }
      }
    }

    for (size_t i = 0; i < 2; i++) {
      std::string topBottom = (i == 0) ? "top" : "bottom";
      AxisSpec deAxis = {26, 0, 26, "DE index"};
      for (size_t j = 0; j < 2; j++) {
        std::string sign = (j == 0) ? "positive" : "negative";
        for (int chamber = 0; chamber < 10; chamber++) {
          std::string histPath = std::string("alignment/residuals/MCH_") + topBottom + "/" + sign + "/CH" + std::to_string(chamber + 1) + "/";
          // Delta x and y at cluster
          mchResidualsHistosPerDE[i][j][chamber]["dx_vs_de"] = registryResidualsMCH.add((histPath + "dx_vs_de").c_str(), "Cluster x residual vs. DE index", {HistType::kTH2F, {deAxis, dxAxis}});
          mchResidualsHistosPerDE[i][j][chamber]["dy_vs_de"] = registryResidualsMCH.add((histPath + "dy_vs_de").c_str(), "Cluster y residual vs. DE index", {HistType::kTH2F, {deAxis, dxAxis}});

          // mixed events
          histPath = std::string("alignment/mixed-events/residuals/MCH_") + topBottom + "/" + sign + "/CH" + std::to_string(chamber + 1) + "/";
          // Delta x and y at cluster
          mchResidualsHistosPerDEMixedEvents[i][j][chamber]["dx_vs_de"] = registryResidualsMCH.add((histPath + "dx_vs_de").c_str(), "Cluster x residual vs. DE index", {HistType::kTH2F, {deAxis, dxAxis}});
          mchResidualsHistosPerDEMixedEvents[i][j][chamber]["dy_vs_de"] = registryResidualsMCH.add((histPath + "dy_vs_de").c_str(), "Cluster y residual vs. DE index", {HistType::kTH2F, {deAxis, dxAxis}});
        }
      }
    }
  }

  void CreateMuonKineHistos()
  {
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

    AxisSpec nCandidatesAxis = {static_cast<int>(fNCandidatesMax), 0.0, static_cast<double>(fNCandidatesMax), "match candidate rank"};
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
  }

  void CreateDimuonHistos()
  {
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
    // Good MFT-MCH-MID tracks with re-scaled MFT kinematics and MFT acceptance cuts
    registry.add("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMassFull_ScaledMftKine_GlobalMatchesCuts", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum", {HistType::kTH1F, {invMassAxisFull}});
    registry.add("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMassFull_ScaledMftKine_GlobalMatchesCuts", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum", {HistType::kTH1F, {invMassAxisFull}});
    // combinations of tracks from top and bottom halfs of MFT
    registry.add("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts_TT", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum, top-top", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts_TB", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum, top-bottom", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts_BT", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum, bottom-top", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts_BB", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum, bottom-bottom", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts_TT", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum, top-top", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts_TB", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum, top-bottom", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts_BT", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum, bottom-top", {HistType::kTH1F, {invMassAxis}});
    registry.add("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts_BB", "M_{#mu^{+}#mu^{-}} - rescaled MFT momentum, bottom-bottom", {HistType::kTH1F, {invMassAxis}});
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
  }

  void createMatchingHistos()
  {
    AxisSpec chi2Axis = {1000, 0, 1000, "chi^{2}"};
    AxisSpec pAxis = {1000, 0, 100, "p (GeV/c)"};
    std::string histPath = "matching/";

    std::string histName = "chi2ProdVsP";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{prod} vs. MCH momentum", {HistType::kTH2F, {pAxis, chi2Axis}});

    histName = "chi2MatchAllVsP";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchAll} vs. MCH momentum", {HistType::kTH2F, {pAxis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchAll} vs. MCH momentum (realigned)", {HistType::kTH2F, {pAxis, chi2Axis}});

    histName = "chi2MatchAllVsProd";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchAll} vs. ch2^{2}_{prod}", {HistType::kTH2F, {chi2Axis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchAll} vs. ch2^{2}_{prod} (realigned)", {HistType::kTH2F, {chi2Axis, chi2Axis}});

    histName = "chi2MatchXYPhiTanlVsP";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. MCH momentum", {HistType::kTH2F, {pAxis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. MCH momentum (realigned)", {HistType::kTH2F, {pAxis, chi2Axis}});

    histName = "chi2MatchXYPhiTanlVsProd";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. ch2^{2}_{prod}", {HistType::kTH2F, {chi2Axis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. ch2^{2}_{prod} (realigned)", {HistType::kTH2F, {chi2Axis, chi2Axis}});

    histName = "chi2MatchXYPhiTanlAltVsP";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. MCH momentum", {HistType::kTH2F, {pAxis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. MCH momentum (realigned)", {HistType::kTH2F, {pAxis, chi2Axis}});

    histName = "chi2MatchXYPhiTanlAltVsProd";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. ch2^{2}_{prod}", {HistType::kTH2F, {chi2Axis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. ch2^{2}_{prod} (realigned)", {HistType::kTH2F, {chi2Axis, chi2Axis}});

    histName = "chi2MatchXYPhiTanlAltVsStd";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchXYPhiTanl_alt} vs. ch2^{2}_{MatchXYPhiTanl_std}", {HistType::kTH2F, {chi2Axis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchXYPhiTanl_alt} vs. ch2^{2}_{MatchXYPhiTanl_std} (realigned)", {HistType::kTH2F, {chi2Axis, chi2Axis}});

    histName = "chi2MatchXYPhiTanlAltAtMCHVsP";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. MCH momentum, MCH matching plane", {HistType::kTH2F, {pAxis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchXYPhiTanl} vs. MCH momentum, MCH matching plane (realigned)", {HistType::kTH2F, {pAxis, chi2Axis}});

    histName = "chi2MatchXYPhiTanlAltAtMCHVsStd";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchXYPhiTanl_alt_at_MCH} vs. ch2^{2}_{MatchXYPhiTanl_std}", {HistType::kTH2F, {chi2Axis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchXYPhiTanl_alt_at_MCH} vs. ch2^{2}_{MatchXYPhiTanl_std} (realigned)", {HistType::kTH2F, {chi2Axis, chi2Axis}});

    histName = "chi2MatchXYPhiTanlAltAtMCHVsMFT";
    matchingHistos[histName] = registryResidualsMCH.add((histPath + histName).c_str(), "chi^{2}_{MatchXYPhiTanl_alt}, MCH vs. MFT matching planes", {HistType::kTH2F, {chi2Axis, chi2Axis}});
    matchingHistosRealigned[histName] = registryResidualsMCH.add((histPath + "realigned/" + histName).c_str(), "chi^{2}_{MatchXYPhiTanl_alt}, MCH vs. MFT matching planes (realigned)", {HistType::kTH2F, {chi2Axis, chi2Axis}});
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
    AxisSpec trackTypeAxis = {static_cast<int>(nTrackTypes), 0.0, static_cast<double>(nTrackTypes), "track type"};
    registry.add("nTracksPerType", "Number of tracks per type", {HistType::kTH1F, {trackTypeAxis}});

    CreateMuonKineHistos();

    CreateDCAHistos();

    CreateAlignementHistos();

    CreateDimuonHistos();

    createMatchingHistos();

    // Define built-in matching functions
    //________________________________________________________________________________
    mMatchingFunctionMap["matchALL"] = [](const GlobalFwdTrack& mchTrack, const TrackParCovFwd& mftTrack) -> double {
      // Match two tracks evaluating all parameters: X,Y, phi, tanl & q/pt

      SMatrix55Sym I = ROOT::Math::SMatrixIdentity(), H_k, V_k;
      SVector5 m_k(mftTrack.getX(), mftTrack.getY(), mftTrack.getPhi(),
                   mftTrack.getTanl(), mftTrack.getInvQPt()),
        r_k_kminus1;
      SVector5 GlobalMuonTrackParameters = mchTrack.getParameters();
      SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
      V_k(0, 0) = mftTrack.getCovariances()(0, 0);
      V_k(1, 1) = mftTrack.getCovariances()(1, 1);
      V_k(2, 2) = mftTrack.getCovariances()(2, 2);
      V_k(3, 3) = mftTrack.getCovariances()(3, 3);
      V_k(4, 4) = mftTrack.getCovariances()(4, 4);
      H_k(0, 0) = 1.0;
      H_k(1, 1) = 1.0;
      H_k(2, 2) = 1.0;
      H_k(3, 3) = 1.0;
      H_k(4, 4) = 1.0;

      // Covariance of residuals
      SMatrix55Std invResCov = (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
      invResCov.Invert();

      // Kalman Gain Matrix
      SMatrix55Std K_k = GlobalMuonTrackCovariances * ROOT::Math::Transpose(H_k) * invResCov;

      // Update Parameters
      r_k_kminus1 = m_k - H_k * GlobalMuonTrackParameters; // Residuals of prediction

      auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

      return matchChi2Track;
    };

    //________________________________________________________________________________
    mMatchingFunctionMap["matchXYPhiTanl"] = [](const GlobalFwdTrack& mchTrack, const TrackParCovFwd& mftTrack) -> double {

    // Match two tracks evaluating positions & angles

    SMatrix55Sym I = ROOT::Math::SMatrixIdentity();
    SMatrix45 H_k;
    SMatrix44 V_k;
    SVector4 m_k(mftTrack.getX(), mftTrack.getY(), mftTrack.getPhi(),
                 mftTrack.getTanl()),
      r_k_kminus1;
    SVector5 GlobalMuonTrackParameters = mchTrack.getParameters();
    SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
    V_k(0, 0) = mftTrack.getCovariances()(0, 0);
    V_k(1, 1) = mftTrack.getCovariances()(1, 1);
    V_k(2, 2) = mftTrack.getCovariances()(2, 2);
    V_k(3, 3) = mftTrack.getCovariances()(3, 3);
    H_k(0, 0) = 1.0;
    H_k(1, 1) = 1.0;
    H_k(2, 2) = 1.0;
    H_k(3, 3) = 1.0;

    // Covariance of residuals
    SMatrix44 invResCov = (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
    invResCov.Invert();

    // Kalman Gain Matrix
    SMatrix54 K_k = GlobalMuonTrackCovariances * ROOT::Math::Transpose(H_k) * invResCov;

    // Residuals of prediction
    r_k_kminus1 = m_k - H_k * GlobalMuonTrackParameters;

    auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

    return matchChi2Track; };

    //________________________________________________________________________________
    mMatchingFunctionMap["matchXY"] = [](const GlobalFwdTrack& mchTrack, const TrackParCovFwd& mftTrack) -> double {

    // Calculate Matching Chi2 - X and Y positions

    SMatrix55Sym I = ROOT::Math::SMatrixIdentity();
    SMatrix25 H_k;
    SMatrix22 V_k;
    SVector2 m_k(mftTrack.getX(), mftTrack.getY()), r_k_kminus1;
    SVector5 GlobalMuonTrackParameters = mchTrack.getParameters();
    SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
    V_k(0, 0) = mftTrack.getCovariances()(0, 0);
    V_k(1, 1) = mftTrack.getCovariances()(1, 1);
    H_k(0, 0) = 1.0;
    H_k(1, 1) = 1.0;

    // Covariance of residuals
    SMatrix22 invResCov = (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
    invResCov.Invert();

    // Kalman Gain Matrix
    SMatrix52 K_k = GlobalMuonTrackCovariances * ROOT::Math::Transpose(H_k) * invResCov;

    // Residuals of prediction
    r_k_kminus1 = m_k - H_k * GlobalMuonTrackParameters;
    auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

    return matchChi2Track; };

    //________________________________________________________________________________
    mMatchingFunctionMap["matchNeedsName"] = [this](const GlobalFwdTrack& mchTrack, const TrackParCovFwd& mftTrack) -> double {

    //Hiroshima's Matching function needs a physics-based name

    //Matching constants
    Double_t LAbs = 415.;    //Absorber Length[cm]
    Double_t mumass = 0.106; //mass of muon [GeV/c^2]
    Double_t l;              //the length that extrapolated MCHtrack passes through absorber

    if (mMatchingPlaneZ >= -90.0) {
      l = LAbs;
    } else {
      l = 505.0 + mMatchingPlaneZ;
    }

    //defference between MFTtrack and MCHtrack

    auto dx = mftTrack.getX() - mchTrack.getX();
    auto dy = mftTrack.getY() - mchTrack.getY();
    auto dthetax = TMath::ATan(mftTrack.getPx() / TMath::Abs(mftTrack.getPz())) - TMath::ATan(mchTrack.getPx() / TMath::Abs(mchTrack.getPz()));
    auto dthetay = TMath::ATan(mftTrack.getPy() / TMath::Abs(mftTrack.getPz())) - TMath::ATan(mchTrack.getPy() / TMath::Abs(mchTrack.getPz()));

    //Multiple Scattering(=MS)

    auto pMCH = mchTrack.getP();
    auto lorentzbeta = pMCH / TMath::Sqrt(mumass * mumass + pMCH * pMCH);
    auto zMS = copysign(1.0, mchTrack.getCharge());
    auto thetaMS = 13.6 / (1000.0 * pMCH * lorentzbeta * 1.0) * zMS * TMath::Sqrt(60.0 * l / LAbs) * (1.0 + 0.038 * TMath::Log(60.0 * l / LAbs));
    auto xMS = thetaMS * l / TMath::Sqrt(3.0);

    //normalize by theoritical Multiple Coulomb Scattering width to be momentum-independent
    //make the dx and dtheta dimensionless

    auto dxnorm = dx / xMS;
    auto dynorm = dy / xMS;
    auto dthetaxnorm = dthetax / thetaMS;
    auto dthetaynorm = dthetay / thetaMS;

    //rotate distribution

    auto dxrot = dxnorm * TMath::Cos(TMath::Pi() / 4.0) - dthetaxnorm * TMath::Sin(TMath::Pi() / 4.0);
    auto dthetaxrot = dxnorm * TMath::Sin(TMath::Pi() / 4.0) + dthetaxnorm * TMath::Cos(TMath::Pi() / 4.0);
    auto dyrot = dynorm * TMath::Cos(TMath::Pi() / 4.0) - dthetaynorm * TMath::Sin(TMath::Pi() / 4.0);
    auto dthetayrot = dynorm * TMath::Sin(TMath::Pi() / 4.0) + dthetaynorm * TMath::Cos(TMath::Pi() / 4.0);

    //convert ellipse to circle

    auto k = 0.7; //need to optimize!!
    auto dxcircle = dxrot;
    auto dycircle = dyrot;
    auto dthetaxcircle = dthetaxrot / k;
    auto dthetaycircle = dthetayrot / k;

    //score

    auto scoreX = TMath::Sqrt(dxcircle * dxcircle + dthetaxcircle * dthetaxcircle);
    auto scoreY = TMath::Sqrt(dycircle * dycircle + dthetaycircle * dthetaycircle);
    auto score = TMath::Sqrt(scoreX * scoreX + scoreY * scoreY);

    return score; };
  }

  int GetQuadrant(double phi)
  {
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

  template<class T>
  int GetQuadrant(const T& track)
  {
    double phi = track.phi() * 180 / TMath::Pi();
    return GetQuadrant(phi);
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

  void TransformMFT(o2::mch::TrackParam& track)
  {
    double zCH10 = -1437.6;
    double z = track.getZ();
    //double dZ = zMCH - z;
    double x = track.getNonBendingCoor();
    double y = track.getBendingCoor();
    double xSlope = track.getNonBendingSlope();
    double ySlope = track.getBendingSlope();

    double xSlopeCorrection = (y > 0) ?
        (-0.0006696 - 0.0005621) / 2.0 :
        (0.00105 + 0.001007) / 2.0;
    double xCorrection = xSlopeCorrection * z;
    track.setNonBendingCoor(x + xCorrection);
    track.setNonBendingSlope(xSlope + xSlopeCorrection);

    double ySlopeCorrection = (y > 0) ?
        (-0.002299 - 0.002442) / 2.0 :
        (-0.0005339 - 0.0006921) / 2.0;
    double yCorrection = ySlopeCorrection * z;
    track.setBendingCoor(y + yCorrection);
    track.setBendingSlope(ySlope + ySlopeCorrection);
    /*
    std::cout << std::format("[TOTO] MFT position:    pos={:0.3f},{:0.3f}", x, y) << std::endl;
    std::cout << std::format("[TOTO] MFT corrections: pos={:0.3f},{:0.3f}  slope={:0.12f},{:0.12f}  angle={:0.12f},{:0.12f}",
        xCorrection, yCorrection, xSlopeCorrection, ySlopeCorrection,
        std::atan2(xSlopeCorrection, 1), std::atan2(ySlopeCorrection, 1)) << std::endl;
    */
  }

  void TransformMFT(o2::dataformats::GlobalFwdTrack& track)
  {
    auto mchTrack = sExtrap.FwdtoMCH(track);

    TransformMFT(mchTrack);

    auto transformedTrack = sExtrap.MCHtoFwd(mchTrack);
    track.setParameters(transformedTrack.getParameters());
    track.setZ(transformedTrack.getZ());
    track.setCovariances(transformedTrack.getCovariances());
  }

  void TransformMFT(o2::track::TrackParCovFwd& fwdtrack)
  {
    o2::dataformats::GlobalFwdTrack track;
    track.setParameters(fwdtrack.getParameters());
    track.setZ(fwdtrack.getZ());
    track.setCovariances(fwdtrack.getCovariances());

    auto mchTrack = sExtrap.FwdtoMCH(track);

    TransformMFT(mchTrack);

    auto transformedTrack = sExtrap.MCHtoFwd(mchTrack);
    fwdtrack.setParameters(transformedTrack.getParameters());
    fwdtrack.setZ(transformedTrack.getZ());
    fwdtrack.setCovariances(transformedTrack.getCovariances());
  }

  void TransformMCH(o2::mch::TrackParam& track)
  {
    double zCH1 = firstMCHPlaneZ;

    double alpha1 = track.getNonBendingSlope();
    double alpha3 = track.getBendingSlope();
    double phi = TMath::ATan2(-alpha3, -alpha1) * 180 / TMath::Pi();
    int quadrant = GetQuadrant(phi);

    //double deltaX[4]{ 0.1003, 0.2825, 0.1807, -0.0588 };
    double deltaX[4]{ 0.1513, 0.2954, 0.1795, -0.0847 };
    //double deltaY[4]{ 0.0806, 0.1589, 0.8761, 0.7888 };
    double deltaY[4]{ 0.1328, 0.1719, 0.8538, 0.8049 };
    double deltaThetaX[4]{ -0.0030, -0.0482, -0.0499, -0.0233 };
    double deltaThetaY[4]{ 0.0018, 0.0018, 0.0731, 0.0669 };

    double deltaSlopeX[4];
    double deltaSlopeY[4];
    for (int i = 0; i < 4; i++) {
      deltaSlopeX[i] = std::tan(deltaThetaX[i] * TMath::Pi() / 180.0);
      deltaSlopeY[i] = std::tan(deltaThetaY[i] * TMath::Pi() / 180.0);
    }

    double z = track.getZ();
    double x = track.getNonBendingCoor();
    double y = track.getBendingCoor();
    double xSlope = track.getNonBendingSlope();
    double ySlope = track.getBendingSlope();

    double xSlopeCorrected = xSlope - deltaSlopeX[quadrant];
    track.setNonBendingSlope(xSlopeCorrected);
    double xShiftAtCH1 = deltaX[quadrant];
    //std::cout << std::format("[TOTO] MFT y={:0.3f}  xShift={:0.3f}",
    //    y, xShiftMCH) << std::endl;
    double xPositionCorrected = x - xShiftAtCH1 + xSlopeCorrected * (z - zCH1);
    track.setNonBendingCoor(xPositionCorrected);

    double ySlopeCorrected = ySlope - deltaSlopeY[quadrant];
    track.setBendingSlope(ySlopeCorrected);
    double yShiftAtCH1 = deltaY[quadrant];
    //std::cout << std::format("[TOTO] MFT y={:0.3f}  xShift={:0.3f}",
    //    y, xShiftMCH) << std::endl;
    double yPositionCorrected = y - yShiftAtCH1 + ySlopeCorrected * (z - zCH1);
    track.setBendingCoor(yPositionCorrected);
  }

  void TransformMCH(o2::dataformats::GlobalFwdTrack& track)
  {
    auto mchTrack = sExtrap.FwdtoMCH(track);

    TransformMCH(mchTrack);

    auto transformedTrack = sExtrap.MCHtoFwd(mchTrack);
    track.setParameters(transformedTrack.getParameters());
    track.setZ(transformedTrack.getZ());
    track.setCovariances(transformedTrack.getCovariances());
  }

  void TransformMCH(o2::track::TrackParCovFwd& fwdtrack)
  {
    o2::dataformats::GlobalFwdTrack track;
    track.setParameters(fwdtrack.getParameters());
    track.setZ(fwdtrack.getZ());
    track.setCovariances(fwdtrack.getCovariances());

    auto mchTrack = sExtrap.FwdtoMCH(track);

    TransformMCH(mchTrack);

    auto transformedTrack = sExtrap.MCHtoFwd(mchTrack);
    fwdtrack.setParameters(transformedTrack.getParameters());
    fwdtrack.setZ(transformedTrack.getZ());
    fwdtrack.setCovariances(transformedTrack.getCovariances());
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

  template<class T>
  bool IsGoodGlobalMatching(const T& muonTrack,
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

  template<class T>
  bool IsGoodGlobalMatching(const T& muonTrack)
  {
    return IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp);
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
    if (bcDiff < static_cast<uint64_t>(fEventMinDeltaBc))
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

  template <typename T>
  o2::dataformats::GlobalFwdTrack FwdToTrackPar(const T& track)
  {
    double chi2 = track.chi2();
    SMatrix5 tpars(track.x(), track.y(), track.phi(), track.tgl(), track.signed1Pt());
    std::vector<double> v1{0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0};
    SMatrix55 tcovs(v1.begin(), v1.end());
    o2::track::TrackParCovFwd trackparCov{track.z(), tpars, tcovs, chi2};
    o2::dataformats::GlobalFwdTrack fwdtrack;
    fwdtrack.setParameters(trackparCov.getParameters());
    fwdtrack.setZ(trackparCov.getZ());
    fwdtrack.setCovariances(trackparCov.getCovariances());
    return fwdtrack;
  }

  template<class TMFT, class C>
  o2::dataformats::GlobalFwdTrack PropagateMftToDCA(const TMFT& mftTrack, const C& collision, float zshift=0 )
  {
    static double Bz = -10001;
    double chi2 = mftTrack.chi2();
    SMatrix5 tpars = {mftTrack.x(), mftTrack.y(), mftTrack.phi(), mftTrack.tgl(), mftTrack.signed1Pt()};
    std::vector<double> v1{0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0};
    SMatrix55 tcovs(v1.begin(), v1.end());
    o2::track::TrackParCovFwd fwdtrack{mftTrack.z(), tpars, tcovs, chi2};
    if (fEnableMFTAlignmentCorrections) {
      TransformMFT(fwdtrack);
    }
    o2::dataformats::GlobalFwdTrack propmuon;

    double propVec[3] = {};
    propVec[0] = collision.posX() - mftTrack.x();
    propVec[1] = collision.posY() - mftTrack.y();
    propVec[2] = collision.posZ() - mftTrack.z();

    //double centerZ[3] = {mftTrack.x() + propVec[0] / 2.,
    //                     mftTrack.y() + propVec[1] / 2.,
    //                     mftTrack.z() + propVec[2] / 2.};
    if (Bz < -10000) {
      double centerZ[3] = {0, 0, -45.f / 2.f};
      o2::field::MagneticField* field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
      Bz = field->getBz(centerZ);
    }
    fwdtrack.propagateToZ(collision.posZ() - zshift, Bz);

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

    double px = mchTrackAtVertex.getP() * sin(M_PI / 2 - atan(mftTrack.tgl())) * cos(mftTrack.phi());
    double py = mchTrackAtVertex.getP() * sin(M_PI / 2 - atan(mftTrack.tgl())) * sin(mftTrack.phi());
    double pt = std::sqrt(std::pow(px, 2) + std::pow(py, 2));
    double sign = mchTrack.sign();
    double signed1Pt = sign / pt;

    //std::cout << std::format("[TOTO]   P(MCH)=({:0.2f},{:0.2f})  P(MFT)=({:0.2f},{:0.2f})  Pt(scaled)={:0.2f}",
    //    mchTrackAtVertex.getP(), mchTrackAtVertex.getPt(),
    //    mftTrack.p(), mftTrack.signed1Pt(), signed1Pt) << std::endl;

    double chi2 = mftTrack.chi2();
    SMatrix5 tpars = {mftTrack.x(), mftTrack.y(), mftTrack.phi(), mftTrack.tgl(), signed1Pt};
    std::vector<double> v1{0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0};
    SMatrix55 tcovs(v1.begin(), v1.end());
    o2::track::TrackParCovFwd fwdtrack{mftTrack.z(), tpars, tcovs, chi2};
    if (fEnableMFTAlignmentCorrections) {
      TransformMFT(fwdtrack);
    }
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
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackChi2"))->Fill(mchTrack.chi2());
          }
        }

        // track momentum distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, 0, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("global-muons/TrackP"))->Fill(mchTrack.p());
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackP"))->Fill(mchTrack.p());
          }
        }

        // track transverse momentum distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, 0, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("global-muons/TrackPt"))->Fill(mchTrack.pt());
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackPt"))->Fill(mchTrack.pt());
          }
        }

        // track eta distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {-doubleMax, doubleMax}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("global-muons/TrackEta"))->Fill(mchTrack.eta());
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackEta"))->Fill(mchTrack.eta());
          }
        }

        // track Rabs distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {0, doubleMax}, fSigmaPdcaUp)) {
          registry.get<TH1>(HIST("global-muons/TrackRabs"))->Fill(mchTrack.rAtAbsorberEnd());
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackRabs"))->Fill(mchTrack.rAtAbsorberEnd());
          }
        }

        // track pDCA distribution
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, doubleMax)) {
          registry.get<TH1>(HIST("global-muons/TrackPDCA"))->Fill(mchTrack.pDca());
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackPDCA"))->Fill(mchTrack.pDca());
          }
        }

        // track DCA and phi distributions
        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          double dca = GetDCA(mchTrack, collision);
          registry.get<TH1>(HIST("global-muons/TrackDCA"))->Fill(dca);
          registry.get<TH1>(HIST("global-muons/TrackPhi"))->Fill(mchTrack.phi() * 180.0 / TMath::Pi());
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackDCA"))->Fill(dca);
            registry.get<TH1>(HIST("global-matches/TrackPhi"))->Fill(mchTrack.phi() * 180.0 / TMath::Pi());
          }
        }

        if (IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp)) {
          if (IsGoodGlobalMatching(muonTrack, doubleMax, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackChi2_MFT"))->Fill(mftTrack.chi2());
          }
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, 0, fMatchingChi2MftMchUp)) {
            registry.get<TH1>(HIST("global-matches/TrackNclusters_MFT"))->Fill(mftTrack.nClusters());
          }
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, doubleMax)) {
            registry.get<TH1>(HIST("global-matches/MatchChi2"))->Fill(muonTrack.chi2MatchMCHMFT());
          }

          // variables not used in the quality cuts
          if (IsGoodGlobalMatching(muonTrack, fTrackChi2MftUp, fTrackNClustMftLow, fMatchingChi2MftMchUp)) {
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
      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);
      auto collisionIndex2 = muon2.first;

      auto mchIndex1 = muon1.second;
      auto mchIndex2 = muon2.second;
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

      // indexes indicating whether the positive and negative tracks come from the top or bottom halves of MFT
      int posTopBottom = (sign1 > 0) ? ((muonTrack1.y() >=0) ? 0 : 1) : ((muonTrack2.y() >=0) ? 0 : 1);
      int negTopBottom = (sign1 < 0) ? ((muonTrack1.y() >=0) ? 0 : 1) : ((muonTrack2.y() >=0) ? 0 : 1);

      bool goodGlobalMuonTracks = (IsGoodGlobalMuon(muonTrack1, collision1) && IsGoodGlobalMuon(muonTrack2, collision1));
      bool goodGlobalMuonMatches = (IsGoodGlobalMatching(muonTrack1) && IsGoodGlobalMatching(muonTrack2));

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

          if (posTopBottom == 0 && negTopBottom == 0) {
            registry.get<TH1>(HIST("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts_TT"))->Fill(massScaled);
          } else if (posTopBottom == 0 && negTopBottom == 1) {
            registry.get<TH1>(HIST("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts_TB"))->Fill(massScaled);
          } else if (posTopBottom == 1 && negTopBottom == 0) {
            registry.get<TH1>(HIST("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts_BT"))->Fill(massScaled);
          } else if (posTopBottom == 1 && negTopBottom == 1) {
            registry.get<TH1>(HIST("dimuon/invariantMass_ScaledMftKine_GlobalMatchesCuts_BB"))->Fill(massScaled);
          }

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

          if (posTopBottom == 0 && negTopBottom == 0) {
            registry.get<TH1>(HIST("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts_TT"))->Fill(massScaled);
          } else if (posTopBottom == 0 && negTopBottom == 1) {
            registry.get<TH1>(HIST("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts_TB"))->Fill(massScaled);
          } else if (posTopBottom == 1 && negTopBottom == 0) {
            registry.get<TH1>(HIST("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts_BT"))->Fill(massScaled);
          } else if (posTopBottom == 1 && negTopBottom == 1) {
            registry.get<TH1>(HIST("dimuon/mixed-events/invariantMass_ScaledMftKine_GlobalMatchesCuts_BB"))->Fill(massScaled);
          }
        }
      }

      // plots for sub-leading matches are only filled in the same-event case
      if (sameEvent) {
        if (globalTracksVector1.size() > 1) {
          auto const& muonTrack1b = muonTracks.rawIteratorAt(globalTracksVector1[1]);
          goodGlobalMuonTracks = (IsGoodGlobalMuon(muonTrack1b, collision1) && IsGoodGlobalMuon(muonTrack2, collision1));
          goodGlobalMuonMatches = (IsGoodGlobalMatching(muonTrack1b) && IsGoodGlobalMatching(muonTrack2));
          double mass = GetMuMuInvariantMass(muonTrack1b, muonTrack2, collision1);
          if (goodGlobalMuonTracks && goodGlobalMuonMatches) {
            registry.get<TH1>(HIST("dimuon/invariantMass_GlobalMuonKine_GlobalMatchesCuts_subleading_leading"))->Fill(mass);
            registry.get<TH1>(HIST("dimuon/invariantMassFull_GlobalMuonKine_GlobalMatchesCuts_subleading_leading"))->Fill(mass);
          }
        }

        if (globalTracksVector2.size() > 1) {
          auto const& muonTrack2b = muonTracks.rawIteratorAt(globalTracksVector2[1]);
          goodGlobalMuonTracks = (IsGoodGlobalMuon(muonTrack1, collision1) && IsGoodGlobalMuon(muonTrack2b, collision1));
          goodGlobalMuonMatches = (IsGoodGlobalMatching(muonTrack1) && IsGoodGlobalMatching(muonTrack2b));
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
          goodGlobalMuonMatches = (IsGoodGlobalMatching(muonTrack1b) && IsGoodGlobalMatching(muonTrack2b));
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
    if (fEnableMCHAlignmentCorrections) {
      TransformMCH(mchTrack);
    }

    o2::mch::TrackExtrap::extrapToZ(mchTrack, z);

    auto proptrack = sExtrap.MCHtoFwd(mchTrack);
    o2::dataformats::GlobalFwdTrack propmuon;
    propmuon.setParameters(proptrack.getParameters());
    propmuon.setZ(proptrack.getZ());
    propmuon.setCovariances(proptrack.getCovariances());

    return propmuon;
  }

  template <typename T>
  T UpdateTrackMomentum(const T& track, const double p, int sign)
  {
    double px = p * sin(M_PI / 2 - atan(track.tgl())) * cos(track.phi());
    double py = p * sin(M_PI / 2 - atan(track.tgl())) * sin(track.phi());
    double pt = std::sqrt(std::pow(px, 2) + std::pow(py, 2));

    SMatrix5 tpars = {track.x(), track.y(), track.phi(), track.tgl(), sign / pt};
    std::vector<double> v1{0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0};
    SMatrix55 tcovs(v1.begin(), v1.end());

    T newTrack;
    newTrack.setParameters(tpars);
    newTrack.setZ(track.z());
    newTrack.setCovariances(tcovs);

    return newTrack;
  }

  template <typename T>
  T UpdateTrackMomentum(const T& track, const o2::mch::TrackParam& track4mom)
  {
    double px = track4mom.p() * sin(M_PI / 2 - atan(track.tgl())) * cos(track.phi());
    double py = track4mom.p() * sin(M_PI / 2 - atan(track.tgl())) * sin(track.phi());
    double pt = std::sqrt(std::pow(px, 2) + std::pow(py, 2));
    double sign = track4mom.getCharge();

    SMatrix5 tpars = {track.x(), track.y(), track.phi(), track.tgl(), sign / pt};
    std::vector<double> v1{0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0};
    SMatrix55 tcovs(v1.begin(), v1.end());

    T newTrack;
    newTrack.setParameters(tpars);
    newTrack.setZ(track.z());
    newTrack.setCovariances(tcovs);

    return track;
  }

  void UpdateTrackMomentum(o2::mch::TrackParam& track, const o2::mch::TrackParam& track4mom)
  {
    double pRatio = track.p() / track4mom.p();
    double newInvBendMom = track.getInverseBendingMomentum() * pRatio;
    track.setInverseBendingMomentum(newInvBendMom);
    track.setCharge(track4mom.getCharge());
  }

  template <typename TMFT>
  o2::dataformats::GlobalFwdTrack PropagateToZMFT(const TMFT& mftTrack, const double pMCH, int signMCH, const double z)
  {
    double px = pMCH * sin(M_PI / 2 - atan(mftTrack.tgl())) * cos(mftTrack.phi());
    double py = pMCH * sin(M_PI / 2 - atan(mftTrack.tgl())) * sin(mftTrack.phi());
    double pt = std::sqrt(std::pow(px, 2) + std::pow(py, 2));
    double sign = signMCH;

    SMatrix5 tpars = {mftTrack.x(), mftTrack.y(), mftTrack.phi(), mftTrack.tgl(), sign / pt};
    std::vector<double> v1{0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0};
    SMatrix55 tcovs(v1.begin(), v1.end());

    o2::dataformats::GlobalFwdTrack track;
    track.setParameters(tpars);
    track.setZ(mftTrack.z());
    track.setCovariances(tcovs);

    auto mchTrackExt = sExtrap.FwdtoMCH(track);
    if (fEnableMFTAlignmentCorrections) {
      TransformMFT(mchTrackExt);
    }

    o2::mch::TrackExtrap::extrapToZ(mchTrackExt, z);

    o2::dataformats::GlobalFwdTrack propmuon;
    auto proptrack = sExtrap.MCHtoFwd(mchTrackExt);
    propmuon.setParameters(proptrack.getParameters());
    propmuon.setZ(proptrack.getZ());
    propmuon.setCovariances(proptrack.getCovariances());

    return propmuon;
  }

  template <typename TMFT>
  o2::dataformats::GlobalFwdTrack PropagateToZMFT(const TMFT& mftTrack, const double invQPt, const double z)
  {
    SMatrix5 tpars = {mftTrack.x(), mftTrack.y(), mftTrack.phi(), mftTrack.tgl(), invQPt};
    std::vector<double> v1{0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0};
    SMatrix55 tcovs(v1.begin(), v1.end());

    o2::dataformats::GlobalFwdTrack track;
    track.setParameters(tpars);
    track.setZ(mftTrack.z());
    track.setCovariances(tcovs);

    auto mchTrackExt = sExtrap.FwdtoMCH(track);
    if (fEnableMFTAlignmentCorrections) {
      TransformMFT(mchTrackExt);
    }

    o2::mch::TrackExtrap::extrapToZ(mchTrackExt, z);

    o2::dataformats::GlobalFwdTrack propmuon;
    auto proptrack = sExtrap.MCHtoFwd(mchTrackExt);
    propmuon.setParameters(proptrack.getParameters());
    propmuon.setZ(proptrack.getZ());
    propmuon.setCovariances(proptrack.getCovariances());

    return propmuon;
  }

  template <typename TMCH, typename TMFT>
  o2::dataformats::GlobalFwdTrack PropagateMFTtoMCH(const TMFT& mftTrack, const TMCH& mchTrack, const double z)
  {
    // extrapolation with MCH tools
    auto mchTrackAtMFT = sExtrap.FwdtoMCH(FwdToTrackPar(mchTrack));
    o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrackAtMFT, mftTrack.z());
    //std::cout << std::format("Extrapolating MFT track to z = {}:", z) << std::endl;
    //std::cout << std::format("    P_MFT    = {:0.2f}", mftTrack.p()) << std::endl;
    //std::cout << std::format("    P_MCH    = {:0.2f}", mchTrack.p()) << std::endl;
    //std::cout << std::format("    P_UPS    = {:0.2f}", mchTrackAtMFT.p()) << std::endl;

    auto mftTrackProp = sExtrap.FwdtoMCH(FwdToTrackPar(mftTrack));
    UpdateTrackMomentum(mftTrackProp, mchTrackAtMFT);
    //std::cout << std::format("    P_ATZ(1) = {:0.2f}", mftTrackProp.p()) << std::endl;
    if (z < -505.f) {
      o2::mch::TrackExtrap::extrapToZ(mftTrackProp, -466.f);
      //std::cout << std::format("    P_ATZ(2) = {:0.2f}", mftTrackProp.p()) << std::endl;
      UpdateTrackMomentum(mftTrackProp, sExtrap.FwdtoMCH(FwdToTrackPar(mchTrack)));
      //std::cout << std::format("    P_ATZ(3) = {:0.2f}", mftTrackProp.p()) << std::endl;
    }
    o2::mch::TrackExtrap::extrapToZ(mftTrackProp, z);
    //std::cout << std::format("    P_ATZ()4 = {:0.2f}", mftTrackProp.p()) << std::endl;

    return sExtrap.MCHtoFwd(mftTrackProp);
  }

  template<class TMFT, class TMCH>
  double DoMatchingStandard(const TMFT& mftTrack, MyMFTCovariance const& mftTrackCov, const TMCH& mchTrack, std::string matchingFunction, bool correctAlignment)
  {
    ///////////////////////////////////////////////////////////////////////////
    // Extrapolat MFT
    ///////////////////////////////////////////////////////////////////////////

    SMatrix5 tmftpars(mftTrack.x(),
        mftTrack.y(),
        mftTrack.phi(),
        mftTrack.tgl(),
        mftTrack.signed1Pt());

    SMatrix55Sym tmftcovs;
    tmftcovs(0, 0) = mftTrackCov.cXX();
    //std::cout << "mftTrackCov.cXX(): " << mftTrackCov.cXX() << std::endl;
    tmftcovs(0, 1) = mftTrackCov.cXY();
    tmftcovs(0, 2) = mftTrackCov.cPhiX();
    tmftcovs(0, 3) = mftTrackCov.cTglX();
    tmftcovs(0, 4) = mftTrackCov.c1PtX();

    tmftcovs(1, 1) = mftTrackCov.cYY();
    tmftcovs(1, 2) = mftTrackCov.cPhiY();
    tmftcovs(1, 3) = mftTrackCov.cTglY();
    tmftcovs(1, 4) = mftTrackCov.c1PtY();

    tmftcovs(2, 2) = mftTrackCov.cPhiPhi();
    tmftcovs(2, 3) = mftTrackCov.cTglPhi();
    tmftcovs(2, 4) = mftTrackCov.c1PtPhi();

    tmftcovs(3, 3) = mftTrackCov.cTglTgl();
    tmftcovs(3, 4) = mftTrackCov.c1PtTgl();

    tmftcovs(4, 4) = mftTrackCov.c1Pt21Pt2();

    o2::track::TrackParCovFwd extrap_mfttrack{mftTrack.z(),
      tmftpars,
      tmftcovs,
      mftTrack.chi2()};
    if (correctAlignment) {
      TransformMFT(extrap_mfttrack);
    }

    float zPlane = o2::mft::constants::mft::LayerZCoordinate()[9];

    //double centerZ[3] = {0,0,mftTrack.z() + zPlane};
    //auto Bz = fieldB->getBz(centerZ);
    extrap_mfttrack.propagateToZ(zPlane, mBzAtMftCenter); // z in cm

    o2::dataformats::GlobalFwdTrack mftTrackAtMatchingPlane;
    mftTrackAtMatchingPlane.setParameters(extrap_mfttrack.getParameters());
    mftTrackAtMatchingPlane.setZ(extrap_mfttrack.getZ());
    mftTrackAtMatchingPlane.setCovariances(extrap_mfttrack.getCovariances());

    ///////////////////////////////////////////////////////////////////////////
    // Extrapolate MCH
    ///////////////////////////////////////////////////////////////////////////

    float cov[15] = {
        mchTrack.cXX(), mchTrack.cXY(), mchTrack.cYY(),
        mchTrack.cPhiX(), mchTrack.cPhiY(), mchTrack.cPhiPhi(),
        mchTrack.cTglX(), mchTrack.cTglY(), mchTrack.cTglPhi(),
        mchTrack.cTglTgl(), mchTrack.c1PtX(), mchTrack.c1PtY(),
        mchTrack.c1PtPhi(), mchTrack.c1PtTgl(), mchTrack.c1Pt21Pt2()};

    SMatrix5 tpars(mchTrack.x(),
        mchTrack.y(),
        mchTrack.phi(),
        mchTrack.tgl(),
        mchTrack.signed1Pt());
    SMatrix55 tcovs(cov, cov + 15);
    double chi2 = mchTrack.chi2();
    //std::cout << "mchTrack.cXX(): " << mchTrack.cXX() << std::endl;

    o2::track::TrackParCovFwd parcovmuontrack{mchTrack.z(), tpars, tcovs, chi2};

    o2::dataformats::GlobalFwdTrack gtrack;
    gtrack.setParameters(tpars);
    gtrack.setZ(parcovmuontrack.getZ());
    gtrack.setCovariances(tcovs);

    auto mchtrack = sExtrap.FwdtoMCH(gtrack);
    if (correctAlignment) {
      TransformMCH(mchtrack);
    }
    o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchtrack,zPlane);

    auto fwdtrack = sExtrap.MCHtoFwd(mchtrack);

    o2::dataformats::GlobalFwdTrack muonTrackAtMatchingPlane;
    muonTrackAtMatchingPlane.setParameters(fwdtrack.getParameters());
    muonTrackAtMatchingPlane.setZ(fwdtrack.getZ());
    muonTrackAtMatchingPlane.setCovariances(fwdtrack.getCovariances());

    ///////////////////////////////////////////////////////////////////////////
    // CALCULATE CHI2
    ///////////////////////////////////////////////////////////////////////////

    //std::cout << "mchTrackAtMatchingPlane.cXX(): " << muonTrackAtMatchingPlane.getCovariances()(0, 0) << std::endl;
    //std::cout << "mftTrackAtMatchingPlane.cXX(): " << mftTrackAtMatchingPlane.getCovariances()(0, 0) << std::endl;
    double matchingChi2 = mMatchingFunctionMap[matchingFunction](muonTrackAtMatchingPlane, mftTrackAtMatchingPlane);
    //std::cout << std::format("Matching standard: z={:0.2f}  MCH=[{:0.2f} {:0.2f} {:0.2f}]  MFT=[{:0.2f} {:0.2f} {:0.2f}]  chi2={:0.3f}",
    //    zPlane,
    //    muonTrackAtMatchingPlane.getX(), muonTrackAtMatchingPlane.getY(), muonTrackAtMatchingPlane.getZ(),
    //    mftTrackAtMatchingPlane.getX(), mftTrackAtMatchingPlane.getY(), mftTrackAtMatchingPlane.getZ(),
    //    matchingChi2) << std::endl;
    //return mMatchingFunctionMap["matchALL"](muonTrackAtMatchingPlane, mftTrackAtMatchingPlane);
    return matchingChi2;
  }

  template<class TMFT, class TMCH>
  double DoMatchingAlt(const TMFT& mftTrack, MyMFTCovariance const& mftTrackCov, const TMCH& mchTrack, std::string matchingFunction, double zMatchingPlane, bool correctAlignment)
  {
    ///////////////////////////////////////////////////////////////////////////
    // Extrapolate MCH
    ///////////////////////////////////////////////////////////////////////////

    float cov[15] = {
        mchTrack.cXX(), mchTrack.cXY(), mchTrack.cYY(),
        mchTrack.cPhiX(), mchTrack.cPhiY(), mchTrack.cPhiPhi(),
        mchTrack.cTglX(), mchTrack.cTglY(), mchTrack.cTglPhi(),
        mchTrack.cTglTgl(), mchTrack.c1PtX(), mchTrack.c1PtY(),
        mchTrack.c1PtPhi(), mchTrack.c1PtTgl(), mchTrack.c1Pt21Pt2()};

    SMatrix5 tpars(mchTrack.x(),
        mchTrack.y(),
        mchTrack.phi(),
        mchTrack.tgl(),
        mchTrack.signed1Pt());
    SMatrix55 tcovs(cov, cov + 15);
    double chi2 = mchTrack.chi2();
    //std::cout << "mchTrack.cXX(): " << mchTrack.cXX() << std::endl;

    //o2::track::TrackParCovFwd parcovmuontrack{mchTrack.z(), tpars, tcovs, chi2};

    o2::dataformats::GlobalFwdTrack mchTrackPars;
    mchTrackPars.setParameters(tpars);
    mchTrackPars.setZ(mchTrack.z());
    mchTrackPars.setCovariances(tcovs);

    auto mchTrackParsAtMFT = sExtrap.FwdtoMCH(mchTrackPars);
    if (correctAlignment) {
      TransformMCH(mchTrackParsAtMFT);
    }

    // extrapolate MCH track to first MFT plane
    o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrackParsAtMFT, mftTrack.z());

    // extrapolate MCH track to matching plane
    auto mchTrackParsTemp = mchTrackParsAtMFT;
    o2::mch::TrackExtrap::extrapToZCov(mchTrackParsTemp, zMatchingPlane);
    //o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrackParsTemp,zMatchingPlane);

    auto mchTrackParsTemp2 = sExtrap.MCHtoFwd(mchTrackParsTemp);

    o2::dataformats::GlobalFwdTrack mchTrackAtMatchingPlane;
    mchTrackAtMatchingPlane.setParameters(mchTrackParsTemp2.getParameters());
    mchTrackAtMatchingPlane.setZ(mchTrackParsTemp2.getZ());
    mchTrackAtMatchingPlane.setCovariances(mchTrackParsTemp2.getCovariances());

    ///////////////////////////////////////////////////////////////////////////
    // Extrapolat MFT
    ///////////////////////////////////////////////////////////////////////////
    double pMCH = mchTrackParsAtMFT.p();
    double signMCH = mchTrack.sign();
    double px = pMCH * sin(M_PI / 2 - atan(mftTrack.tgl())) * cos(mftTrack.phi());
    double py = pMCH * sin(M_PI / 2 - atan(mftTrack.tgl())) * sin(mftTrack.phi());
    double pt = std::sqrt(std::pow(px, 2) + std::pow(py, 2));
    double sign = signMCH;

    SMatrix5 tmftpars(mftTrack.x(),
        mftTrack.y(),
        mftTrack.phi(),
        mftTrack.tgl(),
        sign / pt);

    SMatrix55Sym tmftcovs;
    tmftcovs(0, 0) = mftTrackCov.cXX();
    //std::cout << "mftTrackCov.cXX(): " << mftTrackCov.cXX() << std::endl;
    tmftcovs(0, 1) = mftTrackCov.cXY();
    tmftcovs(0, 2) = mftTrackCov.cPhiX();
    tmftcovs(0, 3) = mftTrackCov.cTglX();
    tmftcovs(0, 4) = mftTrackCov.c1PtX();

    tmftcovs(1, 1) = mftTrackCov.cYY();
    tmftcovs(1, 2) = mftTrackCov.cPhiY();
    tmftcovs(1, 3) = mftTrackCov.cTglY();
    tmftcovs(1, 4) = mftTrackCov.c1PtY();

    tmftcovs(2, 2) = mftTrackCov.cPhiPhi();
    tmftcovs(2, 3) = mftTrackCov.cTglPhi();
    tmftcovs(2, 4) = mftTrackCov.c1PtPhi();

    tmftcovs(3, 3) = mftTrackCov.cTglTgl();
    tmftcovs(3, 4) = mftTrackCov.c1PtTgl();

    tmftcovs(4, 4) = mftTrackCov.c1Pt21Pt2();

    o2::dataformats::GlobalFwdTrack mftTrackPars;
    mftTrackPars.setParameters(tmftpars);
    mftTrackPars.setZ(mftTrack.z());
    mftTrackPars.setCovariances(tmftcovs);

    auto mftTrackParsTemp = sExtrap.FwdtoMCH(mftTrackPars);
    if (correctAlignment) {
      TransformMFT(mftTrackParsTemp);
    }

    o2::mch::TrackExtrap::extrapToZCov(mftTrackParsTemp, zMatchingPlane);

    auto mftTrackParsTemp2 = sExtrap.MCHtoFwd(mftTrackParsTemp);

    o2::dataformats::GlobalFwdTrack mftTrackAtMatchingPlane;
    mftTrackAtMatchingPlane.setParameters(mftTrackParsTemp2.getParameters());
    mftTrackAtMatchingPlane.setZ(mftTrackParsTemp2.getZ());
    mftTrackAtMatchingPlane.setCovariances(mftTrackParsTemp2.getCovariances());

    ///////////////////////////////////////////////////////////////////////////
    // CALCULATE CHI2
    ///////////////////////////////////////////////////////////////////////////

    //std::cout << "mchTrackAtMatchingPlane.cXX(): " << mchTrackAtMatchingPlane.getCovariances()(0, 0) << std::endl;
    //std::cout << "mftTrackAtMatchingPlane.cXX(): " << mftTrackAtMatchingPlane.getCovariances()(0, 0) << std::endl;
    double matchingChi2 = mMatchingFunctionMap[matchingFunction](mchTrackAtMatchingPlane, mftTrackAtMatchingPlane);
    //std::cout << std::format("Matching alt:      z={:0.2f}  MCH=[{:0.2f} {:0.2f} {:0.2f}]  MFT=[{:0.2f} {:0.2f} {:0.2f}]  chi2={:0.3f}",
    //    zMatchingPlane,
    //    mchTrackAtMatchingPlane.getX(), mchTrackAtMatchingPlane.getY(), mchTrackAtMatchingPlane.getZ(),
    //    mftTrackAtMatchingPlane.getX(), mftTrackAtMatchingPlane.getY(), mftTrackAtMatchingPlane.getZ(),
    //    matchingChi2) << std::endl;
    return matchingChi2;
  }


  void FillTrackResidualsPlots(MyEvents const& collisions,
                               aod::BCsWithTimestamps const& bcs,
                               MyMuonsWithCov const& muonTracks,
                               MyMFTs const& mftTracks,
                               const std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    // outer loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {
      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);

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

        std::vector<o2::dataformats::GlobalFwdTrack> mchTrackExtrap;
        for (double z : zRefPlane) {
          mchTrackExtrap.emplace_back(PropagateToZMCH(mchTrack, z));
        }

        // inner loop over collisions
        for (auto& [collisionIndex2, collisionInfo2] : collisionInfos) {
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

            std::vector<o2::dataformats::GlobalFwdTrack> mftTrackExtrap;
            for (double z : zRefPlane) {
              mftTrackExtrap.emplace_back(PropagateToZMFT(mftTrack, mchTrackExtrap[1].getP(), mchTrack.sign(), z));
            }

            std::vector<std::array<double, 2>> xPos;
            std::vector<std::array<double, 2>> yPos;
            std::vector<std::array<double, 2>> thetax;
            std::vector<std::array<double, 2>> thetay;
            for (size_t zi = 0; zi < zRefPlane.size(); zi++) {
              xPos.emplace_back(std::array<double, 2>{mchTrackExtrap[zi].getX(), mftTrackExtrap[zi].getX()});
              yPos.emplace_back(std::array<double, 2>{mchTrackExtrap[zi].getY(), mftTrackExtrap[zi].getY()});
              thetax.emplace_back(std::array<double, 2>{
                std::atan2(mchTrackExtrap[zi].getPx(), -1.0 * mchTrackExtrap[zi].getPz()) * 180 / TMath::Pi(),
                    std::atan2(mftTrackExtrap[zi].getPx(), -1.0 * mftTrackExtrap[zi].getPz()) * 180 / TMath::Pi()
              });
              thetay.emplace_back(std::array<double, 2>{
                std::atan2(mchTrackExtrap[zi].getPy(), -1.0 * mchTrackExtrap[zi].getPz()) * 180 / TMath::Pi(),
                    std::atan2(mftTrackExtrap[zi].getPy(), -1.0 * mftTrackExtrap[zi].getPz()) * 180 / TMath::Pi()
              });
            }

            for (size_t zi = 0; zi < zRefPlane.size(); zi++) {
              if (sameEvent) {
                //std::cout << std::format("[TOTO1] X at abosrber end: {:0.2f} {:0.2f}", xPos[3][0], xPos[3][1]) << std::endl;
                //std::cout << std::format("[TOTO2] Fill({:0.2f}, {:0.2f})", std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]) << std::endl;
                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dx_vs_x"])->Fill(std::fabs(xPos[zi][1]), xPos[zi][0] - xPos[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dx_vs_y"])->Fill(std::fabs(yPos[zi][1]), xPos[zi][0] - xPos[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dy_vs_x"])->Fill(std::fabs(xPos[zi][1]), yPos[zi][0] - yPos[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dy_vs_y"])->Fill(std::fabs(yPos[zi][1]), yPos[zi][0] - yPos[zi][1]);

                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dthetax_vs_x"])->Fill(std::fabs(xPos[zi][1]), thetax[zi][0] - thetax[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dthetax_vs_y"])->Fill(std::fabs(yPos[zi][1]), thetax[zi][0] - thetax[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dthetax_vs_thetax"])->Fill(std::fabs(thetax[zi][1]), thetax[zi][0] - thetax[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dthetay_vs_x"])->Fill(std::fabs(xPos[zi][1]), thetay[zi][0] - thetay[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dthetay_vs_y"])->Fill(std::fabs(yPos[zi][1]), thetay[zi][0] - thetay[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistos[zi][quadrant]["dthetay_vs_thetay"])->Fill(std::fabs(thetay[zi][1]), thetay[zi][0] - thetay[zi][1]);
              }
              if (mixedEvent) {
                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dx_vs_x"])->Fill(std::fabs(xPos[zi][1]), xPos[zi][0] - xPos[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dx_vs_y"])->Fill(std::fabs(yPos[zi][1]), xPos[zi][0] - xPos[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dy_vs_x"])->Fill(std::fabs(xPos[zi][1]), yPos[zi][0] - yPos[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dy_vs_y"])->Fill(std::fabs(yPos[zi][1]), yPos[zi][0] - yPos[zi][1]);

                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dthetax_vs_x"])->Fill(std::fabs(xPos[zi][1]), thetax[zi][0] - thetax[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dthetax_vs_y"])->Fill(std::fabs(yPos[zi][1]), thetax[zi][0] - thetax[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dthetax_vs_thetax"])->Fill(std::fabs(thetax[zi][1]), thetax[zi][0] - thetax[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dthetay_vs_x"])->Fill(std::fabs(xPos[zi][1]), thetay[zi][0] - thetay[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dthetay_vs_y"])->Fill(std::fabs(yPos[zi][1]), thetay[zi][0] - thetay[zi][1]);
                std::get<std::shared_ptr<TH2>>(trackResidualsHistosMixedEvents[zi][quadrant]["dthetay_vs_thetay"])->Fill(std::fabs(thetay[zi][1]), thetay[zi][0] - thetay[zi][1]);
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
    // outer loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {
      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);
      int64_t bc1 = bcs.rawIteratorAt(collision1.bcId()).globalBC();

      std::get<std::shared_ptr<TH2>>(dcaHistos[0][0][0]["vertex_y_vs_x"])->Fill(collision1.posX(), collision1.posY());
      std::get<std::shared_ptr<TH1>>(dcaHistos[0][0][0]["vertex_z"])->Fill(collision1.posZ());

      // loop over muon tracks
      //std::cout << std::format("collisionInfo1.mchTracks.size(): {}", collisionInfo1.mchTracks.size()) << std::endl;
      for (auto mchIndex : collisionInfo1.mchTracks) {
        auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

        int quadrant = GetQuadrant(mchTrack);

        bool isGoodMuon = IsGoodMuon(mchTrack, collision1, fTrackChi2MchUp, 30.0, 4.0, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp);
        //std::cout << std::format("MCH track #{}: isGoodMuon={}", mchIndex, isGoodMuon) << std::endl;
        if (!isGoodMuon) continue;
        int sign = (mchTrack.sign() > 0) ? 1 : 2;

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
            std::get<std::shared_ptr<TH1>>(dcaHistos[1][quadrant][0]["DCA_x"])->Fill(dcax);
            std::get<std::shared_ptr<TH1>>(dcaHistos[1][quadrant][0]["DCA_y"])->Fill(dcay);
            std::get<std::shared_ptr<TH1>>(dcaHistos[1][quadrant][sign]["DCA_x"])->Fill(dcax);
            std::get<std::shared_ptr<TH1>>(dcaHistos[1][quadrant][sign]["DCA_y"])->Fill(dcay);
          }
          if (mixedEvent) {
            //std::cout << std::format("[TOTO1] X at abosrber end: {:0.2f} {:0.2f}", xPos[3][0], xPos[3][1]) << std::endl;
            //std::cout << std::format("[TOTO2] Fill({:0.2f}, {:0.2f})", std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]) << std::endl;
            std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[1][quadrant][0]["DCA_x"])->Fill(dcax);
            std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[1][quadrant][0]["DCA_y"])->Fill(dcay);
            std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[1][quadrant][sign]["DCA_x"])->Fill(dcax);
            std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[1][quadrant][sign]["DCA_y"])->Fill(dcay);
          }
        }
      }

      // outer loop over global muon tracks
      //for (auto& [mchIndex, globalTracksVector] : collisionInfo1.globalMuonTracks) {
        //auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
        //auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[0]);
        //const auto& mftMatchedTrack = muonTrack.template matchMFTTrack_as<MyMFTs>();
        //const auto& mftTrack = muonTrack.template matchMFTTrack_as<MyMFTs>();

        // loop over MFT tracks
      std::get<std::shared_ptr<TH1>>(dcaHistos[0][0][0]["nTracksMFT"])->Fill(collisionInfo1.mftTracks.size());
      auto mftTrackIds = collisionInfo1.mftTracks;
      auto rng = std::default_random_engine {};
      std::shuffle(std::begin(mftTrackIds), std::end(mftTrackIds), rng);
      size_t nTracksMFTmax = 10;
      if (mftTrackIds.size() > nTracksMFTmax) {
        mftTrackIds.resize(nTracksMFTmax);
      }
      //std::cout << std::format("mftTrackIds.size(): {}", mftTrackIds.size()) << std::endl;

        for (auto mftIndex : mftTrackIds) {
          auto const& mftTrack = mftTracks.rawIteratorAt(mftIndex);

          //if (mftTrack.trackTime() != mftMatchedTrack.trackTime())
          //  continue;

          int quadrant = GetQuadrant(mftTrack);
          if (quadrant < 0) continue;

          bool isGoodMFT = IsGoodMFT(mftTrack, 999.f, 5);
          if (!isGoodMFT) continue;
          int sign = (mftTrack.sign() > 0) ? 1 : 2;

          // inner loop over collisions
          for (auto& [collisionIndex2, collisionInfo2] : collisionInfos) {
            auto const& collision2 = collisions.rawIteratorAt(collisionIndex2);

            int64_t bc2 = bcs.rawIteratorAt(collision2.bcId()).globalBC();

            //bool sameEvent = (bc1 == bc2);
            bool sameEvent = (collisionIndex1 == collisionIndex2);
            bool mixedEvent = IsMixedEvent(collisionInfo1, collisionInfo2);

            //if (!sameEvent && !mixedEvent)
            if (!sameEvent)
              continue;


            auto mftTrackAtDCA = PropagateMftToDCA(mftTrack, collision2, fVertexZshift);
            double dcax = mftTrackAtDCA.getX() - collision2.posX();
            double dcay = mftTrackAtDCA.getY() - collision2.posY();
            double phi = mftTrack.phi() * 180 / TMath::Pi();
            int mftNclusters = mftTrack.nClusters();
            int mftTrackType = mftTrack.isCA() ? 1 : 0;

            const int nMftLayers = 10;
            int layerPattern = 0;
            for (int layer = 0; layer < nMftLayers; layer++) {
              if ((mftTrack.mftClusterSizesAndTrackFlags() >> (layer * 6)) & 0x3F) {
                layerPattern += (1 << layer);
              }
            }

            //std::cout << std::format("Computing MFT DCA: dx={} dy={} z={}", dcax, dcay, mftTrackAtDCA.getZ()) << std::endl;

            //auto mftTrackAtDCAalt = PropagateToZMFT(mftTrack, mchTrack, collision2.posZ());
            //double dcaxalt = 0; //mftTrackAtDCAalt.getX() - collision2.posX();
            //double dcayalt = 0; //mftTrackAtDCAalt.getY() - collision2.posY();

            if (sameEvent) {
              //std::cout << std::format("[TOTO1] X at abosrber end: {:0.2f} {:0.2f}", xPos[3][0], xPos[3][1]) << std::endl;
              //std::cout << std::format("[TOTO2] Fill({:0.2f}, {:0.2f})", std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]) << std::endl;
              //std::cout << "[TOTO] DCA_x: " << std::get<std::shared_ptr<TH1>>(dcaHistos[0][quadrant]["DCA_x"]).get() << std::endl;
              std::get<std::shared_ptr<TH1>>(dcaHistos[0][quadrant][0]["DCA_x"])->Fill(dcax);
              std::get<std::shared_ptr<TH1>>(dcaHistos[0][quadrant][0]["DCA_y"])->Fill(dcay);
              std::get<std::shared_ptr<TH1>>(dcaHistos[0][quadrant][sign]["DCA_x"])->Fill(dcax);
              std::get<std::shared_ptr<TH1>>(dcaHistos[0][quadrant][sign]["DCA_y"])->Fill(dcay);
              std::get<std::shared_ptr<TH2>>(dcaHistos[0][quadrant][0]["DCA_x_vs_z"])->Fill(collision2.posZ(), dcax);
              std::get<std::shared_ptr<TH2>>(dcaHistos[0][quadrant][0]["DCA_y_vs_z"])->Fill(collision2.posZ(), dcay);
              std::get<std::shared_ptr<TH2>>(dcaHistos[0][0][0]["DCA_y_vs_x"])->Fill(dcax, dcay);

              std::get<std::shared_ptr<THnSparse>>(dcaHistos[0][0][0]["DCA_x_full"])->Fill(dcax, collision2.posZ(), mftTrack.x(), mftTrack.y(), mftNclusters, mftTrackType);
              std::get<std::shared_ptr<THnSparse>>(dcaHistos[0][0][0]["DCA_y_full"])->Fill(dcay, collision2.posZ(), mftTrack.x(), mftTrack.y(), mftNclusters, mftTrackType);

              std::get<std::shared_ptr<THnSparse>>(dcaHistos[0][0][0]["layers"])->Fill(layerPattern, mftTrack.x(), mftTrack.y(), mftNclusters, mftTrackType);

              if (std::fabs(collision2.posZ()) < 1.f) {
                std::get<std::shared_ptr<TH2>>(dcaHistos[0][quadrant][0]["DCA_x_vs_track_x"])->Fill(mftTrack.x(), dcax);
                std::get<std::shared_ptr<TH2>>(dcaHistos[0][quadrant][0]["DCA_x_vs_track_y"])->Fill(mftTrack.y(), dcax);
                std::get<std::shared_ptr<TH2>>(dcaHistos[0][quadrant][0]["DCA_y_vs_track_x"])->Fill(mftTrack.x(), dcay);
                std::get<std::shared_ptr<TH2>>(dcaHistos[0][quadrant][0]["DCA_y_vs_track_y"])->Fill(mftTrack.y(), dcay);
                std::get<std::shared_ptr<TH2>>(dcaHistos[0][0][0]["DCA_x_vs_phi"])->Fill(phi, dcax);
                std::get<std::shared_ptr<TH2>>(dcaHistos[0][0][0]["DCA_y_vs_phi"])->Fill(phi, dcay);
                float zshift[21] = { // in millimeters
                    -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
                    0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,  4.5,  5.0
                };
                for (int zi = 0; zi < 21; zi++) {
                  auto mftTrackAtDCAshifted = PropagateMftToDCA(mftTrack, collision2, zshift[zi] / 10.f);
                  double dcaxShifted = mftTrackAtDCAshifted.getX() - collision2.posX();
                  double dcayShifted = mftTrackAtDCAshifted.getY() - collision2.posY();
                  std::get<std::shared_ptr<TH3>>(dcaHistos[0][0][0]["DCA_x_vs_phi_vs_zshift"])->Fill(zshift[zi], phi, dcaxShifted);
                  std::get<std::shared_ptr<TH3>>(dcaHistos[0][0][0]["DCA_y_vs_phi_vs_zshift"])->Fill(zshift[zi], phi, dcayShifted);

                }
              }

              //std::get<std::shared_ptr<TH2>>(dcaHistos[0][quadrant][0]["DCA_x_vs_z_alt"])->Fill(collision2.posZ(), dcaxalt);
              //std::get<std::shared_ptr<TH2>>(dcaHistos[0][quadrant][0]["DCA_y_vs_z_alt"])->Fill(collision2.posZ(), dcayalt);
              //std::get<std::shared_ptr<TH2>>(dcaHistos[0][0][0]["DCA_x_vs_phi_alt"])->Fill(phi, dcaxalt);
              //std::get<std::shared_ptr<TH2>>(dcaHistos[0][0][0]["DCA_y_vs_phi_alt"])->Fill(phi, dcayalt);
            }

            if (mixedEvent) {
              //std::cout << std::format("[TOTO1] X at abosrber end: {:0.2f} {:0.2f}", xPos[3][0], xPos[3][1]) << std::endl;
              //std::cout << std::format("[TOTO2] Fill({:0.2f}, {:0.2f})", std::fabs(xPos[i][1]), xPos[i][0] - xPos[i][1]) << std::endl;
              std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[0][quadrant][0]["DCA_x"])->Fill(dcax);
              std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[0][quadrant][0]["DCA_y"])->Fill(dcay);
              std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[0][quadrant][sign]["DCA_x"])->Fill(dcax);
              std::get<std::shared_ptr<TH1>>(dcaHistosMixedEvents[0][quadrant][sign]["DCA_y"])->Fill(dcay);
              std::get<std::shared_ptr<TH2>>(dcaHistosMixedEvents[0][quadrant][0]["DCA_x_vs_z"])->Fill(collision2.posZ(), dcax);
              std::get<std::shared_ptr<TH2>>(dcaHistosMixedEvents[0][quadrant][0]["DCA_y_vs_z"])->Fill(collision2.posZ(), dcay);
            }
          }
        //}
      }
    }
  }

  void FillResidualsPlotsMFT(MyEvents const& collisions,
      aod::BCsWithTimestamps const& bcs,
      MyMuonsWithCov const& muonTracks,
      aod::FwdTrkCls const& clusters,
      const std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    // outer loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {
      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);
      int64_t bc1 = bcs.rawIteratorAt(collision1.bcId()).globalBC();

      // outer loop over global muon tracks
      for (auto& [muonIndex, globalTracksVector] : collisionInfo1.globalMuonTracks) {
        auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[0]);
        const auto& mchTrack = muonTrack.template matchMCHTrack_as<MyMuonsWithCov>();
        const auto& mftTrack = muonTrack.template matchMFTTrack_as<MyMFTs>();
        int quadrant = GetQuadrant(mftTrack);
        int posNeg = (mchTrack.sign() >=0) ? 0 : 1;
        int topBottom = (mftTrack.y() >=0) ? 0 : 1;

        bool isGoodMuon = IsGoodMuon(mchTrack, collision1, fTrackChi2MchUp, 20.0, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp);
        if (!isGoodMuon) continue;

        bool isGoodMFT = IsGoodMFT(mftTrack, fTrackChi2MftUp, fTrackNClustMftLow);
        if (!isGoodMFT) continue;

        double mchMomentum = mchTrack.p();

        // inner loop over collisions
        for (auto& [collisionIndex2, collisionInfo2] : collisionInfos) {
          auto const& collision2 = collisions.rawIteratorAt(collisionIndex2);
          int64_t bc2 = bcs.rawIteratorAt(collision2.bcId()).globalBC();

          bool sameEvent = (bc1 == bc2);
          bool mixedEvent = IsMixedEvent(collisionInfo1, collisionInfo2);

          if (!sameEvent && !mixedEvent)
            continue;

          // inner loop over MCH tracks
          for (auto mchIndex : collisionInfo2.mchTracks) {
            auto const& mchTrack2 = muonTracks.rawIteratorAt(mchIndex);

            // Loop over attached clusters
            for (auto const& cluster : clusters) {

              //std::cout << "Checking cluster" << std::endl;
              if (cluster.template fwdtrack_as<MyMuonsWithCov>() != mchTrack2) {
                continue;
              }

              int deId = cluster.deId();
              int chamber = deId / 100 - 1;
              if (chamber < 0 || chamber > 9)
                continue;
              int deIndex = getDEindex(deId);

              double xCluster = cluster.x();
              double yCluster = cluster.y();
              double zCluster = cluster.z();
              double phiClus = std::atan2(yCluster, xCluster) * 180 / TMath::Pi();

              //auto mftTrackAtCluster = PropagateToZMFT(mftTrack, mchMomentum, mchTrack.sign(), zCluster);
              auto mftTrackAtCluster = PropagateMFTtoMCH(mftTrack, mchTrack, zCluster);

              std::array<double, 2> xPos{xCluster, mftTrackAtCluster.getX()};
              std::array<double, 2> yPos{yCluster, mftTrackAtCluster.getY()};

              std::get<std::shared_ptr<THnSparse>>(residualsHistos[0][0]["dx_vs_chamber"])->Fill(xPos[0] - xPos[1], chamber + 1, quadrant, posNeg);
              std::get<std::shared_ptr<THnSparse>>(residualsHistos[0][0]["dy_vs_chamber"])->Fill(yPos[0] - yPos[1], chamber + 1, quadrant, posNeg);

              std::get<std::shared_ptr<THnSparse>>(residualsHistos[0][0]["dx_vs_de"])->Fill(xPos[0] - xPos[1], deIndex, quadrant, posNeg);
              std::get<std::shared_ptr<THnSparse>>(residualsHistos[0][0]["dy_vs_de"])->Fill(yPos[0] - yPos[1], deIndex, quadrant, posNeg);

              //std::cout << std::format("quadrant={}  chamber={}", quadrant, chamber) << std::endl;
              if (sameEvent) {
                std::get<std::shared_ptr<TH2>>(residualsHistos[quadrant][chamber]["dx_vs_x"])->Fill(std::fabs(xPos[1]), xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistos[quadrant][chamber]["dx_vs_y"])->Fill(std::fabs(yPos[1]), xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistos[quadrant][chamber]["dy_vs_x"])->Fill(std::fabs(xPos[1]), yPos[0] - yPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistos[quadrant][chamber]["dy_vs_y"])->Fill(std::fabs(yPos[1]), yPos[0] - yPos[1]);

                // residuals vs. DE index
                std::get<std::shared_ptr<TH2>>(residualsHistosPerDE[topBottom][posNeg][chamber]["dx_vs_de"])->Fill(deId % 100, xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistosPerDE[topBottom][posNeg][chamber]["dy_vs_de"])->Fill(deId % 100, yPos[0] - yPos[1]);

                // residuals vs. cluster phi
                std::get<std::shared_ptr<TH2>>(residualsHistosPerDE[topBottom][posNeg][chamber]["dx_vs_phi"])->Fill(phiClus, xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistosPerDE[topBottom][posNeg][chamber]["dy_vs_phi"])->Fill(phiClus, yPos[0] - yPos[1]);
              }
              if (mixedEvent) {
                std::get<std::shared_ptr<TH2>>(residualsHistosMixedEvents[quadrant][chamber]["dx_vs_x"])->Fill(std::fabs(xPos[1]), xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistosMixedEvents[quadrant][chamber]["dx_vs_y"])->Fill(std::fabs(yPos[1]), xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistosMixedEvents[quadrant][chamber]["dy_vs_x"])->Fill(std::fabs(xPos[1]), yPos[0] - yPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistosMixedEvents[quadrant][chamber]["dy_vs_y"])->Fill(std::fabs(yPos[1]), yPos[0] - yPos[1]);

                // residuals vs. DE index
                std::get<std::shared_ptr<TH2>>(residualsHistosPerDEMixedEvents[topBottom][posNeg][chamber]["dx_vs_de"])->Fill(deId % 100, xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistosPerDEMixedEvents[topBottom][posNeg][chamber]["dy_vs_de"])->Fill(deId % 100, yPos[0] - yPos[1]);

                // residuals vs. cluster phi
                std::get<std::shared_ptr<TH2>>(residualsHistosPerDEMixedEvents[topBottom][posNeg][chamber]["dx_vs_phi"])->Fill(phiClus, xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(residualsHistosPerDEMixedEvents[topBottom][posNeg][chamber]["dy_vs_phi"])->Fill(phiClus, yPos[0] - yPos[1]);
              }
            }
          }
        }
      }
    }
  }

  void FillResidualsPlotsMCH(MyEvents const& collisions,
      aod::BCsWithTimestamps const& bcs,
      MyMuonsWithCov const& muonTracks,
      aod::FwdTrkCls const& clusters,
      const std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    // outer loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {
      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);
      int64_t bc1 = bcs.rawIteratorAt(collision1.bcId()).globalBC();

      // outer loop over global muon tracks
      for (auto& [muonIndex, globalTracksVector] : collisionInfo1.globalMuonTracks) {
        auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[0]);
        const auto& mchTrack = muonTrack.template matchMCHTrack_as<MyMuonsWithCov>();

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

          // inner loop over MCH tracks
          for (auto mchIndex : collisionInfo2.mchTracks) {
            auto const& mchTrack2 = muonTracks.rawIteratorAt(mchIndex);

            // Loop over attached clusters
            for (auto const& cluster : clusters) {

              //std::cout << "Checking cluster" << std::endl;
              if (cluster.template fwdtrack_as<MyMuonsWithCov>() != mchTrack2) {
                continue;
              }

              int deId = cluster.deId();
              int chamber = deId / 100 - 1;
              if (chamber < 0 || chamber > 9)
                continue;
              int deIndex = deId % 100;
              if (deIndex > 25)
                continue;

              double xCluster = cluster.x();
              double yCluster = cluster.y();
              double zCluster = cluster.z();

              int topBottom = (mchTrack.y() >=0) ? 0 : 1;
              int posNeg = (muonTrack.sign() >=0) ? 0 : 1;

              auto mchTrackAtCluster = PropagateToZMCH(mchTrack, zCluster);

              std::array<double, 2> xPos{xCluster, mchTrackAtCluster.getX()};
              std::array<double, 2> yPos{yCluster, mchTrackAtCluster.getY()};

              if (sameEvent) {
                std::get<std::shared_ptr<TH2>>(mchResidualsHistosPerDE[topBottom][posNeg][chamber]["dx_vs_de"])->Fill(deIndex, xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(mchResidualsHistosPerDE[topBottom][posNeg][chamber]["dy_vs_de"])->Fill(deIndex, yPos[0] - yPos[1]);
              }
              if (mixedEvent) {
                std::get<std::shared_ptr<TH2>>(mchResidualsHistosPerDEMixedEvents[topBottom][posNeg][chamber]["dx_vs_de"])->Fill(deIndex, xPos[0] - xPos[1]);
                std::get<std::shared_ptr<TH2>>(mchResidualsHistosPerDEMixedEvents[topBottom][posNeg][chamber]["dy_vs_de"])->Fill(deIndex, yPos[0] - yPos[1]);
              }
            }
          }
        }
      }
    }
  }

  void FillMatchingPlots(MyEvents const& collisions,
      aod::BCsWithTimestamps const& bcs,
      MyMuonsWithCov const& muonTracks,
      MyMFTCovariances const& mftTrackCovs,
      const std::map<uint64_t, CollisionInfo>& collisionInfos)
  {
    std::unordered_map<int64_t,int32_t> map_mfttrackcovs;
    for (auto &mftTrackCov : mftTrackCovs) {
      map_mfttrackcovs[mftTrackCov.matchMFTTrackId()] = mftTrackCov.globalIndex();
    }

   // outer loop over collisions
    for (auto& [collisionIndex1, collisionInfo1] : collisionInfos) {
      auto const& collision1 = collisions.rawIteratorAt(collisionIndex1);
      int64_t bc1 = bcs.rawIteratorAt(collision1.bcId()).globalBC();

      // outer loop over global muon tracks
      for (auto& [muonIndex, globalTracksVector] : collisionInfo1.globalMuonTracks) {
        auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[0]);
        if (static_cast<int>(muonTrack.trackType()) >= 2) continue;

        auto const& mftTrack = muonTrack.template matchMFTTrack_as<MyMFTs>();
        // Get MFT track ID in a global muon track
        int mftTrackId = muonTrack.matchMFTTrackId();
        // Get the covariances of the MFT track with covariance global ID converted from MFT ID
        auto const& mftTrackCov = mftTrackCovs.rawIteratorAt(map_mfttrackcovs[mftTrackId]);

        auto const& mchTrack = muonTrack.template matchMCHTrack_as<MyMuonsWithCov>();

        double chi2Prod = muonTrack.chi2MatchMCHMFT();
        double mchMom = mchTrack.p();
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2ProdVsP"])->Fill(mchMom, chi2Prod);

        //std::cout << "\n\n========\n" << std::endl;

        // matchAll function, standard extrapolation
        double chi2Std = DoMatchingStandard(mftTrack, mftTrackCov, mchTrack, "matchALL", false);

        //continue;
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchAllVsP"])->Fill(mchMom, chi2Std);
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchAllVsProd"])->Fill(chi2Prod, chi2Std);

        double chi2StdRealign = DoMatchingStandard(mftTrack, mftTrackCov, mchTrack, "matchALL", true);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchAllVsP"])->Fill(mchMom, chi2StdRealign);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchAllVsProd"])->Fill(chi2Prod, chi2StdRealign);

        // matchXYPhiTanl function, standard extrapolation
        double chi2MatchXYPhiTanl = DoMatchingStandard(mftTrack, mftTrackCov, mchTrack, "matchXYPhiTanl", false);
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchXYPhiTanlVsP"])->Fill(mchMom, chi2MatchXYPhiTanl);
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchXYPhiTanlVsProd"])->Fill(chi2Prod, chi2MatchXYPhiTanl);

        double chi2MatchXYPhiTanlRealign = DoMatchingStandard(mftTrack, mftTrackCov, mchTrack, "matchXYPhiTanl", true);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchXYPhiTanlVsP"])->Fill(mchMom, chi2MatchXYPhiTanlRealign);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchXYPhiTanlVsProd"])->Fill(chi2Prod, chi2MatchXYPhiTanlRealign);

        // matchXYPhiTanl function, alternative extrapolation, matching plane at MFT
        double chi2MatchXYPhiTanlAlt = DoMatchingAlt(mftTrack, mftTrackCov, mchTrack, "matchXYPhiTanl", lastMFTPlaneZ, false);
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchXYPhiTanlAltVsP"])->Fill(mchMom, chi2MatchXYPhiTanlAlt);
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchXYPhiTanlAltVsProd"])->Fill(chi2Prod, chi2MatchXYPhiTanlAlt);
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchXYPhiTanlAltVsStd"])->Fill(chi2Std, chi2MatchXYPhiTanlAlt);

        double chi2MatchXYPhiTanlAltRealign = DoMatchingAlt(mftTrack, mftTrackCov, mchTrack, "matchXYPhiTanl", lastMFTPlaneZ, true);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchXYPhiTanlAltVsP"])->Fill(mchMom, chi2MatchXYPhiTanlAltRealign);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchXYPhiTanlAltVsProd"])->Fill(chi2Prod, chi2MatchXYPhiTanlAltRealign);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchXYPhiTanlAltVsStd"])->Fill(chi2StdRealign, chi2MatchXYPhiTanlAltRealign);

        // matchXYPhiTanl function, alternative extrapolation, matching plane at MCH
        double chi2MatchXYPhiTanlAltAtMCH = DoMatchingAlt(mftTrack, mftTrackCov, mchTrack, "matchXYPhiTanl", firstMCHPlaneZ, false);
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchXYPhiTanlAltAtMCHVsP"])->Fill(mchMom, chi2MatchXYPhiTanlAltAtMCH);
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchXYPhiTanlAltAtMCHVsStd"])->Fill(chi2Std, chi2MatchXYPhiTanlAltAtMCH);
        std::get<std::shared_ptr<TH2>>(matchingHistos["chi2MatchXYPhiTanlAltAtMCHVsMFT"])->Fill(chi2MatchXYPhiTanlAlt, chi2MatchXYPhiTanlAltAtMCH);

        double chi2MatchXYPhiTanlAltAtMCHRealign = DoMatchingAlt(mftTrack, mftTrackCov, mchTrack, "matchXYPhiTanl", firstMCHPlaneZ, true);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchXYPhiTanlAltAtMCHVsP"])->Fill(mchMom, chi2MatchXYPhiTanlAltAtMCHRealign);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchXYPhiTanlAltAtMCHVsStd"])->Fill(chi2StdRealign, chi2MatchXYPhiTanlAltAtMCHRealign);
        std::get<std::shared_ptr<TH2>>(matchingHistosRealigned["chi2MatchXYPhiTanlAltAtMCHVsMFT"])->Fill(chi2MatchXYPhiTanlAltRealign, chi2MatchXYPhiTanlAltAtMCHRealign);

        //std::cout << std::format("[TOTO] matching chi2: {} / {}", muonTrack.chi2MatchMCHMFT(), chi2) << std::endl;
     }
    }
 }

  void checkCollisions(MyEvents const& collisions,
                      aod::BCsWithTimestamps const& bcs)
  {
    for (auto& coll : collisions) {
      rctChecker(coll);
    }
  }

  PROCESS_SWITCH(qaMuon, checkCollisions, "check collisions", true);

  void processQA(MyEvents const& collisions,
      aod::BCsWithTimestamps const& bcs,
      MyMuonsWithCov const& muonTracks,
      MyMFTs const& mftTracks,
      //MyMFTCovariances const& mftCovariances,
      aod::FwdTrkCls const& clusters)
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

    //FillMuonPlots(collisions, bcs, muonTracks, collisionInfos);

    //FillDimuonPlots(collisions, muonTracks, collisionInfos);

    FillDCAPlots(collisions, bcs, muonTracks, mftTracks, collisionInfos);
    //FillTrackResidualsPlots(collisions, bcs, muonTracks, mftTracks, collisionInfos);
    FillResidualsPlotsMFT(collisions, bcs, muonTracks, clusters, collisionInfos);
    //FillResidualsPlotsMCH(collisions, bcs, muonTracks, clusters, collisionInfos);

    //FillMatchingPlots(collisions, bcs, muonTracks, mftCovariances, collisionInfos);
  }

  PROCESS_SWITCH(qaMuon, processQA, "process qa", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<qaMuon>(cfgc)};
};
