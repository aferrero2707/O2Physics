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
/// \file global-muon-matcher.cxx
/// \brief Global muon matching
//
#include "PWGDQ/Core/MuonMatchingMlResponse.h"
#include "PWGDQ/Core/VarManager.h"

#include "Common/Core/fwdtrackUtilities.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/CollisionAssociationTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Multiplicity.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/FwdTrackReAlignTables.h"
#include "Tools/ML/MlResponse.h"

#include <CCDB/BasicCCDBManager.h>
#include <CCDB/CcdbApi.h>
#include <CommonConstants/LHCConstants.h>
#include <CommonConstants/MathConstants.h>
#include <CommonConstants/PhysicsConstants.h>
#include <DataFormatsParameters/GRPMagField.h>
#include <DetectorsBase/GeometryManager.h>
#include <DetectorsBase/Propagator.h>
#include <Field/MagneticField.h>
#include <Framework/ASoA.h>
#include <Framework/AnalysisDataModel.h>
#include <Framework/AnalysisTask.h>
#include <Framework/Configurable.h>
#include <Framework/DataTypes.h>
#include <Framework/InitContext.h>
#include <Framework/runDataProcessing.h>
#include <GlobalTracking/MatchGlobalFwd.h>
#include "MCHGeometryTransformer/Transformations.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackParam.h"
#include "MCHTracking/TrackFitter.h"
#include "MCHBase/TrackerParam.h"
#include <MCHTracking/TrackExtrap.h>
#include <MFTTracking/Constants.h>
#include <ReconstructionDataFormats/GlobalFwdTrack.h>
#include <ReconstructionDataFormats/TrackFwd.h>

#include <Math/MatrixFunctions.h>
#include <Math/MatrixRepresentationsStatic.h>
#include <Math/ProbFuncMathCore.h>
#include <Math/SMatrix.h>
#include <Math/SVector.h>
#include <TGeoGlobalMagField.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <math.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::aod;

namespace o2::aod::globalmuonmatching
{
DECLARE_SOA_COLUMN(MchTrackId, mchTrackId, int64_t);
DECLARE_SOA_COLUMN(MftTrackId, mftTrackId, int64_t);
DECLARE_SOA_COLUMN(MatchChi2, matchChi2, float);
DECLARE_SOA_COLUMN(MatchScore, matchScore, float);
DECLARE_SOA_COLUMN(MatchRanking, matchRanking, int32_t);
DECLARE_SOA_COLUMN(IsTagged, isTagged, bool);
} // namespace o2::aod::globalmuonmatching

namespace o2::aod
{
DECLARE_SOA_TABLE(GlobalMuonMatchCandidates, "AOD", "GMCAND",
                  o2::soa::Index<>,
                  globalmuonmatching::MchTrackId,
                  globalmuonmatching::MftTrackId,
                  globalmuonmatching::MatchChi2, globalmuonmatching::MatchScore, globalmuonmatching::MatchRanking,
                  globalmuonmatching::IsTagged);

namespace globalmuonmatching
{
DECLARE_SOA_ARRAY_INDEX_COLUMN(GlobalMuonMatchCandidate, matchCandidate); //! Array of GlobalMuonMatchCandidates indices
} // namespace globalmuonmatching

DECLARE_SOA_TABLE(FwdTrkMatchCands, "AOD", "FWDTRKMATCHCAND", //! Vectors of match-candidate indices stored per fwdtrack
                  globalmuonmatching::GlobalMuonMatchCandidateIds, o2::soa::Marker<3>);
} // namespace o2::aod

using MyEvents = soa::Join<aod::Collisions, aod::EvSels, aod::FT0Mults, aod::MFTMults, aod::PVMults, aod::CentFT0Ms, aod::CentFT0As, aod::CentFT0Cs>;
using MyMuons = soa::Join<aod::FwdTracks, aod::FwdTracksCov>;
using MyMFTs = aod::MFTTracks;
using MyMFTCovariances = aod::MFTTracksCov;

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix5 = ROOT::Math::SVector<double, 5>;

const int fgNDetElemCh[10] = {4, 4, 4, 4, 18, 18, 26, 26, 26, 26};
const int fgSNDetElemCh[11] = {0, 4, 8, 12, 16, 34, 52, 78, 104, 130, 156};

static float chi2ToScore(float chi2, int ndf, float chi2max)
{
  double p = -std::log10(ROOT::Math::chisquared_cdf_c(chi2, ndf));
  double pnorm = -std::log10(ROOT::Math::chisquared_cdf_c(chi2max, ndf));
  double result = (1.f / (p / pnorm + 1.f));
  return static_cast<float>(result);
}

struct GlobalMuonMatching {

  static constexpr int GlobalTrackTypeMax = 2;
  static constexpr int MchMidTrackType = 3;
  static constexpr int ThetaAbsBoundaryDeg = 3;
  static constexpr double SlopeResolutionZ = 535.;
  static constexpr int MatchingDegreesOfFreedom = 5;
  static constexpr float MatchingScoreChi2Max = 50.f;
  static constexpr int ExtrapolationMethodStandard = 0;
  static constexpr int ExtrapolationMethodMftFirstPoint = 2;
  static constexpr int ExtrapolationMethodVertex = 3;
  static constexpr int ExtrapolationMethodMftDca = 4;
  static constexpr float MatchingPlaneDefaultZ = -77.5;

  struct MatchingCandidate {
    int64_t mftTrackId{-1};
    double matchScore{-1};
    double matchChi2{-1};
    int matchRanking{-1};
  };


  ////   Variables for selecting muon tracks
  Configurable<float> cfgPMchLow{"cfgPMchLow", 0.0f, ""};
  Configurable<float> cfgPtMchLow{"cfgPtMchLow", 0.7f, ""};
  Configurable<float> cfgEtaMchLow{"cfgEtaMchLow", -4.0f, ""};
  Configurable<float> cfgEtaMchUp{"cfgEtaMchUp", -2.5f, ""};
  Configurable<float> cfgRabsLow{"cfgRabsLow", 17.6f, ""};
  Configurable<float> cfgRabsUp{"cfgRabsUp", 89.5f, ""};
  Configurable<float> cfgPdcaUp{"cfgPdcaUp", 6.f, ""};
  Configurable<float> cfgTrackChi2MchUp{"cfgTrackChi2MchUp", 5.f, ""};

  ////   Variables for selecting mft tracks
  Configurable<float> cfgEtaMftLow{"cfgEtaMftLow", -3.6f, ""};
  Configurable<float> cfgEtaMftUp{"cfgEtaMftUp", -2.5f, ""};

  ////   Variables for selecting tagged muons
  Configurable<int> cfgMuonTaggingNCrossedMftPlanesLow{"cfgMuonTaggingNCrossedMftPlanesLow", 5, ""};
  Configurable<float> cfgMuonTaggingTrackChi2MchUp{"cfgMuonTaggingTrackChi2MchUp", 5.f, ""};
  Configurable<float> cfgMuonTaggingPMchLow{"cfgMuonTaggingPMchLow", 0.0f, ""};
  Configurable<float> cfgMuonTaggingPtMchLow{"cfgMuonTaggingPtMchLow", 0.7f, ""};
  Configurable<float> cfgMuonTaggingEtaMchLow{"cfgMuonTaggingEtaMchLow", -3.6f, ""};
  Configurable<float> cfgMuonTaggingEtaMchUp{"cfgMuonTaggingEtaMchUp", -2.5f, ""};
  Configurable<float> cfgMuonTaggingRabsLow{"cfgMuonTaggingRabsLow", 17.6f, ""};
  Configurable<float> cfgMuonTaggingRabsUp{"cfgMuonTaggingRabsUp", 89.5f, ""};
  Configurable<float> cfgMuonTaggingPdcaUp{"cfgMuonTaggingPdcaUp", 4.f, ""};
  Configurable<float> cfgMuonTaggingRadiusAtMftFrontLow{"cfgMuonTaggingRadiusAtMftFrontLow", 3.f, ""};
  Configurable<float> cfgMuonTaggingRadiusAtMftFrontUp{"cfgMuonTaggingRadiusAtMftFrontUp", 9.f, ""};
  Configurable<float> cfgMuonTaggingRadiusAtMftBackLow{"cfgMuonTaggingRadiusAtMftBackLow", 5.f, ""};
  Configurable<float> cfgMuonTaggingRadiusAtMftBackUp{"cfgMuonTaggingRadiusAtMftBackUp", 12.f, ""};

  ////   Variables for ccdb
  Configurable<bool> cfgEnableMCHRealign{"cfgEnableMCHRealign", true, "Enable re-alignment of MCH clusters and tracks"};
  Configurable<std::string> ccdbUrl{"ccdbUrl", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
  Configurable<std::string> grpPath{"grpPath", "GLO/GRP/GRP", "Path of the grp file"};
  Configurable<std::string> grpMagPath{"grpMagPath", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
  Configurable<std::string> geoPath{"geoPath", "GLO/Config/GeometryAligned", "Path of the geometry file"};
  Configurable<std::string> geoRefPath{"geoRefPath", "GLO/Config/GeometryAligned", "Path of the reference geometry file"};
  Configurable<std::string> geoNewPath{"geoNewPath", "GLO/Config/GeometryAligned", "Path of the new geometry file"};
  Configurable<int64_t> nolaterthanRef{"ccdb-no-later-than-ref", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), "latest acceptable timestamp of creation for the object of reference basis"};
  Configurable<int64_t> nolaterthanNew{"ccdb-no-later-than-new", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), "latest acceptable timestamp of creation for the object of new basis"};
  Configurable<double> cfgChamberResolutionX{"cfgChamberResolutionX", 0.04, "Chamber resolution along X configuration for refit"}; // 0.4cm pp, 0.2cm PbPb
  Configurable<double> cfgChamberResolutionY{"cfgChamberResolutionY", 0.04, "Chamber resolution along Y configuration for refit"}; // 0.4cm pp, 0.2cm PbPb
  Configurable<double> cfgSigmaCutImprove{"cfgSigmaCutImprove", 6., "Sigma cut for track improvement"};                            // 6 for pp, 4 for PbPb

  // CCDB connection configurables
  struct : ConfigurableGroup {
    Configurable<std::string> cfgCcdbUrl{"cfgCcdbUrl", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
    Configurable<int64_t> cfgCcdbNoLaterThan{"cfgCcdbNoLaterThan", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), "latest acceptable timestamp of creation for the object"};
    Configurable<std::string> cfgGrpPath{"cfgGrpPath", "GLO/GRP/GRP", "Path of the grp file"};
    Configurable<std::string> cfgGeoPath{"cfgGeoPath", "GLO/Config/GeometryAligned", "Path of the geometry file"};
    Configurable<std::string> cfgGrpmagPath{"cfgGrpmagPath", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
  } configCcdb;

  // Matching strategy for the *custom* matches (production baseline is always computed).
  // 0 = chi2 (runChi2Matching), 1 = ML (runMlMatching)
  Configurable<int> cfgCustomMatchingStrategy{"cfgCustomMatchingStrategy", 0, "0=chi2, 1=ML for custom matches"};
  Configurable<bool> cfgProduceCandidateFwdTracks{"cfgProduceCandidateFwdTracks", false, "Produce GMMCANDTRK/GMMCANDTRKCOV tables (all FwdTracks + match candidates)"};
  Configurable<bool> cfgIncludeGlobalMuonsInFwdTracks{"cfgIncludeGlobalMuonsInFwdTracks", false, "Include MFT-MCH-MID global muons in GMMCANDTRK table"};
  Configurable<int> cfgMaxCandidatesPerMchTrack{"cfgMaxCandidatesPerMchTrack", -1, "Maximum number of match candidates stored per MCH track (-1: no limit)"};

  double mBzAtMftCenter{0};

  o2::globaltracking::MatchGlobalFwd mExtrap;

  using MatchingFunc = std::function<std::tuple<double, int>(const o2::dataformats::GlobalFwdTrack& mchtrack, const o2::track::TrackParCovFwd& mfttrack)>;
  std::map<std::string, MatchingFunc> mMatchingFunctionMap; ///< MFT-MCH Matching function

  // Chi2 matching interface (single configurable method)
  struct : ConfigurableGroup {
    Configurable<std::string> cfgChi2FunctionLabel{"cfgChi2FunctionLabel", std::string{"ProdAll"}, "Text label identifying the chi2 matching method"};
    Configurable<std::string> cfgChi2FunctionName{"cfgChi2FunctionName", std::string{"prod"}, "Name of the chi2 matching function"};
    Configurable<float> cfgChi2FunctionMatchingPlaneZ{"cfgChi2FunctionMatchingPlaneZ", static_cast<float>(o2::mft::constants::mft::LayerZCoordinate()[9]), "Z position of the matching plane"};
    Configurable<int> cfgChi2MatchingExtrapMethod{"cfgChi2MatchingExtrapMethod", 0, "Method for MCH track extrapolation to matching plane"};
  } configChi2MatchingOptions;

  // ML interface (single configurable model)
  struct : ConfigurableGroup {
    Configurable<std::string> cfgMlModelLabel{"cfgMlModelLabel", std::string{""}, "Text label identifying this ML model"};
    Configurable<std::string> cfgMlModelPathCcdb{"cfgMlModelPathCcdb", "Users/m/mcoquet/MLTest", "Path of model on CCDB"};
    Configurable<std::string> cfgMlModelName{"cfgMlModelName", "model.onnx", "ONNX file name (if not from CCDB full path)"};
    Configurable<std::vector<std::string>> cfgMlInputFeatures{"cfgMlInputFeatures", std::vector<std::string>{"chi2MCHMFT"}, "Names of ML model input features"};
    Configurable<float> cfgMlModelMatchingPlaneZ{"cfgMlModelMatchingPlaneZ", static_cast<float>(o2::mft::constants::mft::LayerZCoordinate()[9]), "Z position of the matching plane"};
    Configurable<int> cfgMlMatchingExtrapMethod{"cfgMlMatchingExtrapMethod", 0, "Method for MCH track extrapolation to matching plane"};
  } configMlOptions;

  std::vector<double> binsPtMl;
  std::array<double, 1> cutValues;
  std::vector<int> cutDirMl;
  bool hasActiveChi2Matching{false};
  std::string activeChi2FunctionName;
  double activeChi2MatchingPlaneZ{0.};
  int activeChi2ExtrapMethod{0};

  bool hasActiveMlMatching{false};
  o2::analysis::MlResponseMFTMuonMatch<float> activeMlResponse;
  double activeMlMatchingPlaneZ{0.};

  int mRunNumber{0}; // needed to detect if the run changed and trigger update of magnetic field

  Service<o2::ccdb::BasicCCDBManager> ccdbManager;
  o2::ccdb::CcdbApi fCCDBApi;

  // vector of all MFT-MCH(-MID) matching candidates associated to the same MCH(-MID) track,
  // to be sorted in descending order with respect to the matching score
  // the map key is the MCH(-MID) track global index
  using MatchingCandidates = std::map<int64_t, std::vector<MatchingCandidate>>;

  struct CollisionInfo {
    int64_t index{0};
    uint64_t bc{0};
    // z position of the collision
    double zVertex{0};
    // number of MFT tracks associated to the collision
    int mftTracksMultiplicity{0};
    // vector of MFT track indexes
    std::vector<int64_t> mftTracks;
    // vector of MCH(-MID) track indexes
    std::vector<int64_t> mchTracks;
    // matching candidates
    MatchingCandidates matchingCandidates;
  };

  using CollisionInfos = std::map<int64_t, CollisionInfo>;

  class TrackParExt: public o2::track::TrackParCovFwd
  {
  public:
    TrackParExt() = default;
    TrackParExt(const TrackParExt& t) = default;
    //TrackParExt(const GlobalFwdTrack& t) { *this = t; }
    TrackParExt(o2::track::TrackParCovFwd const& t) { *this = t; }
    ~TrackParExt() = default;

    void setNClusters(int n) { nClusters = n; }
    int getNClusters() const { return nClusters; }

    void setRemovable() { removable = true; }
    bool isRemovable() const { return removable; }
  private:
    int nClusters{-1};
    bool removable{false};
  };
  std::unordered_map<int64_t, TrackParExt> mMchTrackPars;

  std::unordered_map<int64_t, int32_t> mftTrackCovs;

  Produces<o2::aod::GlobalMuonMatchCandidates> globalMuonMatchCandidates;
  Produces<o2::aod::FwdTrkMatchCands> fwdTrkMatchCands;
  Produces<o2::aod::StoredFwdTracksReAlign> gmCandidateFwdTracks;
  Produces<o2::aod::StoredFwdTrksCovReAlign> gmCandidateFwdTracksCov;

  int32_t mMatchCandidateCounter{0};
  std::unordered_map<int64_t, std::vector<int32_t>> mMchTrackToCandidateIndices;
  std::unordered_map<int64_t, std::vector<MatchingCandidate>> mMchTrackMatchingCandidates;
  std::unordered_map<int64_t, bool> mMchTrackIsTagged;
  std::unordered_map<int64_t, int32_t> mFwdTrackToGmmCandTrkIndex;

  CollisionInfos fCollisionInfos;

  mch::TrackFitter trackFitter; // Track fitter from MCH tracking library
  mch::geo::TransformationCreator transformation;
  std::map<int, math_utils::Transform3D> transformRef; // reference geometry w.r.t track data
  std::map<int, math_utils::Transform3D> transformNew; // new geometry
  double mImproveCutChi2; // Chi2 cut for track improvement.
  TGeoManager* geoNew = nullptr;
  TGeoManager* geoRef = nullptr;
  globaltracking::MatchGlobalFwd mMatching;

  Preslice<aod::FwdTrkCl> perMuon = aod::fwdtrkcl::fwdtrackId;

  int GetDetElemId(int iDetElemNumber)
  {
    // make sure detector number is valid
    if (!(iDetElemNumber >= fgSNDetElemCh[0] &&
          iDetElemNumber < fgSNDetElemCh[10])) {
      LOGF(fatal, "Invalid detector element number: %d", iDetElemNumber);
    }
    /// get det element number from ID
    // get chamber and element number in chamber
    int iCh = 0;
    int iDet = 0;
    for (int i = 1; i <= 10; i++) {
      if (iDetElemNumber < fgSNDetElemCh[i]) {
        iCh = i;
        iDet = iDetElemNumber - fgSNDetElemCh[i - 1];
        break;
      }
    }

    // make sure detector index is valid
    if (!(iCh > 0 && iCh <= 10 && iDet < fgNDetElemCh[iCh - 1])) {
      LOGF(fatal, "Invalid detector element id: %d", 100 * iCh + iDet);
    }

    // add number of detectors up to this chamber
    return 100 * iCh + iDet;
  }

  bool RemoveTrack(mch::Track& track)
  {
    // Refit track with re-aligned clusters
    bool removeTrack = false;
    try {
      trackFitter.fit(track, false);
    } catch (std::exception const& e) {
      removeTrack = true;
      return removeTrack;
    }

    auto itStartingParam = std::prev(track.rend());

    while (true) {

      try {
        trackFitter.fit(track, true, false, (itStartingParam == track.rbegin()) ? nullptr : &itStartingParam);
      } catch (std::exception const&) {
        removeTrack = true;
        break;
      }

      double worstLocalChi2 = -1.0;

      track.tagRemovableClusters(0x1F, false);

      auto itWorstParam = track.end();

      for (auto itParam = track.begin(); itParam != track.end(); ++itParam) {
        if (itParam->getLocalChi2() > worstLocalChi2) {
          worstLocalChi2 = itParam->getLocalChi2();
          itWorstParam = itParam;
        }
      }

      if (worstLocalChi2 < mImproveCutChi2) {
        break;
      }

      if (!itWorstParam->isRemovable()) {
        removeTrack = true;
        track.removable();
        break;
      }

      auto itNextParam = track.removeParamAtCluster(itWorstParam);
      auto itNextToNextParam = (itNextParam == track.end()) ? itNextParam : std::next(itNextParam);
      itStartingParam = track.rbegin();

      if (track.getNClusters() < 10) {
        removeTrack = true;
        break;
      } else {
        while (itNextToNextParam != track.end()) {
          if (itNextToNextParam->getClusterPtr()->getChamberId() != itNextParam->getClusterPtr()->getChamberId()) {
            itStartingParam = std::make_reverse_iterator(++itNextParam);
            break;
          }
          ++itNextToNextParam;
        }
      }
    }

    if (!removeTrack) {
      for (auto& param : track) {
        param.setParameters(param.getSmoothParameters());
        param.setCovariances(param.getSmoothCovariances());
      }
    }

    return removeTrack;
  }

  template <typename BC>
  void initCcdb(BC const& bc)
  {
    if (mRunNumber == bc.runNumber())
      return;

    mRunNumber = bc.runNumber();
    std::map<std::string, std::string> metadata;
    auto soreor = o2::ccdb::BasicCCDBManager::getRunDuration(fCCDBApi, mRunNumber);
    auto ts = soreor.first;
    auto grpmag = fCCDBApi.retrieveFromTFileAny<o2::parameters::GRPMagField>(grpMagPath, metadata, ts);
    o2::base::Propagator::initFieldFromGRP(grpmag);
    LOGF(info, "Set field for muons");
    VarManager::SetupMuonMagField();
    if (!o2::base::GeometryManager::isGeometryLoaded()) {
      ccdbManager->get<TGeoManager>(geoPath);
    }
    mch::TrackExtrap::setField();
    mch::TrackExtrap::useExtrapV2();

    // Load geometry information from CCDB/local
    LOGF(info, "Loading reference aligned geometry from CCDB no later than %d", nolaterthanRef.value);
    ccdbManager->setCreatedNotAfter(nolaterthanRef.value); // this timestamp has to be consistent with what has been used in reco
    geoRef = ccdbManager->getForTimeStamp<TGeoManager>(geoRefPath, bc.timestamp());
    ccdbManager->clearCache(geoRefPath);
    if (geoRef != nullptr) {
      transformation = mch::geo::transformationFromTGeoManager(*geoRef);
    } else {
      LOGF(fatal, "Reference aligned geometry object is not available in CCDB at timestamp=%llu", bc.timestamp());
    }
    for (int i = 0; i < 156; i++) {
      int iDEN = GetDetElemId(i);
      transformRef[iDEN] = transformation(iDEN);
    }

    LOGF(info, "Loading new aligned geometry from CCDB no later than %d", nolaterthanNew.value);
    ccdbManager->setCreatedNotAfter(nolaterthanNew.value); // make sure this timestamp can be resolved regarding the reference one
    geoNew = ccdbManager->getForTimeStamp<TGeoManager>(geoNewPath, bc.timestamp());
    ccdbManager->clearCache(geoNewPath);
    if (geoNew != nullptr) {
      transformation = mch::geo::transformationFromTGeoManager(*geoNew);
    } else {
      LOGF(fatal, "New aligned geometry object is not available in CCDB at timestamp=%llu", bc.timestamp());
    }
    for (int i = 0; i < 156; i++) {
      int iDEN = GetDetElemId(i);
      transformNew[iDEN] = transformation(iDEN);
    }

    // Init magnetic field for MFT track extrapolation
    auto* fieldB = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    if (fieldB) {
      double centerMft[3] = {0, 0, -61.4}; // Field at center of MFT
      mBzAtMftCenter = fieldB->getBz(centerMft);
      // std::cout << "fieldB: " << (void*)fieldB << std::endl;
    }
  }

  void initMatchingFunctions()
  {
    using SMatrix55Std = ROOT::Math::SMatrix<double, 5>;
    using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;

    using SVector2 = ROOT::Math::SVector<double, 2>;
    using SVector4 = ROOT::Math::SVector<double, 4>;
    using SVector5 = ROOT::Math::SVector<double, 5>;

    using SMatrix44 = ROOT::Math::SMatrix<double, 4>;
    using SMatrix45 = ROOT::Math::SMatrix<double, 4, 5>;
    using SMatrix22 = ROOT::Math::SMatrix<double, 2>;
    using SMatrix25 = ROOT::Math::SMatrix<double, 2, 5>;

    // Define built-in matching functions
    //________________________________________________________________________________
    mMatchingFunctionMap["matchALL"] = [](const o2::dataformats::GlobalFwdTrack& mchTrack, const o2::track::TrackParCovFwd& mftTrack) -> std::tuple<double, int> {
      // Match two tracks evaluating all parameters: X,Y, phi, tanl & q/pt

      SMatrix55Sym hK, vK;
      SVector5 mK(mftTrack.getX(), mftTrack.getY(), mftTrack.getPhi(),
                  mftTrack.getTanl(), mftTrack.getInvQPt()),
        rKKminus1;
      SVector5 globalMuonTrackParameters = mchTrack.getParameters();
      SMatrix55Sym globalMuonTrackCovariances = mchTrack.getCovariances();
      vK(0, 0) = mftTrack.getCovariances()(0, 0);
      vK(1, 1) = mftTrack.getCovariances()(1, 1);
      vK(2, 2) = mftTrack.getCovariances()(2, 2);
      vK(3, 3) = mftTrack.getCovariances()(3, 3);
      vK(4, 4) = mftTrack.getCovariances()(4, 4);
      hK(0, 0) = 1.0;
      hK(1, 1) = 1.0;
      hK(2, 2) = 1.0;
      hK(3, 3) = 1.0;
      hK(4, 4) = 1.0;

      // Covariance of residuals
      SMatrix55Std invResCov = (vK + ROOT::Math::Similarity(hK, globalMuonTrackCovariances));
      invResCov.Invert();

      // Update Parameters
      rKKminus1 = mK - hK * globalMuonTrackParameters; // Residuals of prediction

      auto matchChi2Track = ROOT::Math::Similarity(rKKminus1, invResCov);

      // return chi2 and NDF
      return {matchChi2Track, 5};
    };

    //________________________________________________________________________________
    mMatchingFunctionMap["matchXYPhiTanl"] = [](const o2::dataformats::GlobalFwdTrack& mchTrack, const o2::track::TrackParCovFwd& mftTrack) -> std::tuple<double, int> {
      // Match two tracks evaluating positions & angles

      SMatrix45 hK;
      SMatrix44 vK;
      SVector4 mK(mftTrack.getX(), mftTrack.getY(), mftTrack.getPhi(),
                  mftTrack.getTanl()),
        rKKminus1;
      SVector5 globalMuonTrackParameters = mchTrack.getParameters();
      SMatrix55Sym globalMuonTrackCovariances = mchTrack.getCovariances();
      vK(0, 0) = mftTrack.getCovariances()(0, 0);
      vK(1, 1) = mftTrack.getCovariances()(1, 1);
      vK(2, 2) = mftTrack.getCovariances()(2, 2);
      vK(3, 3) = mftTrack.getCovariances()(3, 3);
      hK(0, 0) = 1.0;
      hK(1, 1) = 1.0;
      hK(2, 2) = 1.0;
      hK(3, 3) = 1.0;

      // Covariance of residuals
      SMatrix44 invResCov = (vK + ROOT::Math::Similarity(hK, globalMuonTrackCovariances));
      invResCov.Invert();

      // Residuals of prediction
      rKKminus1 = mK - hK * globalMuonTrackParameters;

      auto matchChi2Track = ROOT::Math::Similarity(rKKminus1, invResCov);

      // return chi2 and NDF
      return {matchChi2Track, 4};
    };

    //________________________________________________________________________________
    mMatchingFunctionMap["matchXY"] = [](const o2::dataformats::GlobalFwdTrack& mchTrack, const o2::track::TrackParCovFwd& mftTrack) -> std::tuple<double, int> {
      // Calculate Matching Chi2 - X and Y positions

      SMatrix25 hK;
      SMatrix22 vK;
      SVector2 mK(mftTrack.getX(), mftTrack.getY()), rKKminus1;
      SVector5 globalMuonTrackParameters = mchTrack.getParameters();
      SMatrix55Sym globalMuonTrackCovariances = mchTrack.getCovariances();
      vK(0, 0) = mftTrack.getCovariances()(0, 0);
      vK(1, 1) = mftTrack.getCovariances()(1, 1);
      hK(0, 0) = 1.0;
      hK(1, 1) = 1.0;

      // Covariance of residuals
      SMatrix22 invResCov = (vK + ROOT::Math::Similarity(hK, globalMuonTrackCovariances));
      invResCov.Invert();

      // Residuals of prediction
      rKKminus1 = mK - hK * globalMuonTrackParameters;
      auto matchChi2Track = ROOT::Math::Similarity(rKKminus1, invResCov);

      // return reduced chi2
      return {matchChi2Track, 2};
    };
  }

  void init(o2::framework::InitContext&)
  {
    // Load geometry
    ccdbManager->setURL(ccdbUrl);
    ccdbManager->setCaching(true);
    ccdbManager->setLocalObjectValidityChecking();
    fCCDBApi.init(ccdbUrl);
    mRunNumber = 0;

    // Configuration for track fitter
    const auto& trackerParam = mch::TrackerParam::Instance();
    trackFitter.setBendingVertexDispersion(trackerParam.bendingVertexDispersion);
    trackFitter.setChamberResolution(cfgChamberResolutionX.value, cfgChamberResolutionY.value);
    trackFitter.smoothTracks(true);
    trackFitter.useChamberResolution();
    mImproveCutChi2 = 2. * cfgSigmaCutImprove.value * cfgSigmaCutImprove.value;

    if (!o2::base::GeometryManager::isGeometryLoaded()) {
      LOGF(info, "Load geometry from CCDB");
      ccdbManager->get<TGeoManager>(geoPath);
    }

    // Reset matching configuration, then populate only what we need.
    hasActiveChi2Matching = false;
    activeChi2FunctionName.clear();
    activeChi2MatchingPlaneZ = 0.;
    activeChi2ExtrapMethod = 0;

    hasActiveMlMatching = false;
    activeMlMatchingPlaneZ = 0.;

    if (cfgCustomMatchingStrategy.value == 0) {
      // Matching functions (custom chi2)
      initMatchingFunctions();
      auto label = configChi2MatchingOptions.cfgChi2FunctionLabel.value;
      auto funcName = configChi2MatchingOptions.cfgChi2FunctionName.value;
      auto matchingPlaneZ = configChi2MatchingOptions.cfgChi2FunctionMatchingPlaneZ.value;
      auto extrapMethod = configChi2MatchingOptions.cfgChi2MatchingExtrapMethod.value;

      if (label != "" && funcName != "") {
        hasActiveChi2Matching = true;
        activeChi2FunctionName = funcName;
        activeChi2MatchingPlaneZ = matchingPlaneZ;
        activeChi2ExtrapMethod = extrapMethod;
      }
    } else {
      // Matching ML models (custom ML)
      // TODO : for now we use hard coded values since the current models use 1 pT bin
      binsPtMl = {-1e-6, 1000.0};
      cutValues = {0.0};
      cutDirMl = {cuts_ml::CutNot};
      o2::framework::LabeledArray<double> mycutsMl(cutValues.data(), 1, 1, std::vector<std::string>{"pT bin 0"}, std::vector<std::string>{"score"});

      auto label = configMlOptions.cfgMlModelLabel.value;
      auto modelPath = configMlOptions.cfgMlModelPathCcdb.value;
      auto inputFeatures = configMlOptions.cfgMlInputFeatures.value;
      auto modelName = configMlOptions.cfgMlModelName.value;
      auto matchingPlaneZ = configMlOptions.cfgMlModelMatchingPlaneZ.value;
      auto extrapMethod = configMlOptions.cfgMlMatchingExtrapMethod.value;

      if (label != "" && modelPath != "" && !inputFeatures.empty() && modelName != "") {
        activeMlResponse.configure(binsPtMl, mycutsMl, cutDirMl, 1);
        activeMlResponse.setModelPathsCCDB(std::vector<std::string>{modelName}, fCCDBApi, std::vector<std::string>{modelPath}, configCcdb.cfgCcdbNoLaterThan.value);
        activeMlResponse.cacheInputFeaturesIndices(inputFeatures);
        activeMlResponse.init();

        hasActiveMlMatching = true;
        activeMlMatchingPlaneZ = matchingPlaneZ;
        (void)extrapMethod;
      }
    }
  }

  template <class T, class C>
  bool pDcaCut(const T& mchTrack, const C& collision, double nSigmaPDCA)
  {
    static const double sigmaPDCA23 = 80.;
    static const double sigmaPDCA310 = 54.;
    static const double relPRes = 0.0004;
    static const double slopeRes = 0.0005;

    constexpr double AbsorberEndZ = 505.;
    constexpr double RadToDeg = 180. / o2::constants::math::PI;
    double thetaAbs = std::atan(mchTrack.rAtAbsorberEnd() / AbsorberEndZ) * RadToDeg;

    // propagate muon track to vertex
    auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToVertex);

    // double pUncorr = mchTrack.p();
    double p = mchTrackAtVertex.getP();

    double pDCA = mchTrack.pDca();
    double sigmaPDCA = (thetaAbs < ThetaAbsBoundaryDeg) ? sigmaPDCA23 : sigmaPDCA310;
    double nrp = nSigmaPDCA * relPRes * p;
    double pResEffect = sigmaPDCA / (1. - nrp / (1. + nrp));
    double slopeResEffect = SlopeResolutionZ * slopeRes * p;
    double sigmaPDCAWithRes = std::sqrt(pResEffect * pResEffect + slopeResEffect * slopeResEffect);
    if (pDCA > nSigmaPDCA * sigmaPDCAWithRes) {
      return false;
    }

    return true;
  }

  template <class T, class C>
  bool isGoodMuon(const T& mchTrack, const C& collision,
                  double chi2Cut,
                  double pCut,
                  double pTCut,
                  std::array<double, 2> etaCut,
                  std::array<double, 2> rAbsCut,
                  double nSigmaPdcaCut)
  {
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
    if (!pDcaCut(mchTrack, collision, nSigmaPdcaCut)) {
      return false;
    }

    return true;
  }

  template <class T, class C>
  bool isGoodMuon(const T& muonTrack, const C& collision)
  {
    return isGoodMuon(muonTrack, collision, cfgTrackChi2MchUp, cfgPMchLow, cfgPtMchLow, {cfgEtaMchLow, cfgEtaMchUp}, {cfgRabsLow, cfgRabsUp}, cfgPdcaUp);
  }

  template <class T, class C>
  bool isGoodGlobalMuon(const T& muonTrack, const C& collision)
  {
    return isGoodMuon(muonTrack, collision, cfgTrackChi2MchUp, cfgPMchLow, cfgPtMchLow, {cfgEtaMftLow, cfgEtaMftUp}, {cfgRabsLow, cfgRabsUp}, cfgPdcaUp);
  }

  template <typename T>
  void storeFwdTrackCovarianceFromTrack(const T& track)
  {
    std::vector<double> v1{track.cXX(), track.cXY(), track.cYY(), track.cPhiX(), track.cPhiY(),
                           track.cPhiPhi(), track.cTglX(), track.cTglY(), track.cTglPhi(), track.cTglTgl(),
                           track.c1PtX(), track.c1PtY(), track.c1PtPhi(), track.c1PtTgl(), track.c1Pt21Pt2()};
    SMatrix55 tcovs(v1.begin(), v1.end());
    storeFwdTrackCovariance(tcovs);
  }

  void storeFwdTrackCovariance(const SMatrix55& cov)
  {
    const float sigX = std::sqrt(cov(0, 0));
    const float sigY = std::sqrt(cov(1, 1));
    const float sigPhi = std::sqrt(cov(2, 2));
    const float sigTgl = std::sqrt(cov(3, 3));
    const float sig1Pt = std::sqrt(cov(4, 4));
    const auto rhoXY = static_cast<int8_t>(128.f * cov(0, 1) / (sigX * sigY));
    const auto rhoPhiX = static_cast<int8_t>(128.f * cov(0, 2) / (sigPhi * sigX));
    const auto rhoPhiY = static_cast<int8_t>(128.f * cov(1, 2) / (sigPhi * sigY));
    const auto rhoTglX = static_cast<int8_t>(128.f * cov(0, 3) / (sigTgl * sigX));
    const auto rhoTglY = static_cast<int8_t>(128.f * cov(1, 3) / (sigTgl * sigY));
    const auto rhoTglPhi = static_cast<int8_t>(128.f * cov(2, 3) / (sigTgl * sigPhi));
    const auto rho1PtX = static_cast<int8_t>(128.f * cov(0, 4) / (sig1Pt * sigX));
    const auto rho1PtY = static_cast<int8_t>(128.f * cov(1, 4) / (sig1Pt * sigY));
    const auto rho1PtPhi = static_cast<int8_t>(128.f * cov(2, 4) / (sig1Pt * sigPhi));
    const auto rho1PtTgl = static_cast<int8_t>(128.f * cov(3, 4) / (sig1Pt * sigTgl));
    gmCandidateFwdTracksCov(sigX, sigY, sigPhi, sigTgl, sig1Pt,
                                         rhoXY, rhoPhiY, rhoPhiX, rhoTglX, rhoTglY, rhoTglPhi, rho1PtX, rho1PtY, rho1PtPhi, rho1PtTgl);
  }

  template <class TMCH>
  void fillBaseGmmCandFwdTrack(TMCH const& track,
                               int32_t gmmMchTrackId,
                               int32_t matchRanking,
                               bool isTagged,
                               float chi2MatchMCHMFT,
                               float matchScoreMCHMFT)
  {
    const int32_t collisionId = track.has_collision() ? track.collisionId() : -1;

    bool isRemovable = false;

    gmCandidateFwdTracks(
      collisionId,
      track.trackType(),
      track.x(),
      track.y(),
      track.z(),
      track.phi(),
      track.tgl(),
      track.signed1Pt(),
      track.nClusters(),
      track.pDca(),
      track.rAtAbsorberEnd(),
      isRemovable,
      track.chi2(),
      track.chi2MatchMCHMID(),
      chi2MatchMCHMFT,
      matchScoreMCHMFT,
      //matchRanking,
      //isTagged,
      track.matchMFTTrackId(),
      gmmMchTrackId,
      track.mchBitMap(),
      track.midBitMap(),
      track.midBoards(),
      track.trackTime(),
      track.trackTimeRes());

    storeFwdTrackCovarianceFromTrack(track);
  }

  template <class TMCH>
  void fillBaseGmmCandFwdTrack(TMCH const& track,
                               TrackParExt const& fwdtrack,
                               int32_t gmmMchTrackId,
                               int32_t matchRanking,
                               bool isTagged,
                               float chi2MatchMCHMFT,
                               float matchScoreMCHMFT)
  {
    const int32_t collisionId = track.has_collision() ? track.collisionId() : -1;

    bool isRemovable = false;

    gmCandidateFwdTracks(
      collisionId,
      track.trackType(),
      fwdtrack.getX(),
      fwdtrack.getY(),
      fwdtrack.getZ(),
      fwdtrack.getPhi(),
      fwdtrack.getTgl(),
      fwdtrack.getInvQPt(),
      fwdtrack.getNClusters(),
      track.pDca(),
      track.rAtAbsorberEnd(),
      fwdtrack.isRemovable(),
      fwdtrack.getTrackChi2(),
      track.chi2MatchMCHMID(),
      chi2MatchMCHMFT,
      matchScoreMCHMFT,
      //matchRanking,
      //isTagged,
      track.matchMFTTrackId(),
      gmmMchTrackId,
      track.mchBitMap(),
      track.midBitMap(),
      track.midBoards(),
      track.trackTime(),
      track.trackTimeRes());

    storeFwdTrackCovarianceFromTrack(track);
  }

  template <class TCOLLISION, class TMCH, class TMFT, class CMFT>
  void fillCandidateFwdTrack(TCOLLISION const& collision,
                             TMCH const& mchTrack,
                             int32_t gmmMchTrackId,
                             TMFT const& mftTrack,
                             CMFT const& mftCovs,
                             const MatchingCandidate& candidate,
                             bool isTagged)
  {
    using o2::aod::fwdtrack::ForwardTrackTypeEnum;
    using o2::aod::fwdtrackutils::propagationPoint;

    constexpr float matchingZ = 0.f;
    constexpr uint8_t candidateTrackType = static_cast<uint8_t>(ForwardTrackTypeEnum::GlobalForwardTrack);
    const float bz = static_cast<float>(mBzAtMftCenter);

    const auto propmuonAtPV = o2::aod::fwdtrackutils::propagateMuon(mchTrack, mchTrack, collision, propagationPoint::kToVertex, matchingZ, bz);

    o2::track::TrackParCovFwd mftPar = o2::aod::fwdtrackutils::getTrackParCovFwdShift(mftTrack, 0.f);
    if (mftTrackCovs.count(mftTrack.globalIndex()) > 0) {
      const auto& mftCov = mftCovs.rawIteratorAt(mftTrackCovs.at(mftTrack.globalIndex()));
      mftPar = o2::aod::fwdtrackutils::getTrackParCovFwd(mftTrack, mftCov);
    }

    const auto globalMuonRefit = o2::aod::fwdtrackutils::refitGlobalMuonCov(propmuonAtPV, mftPar);

    int8_t nClusters = mchTrack.nClusters();
    if constexpr (requires { mftTrack.nClusters(); }) {
      nClusters = static_cast<int8_t>(std::min(127, static_cast<int>(mchTrack.nClusters()) + static_cast<int>(mftTrack.nClusters())));
    }

    const float chi2 = static_cast<float>(mchTrack.chi2());
    const int32_t collisionId = mchTrack.has_collision() ? mchTrack.collisionId() : -1;

    bool isRemovable = false;

    gmCandidateFwdTracks(
      collisionId,
      candidateTrackType,
      globalMuonRefit.getX(),
      globalMuonRefit.getY(),
      globalMuonRefit.getZ(),
      globalMuonRefit.getPhi(),
      globalMuonRefit.getTgl(),
      globalMuonRefit.getInvQPt(),
      nClusters,
      mchTrack.pDca(),
      mchTrack.rAtAbsorberEnd(),
      isRemovable,
      chi2,
      mchTrack.chi2MatchMCHMID(),
      static_cast<float>(candidate.matchChi2),
      static_cast<float>(candidate.matchScore),
      //static_cast<int32_t>(candidate.matchRanking),
      //isTagged,
      static_cast<int>(mftTrack.globalIndex()),
      gmmMchTrackId,
      mchTrack.mchBitMap(),
      mchTrack.midBitMap(),
      mchTrack.midBoards(),
      mchTrack.trackTime(),
      mchTrack.trackTimeRes());

    storeFwdTrackCovariance(globalMuonRefit.getCovariances());
  }
/*
  template <typename T>
  o2::dataformats::GlobalFwdTrack fwdToTrackPar(const T& track)
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

  template <typename T, typename C>
  o2::dataformats::GlobalFwdTrack fwdToTrackPar(const T& track, const C& cov)
  {
    double chi2 = track.chi2();
    SMatrix5 tpars(track.x(), track.y(), track.phi(), track.tgl(), track.signed1Pt());
    std::vector<double> v1{cov.cXX(), cov.cXY(), cov.cYY(), cov.cPhiX(), cov.cPhiY(),
                           cov.cPhiPhi(), cov.cTglX(), cov.cTglY(), cov.cTglPhi(), cov.cTglTgl(),
                           cov.c1PtX(), cov.c1PtY(), cov.c1PtPhi(), cov.c1PtTgl(), cov.c1Pt21Pt2()};
    SMatrix55 tcovs(v1.begin(), v1.end());
    o2::track::TrackParCovFwd trackparCov{track.z(), tpars, tcovs, chi2};
    o2::dataformats::GlobalFwdTrack fwdtrack;
    fwdtrack.setParameters(trackparCov.getParameters());
    fwdtrack.setZ(trackparCov.getZ());
    fwdtrack.setCovariances(trackparCov.getCovariances());
    return fwdtrack;
  }
*/
  o2::track::TrackParCovFwd propagateToZMchPar(const o2::track::TrackParCovFwd& muon, const double z)
  {
    auto mchTrack = mExtrap.FwdtoMCH(muon);

    float absFront = -90.f;
    float absBack = -505.f;

    if (muon.getZ() < absBack && z > absFront) {
      // extrapolation through the absorber in the upstream direction
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrack, z);
    } else {
      // all other cases
      o2::mch::TrackExtrap::extrapToZCov(mchTrack, z);
    }

    auto proptrack = mExtrap.MCHtoFwd(mchTrack);
    //o2::dataformats::GlobalFwdTrack propmuon;
    //propmuon.setParameters(proptrack.getParameters());
    //propmuon.setZ(proptrack.getZ());
    //propmuon.setCovariances(proptrack.getCovariances());

    return proptrack;
  }

  template <typename T>
  o2::track::TrackParCovFwd propagateToZMch(const T& muon, const double z)
  {
    double chi2 = muon.chi2();
    SMatrix5 tpars(muon.x(), muon.y(), muon.phi(), muon.tgl(), muon.signed1Pt());
    std::vector<double> v1{muon.cXX(), muon.cXY(), muon.cYY(), muon.cPhiX(), muon.cPhiY(),
                           muon.cPhiPhi(), muon.cTglX(), muon.cTglY(), muon.cTglPhi(), muon.cTglTgl(),
                           muon.c1PtX(), muon.c1PtY(), muon.c1PtPhi(), muon.c1PtTgl(), muon.c1Pt21Pt2()};
    SMatrix55 tcovs(v1.begin(), v1.end());
    o2::track::TrackParCovFwd fwdtrack{muon.z(), tpars, tcovs, chi2};
    //o2::dataformats::GlobalFwdTrack track;
    //track.setParameters(fwdtrack.getParameters());
    //track.setZ(fwdtrack.getZ());
    //track.setCovariances(fwdtrack.getCovariances());

    return propagateToZMchPar(fwdtrack, z);
  }

  o2::track::TrackParCovFwd propagateToZMftPar(const o2::track::TrackParCovFwd& mftTrack, const double z)
  {
    o2::track::TrackParCovFwd trackExtrap{mftTrack};
    trackExtrap.propagateToZ(z, mBzAtMftCenter);
    return trackExtrap;
  }

  template <typename TMFT, typename CMFT>
  o2::track::TrackParCovFwd propagateToZMft(const TMFT& mftTrack, const CMFT& mftCov, const double z)
  {
    auto fwdtrack = fwdtrackutils::getTrackParCovFwd(mftTrack, mftCov);
    return propagateToZMftPar(fwdtrack, z);
  }

  // method 0: standard extrapolation
  // method 1: MFT extrapolation using MCH momentum
  // method 2: MCH track extrapolation constrained to the first MFT track point, MFT extrapolation using MCH momentum
  // method 3: MCH track extrapolation constrained to the collision point, MFT extrapolation using MCH momentum
  template <typename TMCH, typename TMFT, typename CMFT, typename C>
  o2::track::TrackParCovFwd propagateToMatchingPlaneMch(const TMCH& mchTrack, const TMFT& mftTrack, const CMFT& mftTrackCov, const C& collision, const double z, int method)
  {
    if (method == ExtrapolationMethodStandard || method == 1) {
      // simple extrapolation upstream through the absorber
      return propagateToZMch(mchTrack, z);
    }

    if (method == ExtrapolationMethodMftFirstPoint) {
      // extrapolation to the first MFT point and then back to the matching plane
      auto mftTrackPar = fwdtrackutils::getTrackParCovFwd(mftTrack, mftTrackCov);
      // std::cout << std::format("[propagateToMatchingPlaneMch] extrapolating to MFT: x={:0.3f} y={:0.3f} z={:0.3f}", mftTrackPar.getX(), mftTrackPar.getY(), mftTrackPar.getZ()) << std::endl;
      auto mchTrackAtMFT = propagateToVertexMch(fwdtrackutils::getTrackParCovFwd(mchTrack, mchTrack),
                                                mftTrackPar.getX(), mftTrackPar.getY(), mftTrackPar.getZ(),
                                                mftTrackPar.getSigma2X(), mftTrackPar.getSigma2Y());
      // std::cout << std::format("[propagateToMatchingPlaneMch] extrapolating to z={:0.3f}", z) << std::endl;
      return propagateToZMchPar(mchTrackAtMFT, z);
    }

    if (method == ExtrapolationMethodVertex) {
      // extrapolation to the vertex and then back to the matching plane
      //auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToVertex);
      auto mchTrackAtVertex =  propagateToVertexMch(fwdtrackutils::getTrackParCovFwd(mchTrack, mchTrack),
                                                    collision.posX(), collision.posY(), collision.posZ(),
                                                    collision.covXX(), collision.covYY());
      return propagateToZMchPar(mchTrackAtVertex, z);
    }

    if (method == ExtrapolationMethodMftDca) {
      // extrapolation to the MFT DCA and then back to the matching plane
      auto mftTrackDCA = propagateToZMftPar(fwdtrackutils::getTrackParCovFwd(mftTrack, mftTrackCov), collision.posZ());
      auto mchTrackAtDCA = propagateToVertexMch(fwdtrackutils::getTrackParCovFwd(mchTrack, mchTrack),
                                                mftTrackDCA.getX(), mftTrackDCA.getY(), mftTrackDCA.getZ(),
                                                mftTrackDCA.getSigma2X(), mftTrackDCA.getSigma2Y());
      return propagateToZMchPar(mchTrackAtDCA, z);
    }

    return {};
  }

  template <typename TMCH, typename TMFT, typename CMFT, typename C>
  o2::track::TrackParCovFwd propagateToMatchingPlaneMft(const TMCH& mchTrack, const TMFT& mftTrack, const CMFT& mftTrackCov, const C& collision, const double z, int method)
  {
    if (method == ExtrapolationMethodStandard) {
      // extrapolation with MFT tools
      return propagateToZMft(mftTrack, mftTrackCov, z);
    }

    if (method > 0) {
      // extrapolation with MCH tools
      auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToVertex);
      double pMCH = mchTrackAtVertex.getP();
      double px = pMCH * std::sin(o2::constants::math::PIHalf - std::atan(mftTrack.tgl())) * std::cos(mftTrack.phi());
      double py = pMCH * std::sin(o2::constants::math::PIHalf - std::atan(mftTrack.tgl())) * std::sin(mftTrack.phi());
      double pt = std::hypot(px, py);
      double sign = mchTrack.sign();

      auto track = fwdtrackutils::getTrackParCovFwd(mftTrack, mftTrackCov);

      // update momentum in track parameters and errors
      auto newCov = track.getCovariances();
      newCov(4, 4) = mchTrackAtVertex.getSigma2InvQPt();
      track.setCovariances(newCov);
      track.setInvQPt(sign / pt);

      auto trackExt = mExtrap.FwdtoMCH(track);
      o2::mch::TrackExtrap::extrapToZCov(trackExt, z);

      auto proptrack = mExtrap.MCHtoFwd(trackExt);
      //o2::dataformats::GlobalFwdTrack propmuon;
      //propmuon.setParameters(proptrack.getParameters());
      //propmuon.setZ(proptrack.getZ());
      //propmuon.setCovariances(proptrack.getCovariances());

      return proptrack;
    }

    return {};
  }

  o2::track::TrackParCovFwd propagateToVertexMch(const o2::track::TrackParCovFwd& muon,
                                                       const double vx, const double vy, const double vz,
                                                       const double covVx, const double covVy)
  {
    auto mchTrack = mExtrap.FwdtoMCH(muon);

    o2::mch::TrackExtrap::extrapToVertex(mchTrack, vx, vy, vz, covVx, covVy);

    auto proptrack = mExtrap.MCHtoFwd(mchTrack);
    //o2::dataformats::GlobalFwdTrack propmuon;
    //propmuon.setParameters(proptrack.getParameters());
    //propmuon.setZ(proptrack.getZ());
    //propmuon.setCovariances(proptrack.getCovariances());

    return proptrack;
  }

  template <class TMCH, class C>
  o2::track::TrackParCovFwd propagateToVertexMch(const TMCH& muon,
                                                       const C& collision)
  {
    return propagateToVertexMch(fwdtrackutils::getTrackParCovFwd(muon, muon),
                                collision.posX(),
                                collision.posY(),
                                collision.posZ(),
                                collision.covXX(),
                                collision.covYY());
  }

  o2::track::TrackParCovFwd propagateToVertexMft(o2::track::TrackParCovFwd muon,
                                                       const float vx, const float vy, const float vz,
                                                       const float covVx, const float covVy)
  {
    auto geoMan = o2::base::GeometryManager::meanMaterialBudget(muon.getX(), muon.getY(), muon.getZ(), vx, vy, vz);
    auto x2x0 = static_cast<float>(geoMan.meanX2X0);
    muon.propagateToVtxhelixWithMCS(vz, {vx, vy}, {covVx, covVy}, mBzAtMftCenter, x2x0);
    //o2::dataformats::GlobalFwdTrack propmuon;
    //propmuon.setParameters(muon.getParameters());
    //propmuon.setZ(muon.getZ());
    //propmuon.setCovariances(muon.getCovariances());

    return muon;
  }


  template <typename TMCH, typename TMFT, class C>
  o2::track::TrackParCovFwd propagateToVertexMft(const TMFT& mftTrack,
                                                       const TMCH& mchTrack,
                                                       const C& collision)
  {
    // extrapolation with MCH tools
    auto mchTrackAtMFT = mExtrap.FwdtoMCH(fwdtrackutils::getTrackParCovFwd(mchTrack));
    o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrackAtMFT, mftTrack.z());

    auto mftTrackProp = mExtrap.FwdtoMCH(fwdtrackutils::getTrackParCovFwd(mftTrack));

    // update global track momentum from the MCH track
    double pRatio = mftTrackProp.p() / mchTrackAtMFT.p();
    double newInvBendMom = mftTrackProp.getInverseBendingMomentum() * pRatio;
    mftTrackProp.setInverseBendingMomentum(newInvBendMom);
    mftTrackProp.setCharge(mchTrackAtMFT.getCharge());

    o2::mch::TrackExtrap::extrapToVertex(mftTrackProp,
                                         collision.posX(),
                                         collision.posY(),
                                         collision.posZ(),
                                         collision.covXX(),
                                         collision.covYY());

    return mExtrap.MCHtoFwd(mftTrackProp);
  }

  // tag muons based on the track quality and the track position at the front and back MFT planes
  template <class TMUON, class C>
  void getTaggedMuons(const CollisionInfo& collisionInfo,
                      C const& collisions,
                      TMUON const& muonTracks,
                      std::vector<int64_t>& taggedMuons)
  {
    taggedMuons.clear();
    for (const auto& muonTrack : muonTracks) {

      // only consider MCH-MID matches
      if (static_cast<int>(muonTrack.trackType()) != MchMidTrackType) {
        continue;
      }

      // only select MCH-MID tracks from the current collision
      if (!muonTrack.has_collision())
        continue;
      if (static_cast<int64_t>(muonTrack.collisionId()) != collisionInfo.index)
        continue;

      const auto& collision = collisions.rawIteratorAt(muonTrack.collisionId());

      // select MCH tracks with strict quality cuts
      if (!isGoodMuon(muonTrack, collision,
                      cfgMuonTaggingTrackChi2MchUp,
                      cfgMuonTaggingPMchLow,
                      cfgMuonTaggingPtMchLow,
                      {cfgMuonTaggingEtaMchLow, cfgMuonTaggingEtaMchUp},
                      {cfgMuonTaggingRabsLow, cfgMuonTaggingRabsUp},
                      cfgMuonTaggingPdcaUp)) {
        continue;
      }

      // propagate MCH track to the vertex
      o2::track::TrackParCovFwd mchTrackAtVertex = VarManager::PropagateMuon(muonTrack, collision, VarManager::kToVertex);

      // propagate the track from the vertex to the first MFT plane
      const auto& extrapToMFTfirst = propagateToZMchPar(mchTrackAtVertex, o2::mft::constants::mft::LayerZCoordinate()[0]);
      double rFront = std::sqrt(extrapToMFTfirst.getX() * extrapToMFTfirst.getX() + extrapToMFTfirst.getY() * extrapToMFTfirst.getY());
      if (rFront < cfgMuonTaggingRadiusAtMftFrontLow.value || rFront > cfgMuonTaggingRadiusAtMftFrontUp.value)
        continue;

      // propagate the track from the vertex to the last MFT plane
      const auto& extrapToMFTlast = propagateToZMchPar(mchTrackAtVertex, o2::mft::constants::mft::LayerZCoordinate()[9]);
      double rBack = std::sqrt(extrapToMFTlast.getX() * extrapToMFTlast.getX() + extrapToMFTlast.getY() * extrapToMFTlast.getY());
      if (rBack < cfgMuonTaggingRadiusAtMftBackLow.value || rBack > cfgMuonTaggingRadiusAtMftBackUp.value)
        continue;

      int64_t muonTrackIndex = muonTrack.globalIndex();
      taggedMuons.emplace_back(muonTrackIndex);
    }
  }

  template <class EVT, class BC, class TMUON, class TMFT>
  bool isMftMchTimeCompatible(EVT const& collisions,
                                BC const& bcs,
                                TMUON const& mchTrack,
                                TMFT const& mftTrack)
  {
    if (!mchTrack.has_collision() || !mftTrack.has_collision()) {
      return false;
    }

    const auto& collMch = collisions.rawIteratorAt(mchTrack.collisionId());
    const auto& bcMch = bcs.rawIteratorAt(collMch.bcId());
    const auto& collMft = collisions.rawIteratorAt(mftTrack.collisionId());
    const auto& bcMft = bcs.rawIteratorAt(collMft.bcId());

    int64_t deltaBc = static_cast<int64_t>(bcMft.globalBC()) - static_cast<int64_t>(bcMch.globalBC());
    double deltaBcNS = o2::constants::lhc::LHCBunchSpacingNS * deltaBc;
    double deltaTrackTime = mftTrack.trackTime() - mchTrack.trackTime() + deltaBcNS;
    double trackTimeResTot = mftTrack.trackTimeRes() + mchTrack.trackTimeRes();

    return std::fabs(deltaTrackTime) <= trackTimeResTot;
  }

  template <class EVT, class BC, class TMUON, class TMFT>
  void fillCollisions(EVT const& collisions,
                      BC const& bcs,
                      TMUON const& muonTracks,
                      TMFT const& mftTracks,
                      MyMFTCovariances const& mftCovs,
                      CollisionInfos& collisionInfos)
  {
    collisionInfos.clear();

    std::vector<int64_t> collisionIds;
    for (const auto& collision : collisions) {
      collisionIds.push_back(collision.globalIndex());
    }

    if (collisionIds.empty())
      return;

    LOGF(info, "Filling matching candidate tables");
    for (size_t cid = 0; cid < collisionIds.size(); cid++) {
      const auto& collision = collisions.rawIteratorAt(collisionIds[cid]);
      int64_t collisionIndex = collision.globalIndex();
      auto bc = bcs.rawIteratorAt(collision.bcId());

      auto& collisionInfo = collisionInfos[collisionIndex];
      collisionInfo.index = collisionIndex;
      collisionInfo.bc = bc.globalBC();
      collisionInfo.zVertex = collision.posZ();

      // collect standalone MCH or MCH-MID tracks associated to this collision
      for (const auto& muonTrack : muonTracks) {
        if (!muonTrack.has_collision()) {
          continue;
        }
        if (collisionIndex != muonTrack.collisionId()) {
          continue;
        }
        if (static_cast<int>(muonTrack.trackType()) <= GlobalTrackTypeMax) {
          continue;
        }

        int64_t mchTrackIndex = muonTrack.globalIndex();
        if (std::find(collisionInfo.mchTracks.begin(), collisionInfo.mchTracks.end(), mchTrackIndex) == collisionInfo.mchTracks.end()) {
          collisionInfo.mchTracks.push_back(mchTrackIndex);
        }

        // initialize the MCH track parameters, which will be updated by the realignment if enabled
        auto trackParIt = mMchTrackPars.find(mchTrackIndex);
        if (trackParIt == mMchTrackPars.end()) {
          // add an empty placeholder
          trackParIt = mMchTrackPars.emplace(mchTrackIndex, TrackParExt()).first;
        }
        // assign the parameters of the current track
        trackParIt->second = fwdtrackutils::getTrackParCovFwd(muonTrack, muonTrack);
        trackParIt->second.setNClusters(muonTrack.nClusters());
      }

      // collect MFT standalone tracks associated to this collision
      for (const auto& mftTrack : mftTracks) {
        if (!mftTrack.has_collision()) {
          continue;
        }
        if (collisionIndex != mftTrack.collisionId()) {
          continue;
        }

        int64_t mftTrackIndex = mftTrack.globalIndex();
        if (std::find(collisionInfo.mftTracks.begin(), collisionInfo.mftTracks.end(), mftTrackIndex) == collisionInfo.mftTracks.end()) {
          collisionInfo.mftTracks.push_back(mftTrackIndex);
        }
      }

      // build matching candidates from all time-compatible MFT-MCH pairs
      for (int64_t mchTrackIndex : collisionInfo.mchTracks) {
        const auto& mchTrack = muonTracks.rawIteratorAt(mchTrackIndex);
        for (const auto& mftTrack : mftTracks) {
          if (!isMftMchTimeCompatible(collisions, bcs, mchTrack, mftTrack)) {
            continue;
          }
          if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
            continue;
          }

          collisionInfo.matchingCandidates[mchTrackIndex].emplace_back(MatchingCandidate{
            mftTrack.globalIndex()});
        }
      }
    }
  }

  template <typename TMuons, typename TMuonCls>
  void runMuonRealignment(TMuons const& muons, TMuonCls const& clusters)
  {
    // Loop over forward tracks
    for (auto const& muon : muons) {
      int mchIndex = muon.globalIndex();
      // skip global forward matches
      if (static_cast<int>(muon.trackType() > 2)) {
        continue;
      }

      auto mchTrackParIt = mMchTrackPars.find(mchIndex);
      if (mchTrackParIt == mMchTrackPars.end()) {
        continue;
      }

      auto clustersSliced = clusters.sliceBy(perMuon, muon.globalIndex()); // Slice clusters by muon id
      mch::Track convertedTrack = mch::Track();                            // Temporary variable to store re-aligned clusters
      int clIndex = -1;
      // Get re-aligned clusters associated to current track
      for (auto const& cluster : clustersSliced) {
        clIndex += 1;

        mch::Cluster* clusterMCH = new mch::Cluster();

        math_utils::Point3D<double> local;
        math_utils::Point3D<double> master;
        master.SetXYZ(cluster.x(), cluster.y(), cluster.z());

        // Transformation from reference geometry frame to new geometry frame
        transformRef[cluster.deId()].MasterToLocal(master, local);
        transformNew[cluster.deId()].LocalToMaster(local, master);

        clusterMCH->x = master.x();
        clusterMCH->y = master.y();
        clusterMCH->z = master.z();

        uint32_t ClUId = mch::Cluster::buildUniqueId(static_cast<int>(cluster.deId() / 100) - 1, cluster.deId(), clIndex);
        clusterMCH->uid = ClUId;
        clusterMCH->ex = cluster.isGoodX() ? 0.2 : 10.0;
        clusterMCH->ey = cluster.isGoodY() ? 0.2 : 10.0;

        // Add transformed cluster into temporary variable
        convertedTrack.createParamAtCluster(*clusterMCH);
        //LOGF(debug, "Track %d, cluster DE%d:  x:%g  y:%g  z:%g", muon.globalIndex(), cluster.deId(), cluster.x(), cluster.y(), cluster.z());
        //LOGF(debug, "Track %d, re-aligned cluster DE%d:  x:%g  y:%g  z:%g", muonRealignId, cluster.deId(), clusterMCH->getX(), clusterMCH->getY(), clusterMCH->getZ());
      }

      // Refit the re-aligned track
      int removable = 0;
      if (convertedTrack.getNClusters() != 0) {
        removable = RemoveTrack(convertedTrack);
      } else {
        LOGF(fatal, "Muon track %d has no associated clusters.", muon.globalIndex());
      }

      // Get the re-aligned track parameter: track param at the first cluster
      mch::TrackParam trackParam = mch::TrackParam(convertedTrack.first());

      // Convert MCH track to FWD track and store new parameters after realignment
      mchTrackParIt->second = mMatching.MCHtoFwd(trackParam);
      mchTrackParIt->second.setTrackChi2(trackParam.getTrackChi2() / convertedTrack.getNDF());
      mchTrackParIt->second.setNClusters(convertedTrack.getNClusters());
      if (removable) {
        mchTrackParIt->second.setRemovable();
      }
    }
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void runChi2Matching(C const& collisions,
                       TMUON const& muonTracks,
                       TMFT const& mftTracks,
                       CMFT const& mftCovs,
                       std::string funcName,
                       float matchingPlaneZ,
                       int extrapMethod,
                       int mchTrackIndex,
                       const std::vector<MatchingCandidate>& candidatesVector,
                       std::vector<MatchingCandidate>& newCandidatesVector)
  {
    newCandidatesVector.clear();

    std::string funcNameEffective = funcName;
    float matchingPlaneZEffective = matchingPlaneZ;
    int extrapMethodEffective = extrapMethod;
    if (funcName == "prod") {
      funcNameEffective = "matchALL";
      matchingPlaneZEffective = MatchingPlaneDefaultZ;
      extrapMethodEffective = ExtrapolationMethodStandard;
    }

    if (mMatchingFunctionMap.count(funcNameEffective) < 1) {
      return;
    }
    auto matchingFunc = mMatchingFunctionMap.at(funcNameEffective);

    //for (const auto& [mchIndex, candidatesVector] : matchingCandidates) {
      auto const& mchTrack = muonTracks.rawIteratorAt(mchTrackIndex);
      if (!mchTrack.has_collision()) {
        return;
      }

      auto collision = collisions.rawIteratorAt(mchTrack.collisionId());

      for (const auto& candidate : candidatesVector) {
        auto const& mftTrack = mftTracks.rawIteratorAt(candidate.mftTrackId);
        if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
          continue;
        }
        auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

        auto mftTrackProp = fwdtrackutils::getTrackParCovFwd(mftTrack, mftTrackCov);
        auto mchTrackProp = fwdtrackutils::getTrackParCovFwd(mchTrack, mchTrack);

        if (matchingPlaneZEffective < 0.) {
          mftTrackProp = propagateToMatchingPlaneMft(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZEffective, extrapMethodEffective);
          mchTrackProp = propagateToMatchingPlaneMch(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZEffective, extrapMethodEffective);
        }

        auto matchResult = matchingFunc(mchTrackProp, mftTrackProp);
        float matchChi2 = std::get<0>(matchResult) / std::get<1>(matchResult);
        float matchScore = chi2ToScore(std::get<0>(matchResult), std::get<1>(matchResult), 10.f * std::get<1>(matchResult));

        newCandidatesVector.emplace_back(MatchingCandidate{
          candidate.mftTrackId,
          matchScore,
          matchChi2});
      }
    //}

    auto compareMatchingChi2 = [](const MatchingCandidate& track1, const MatchingCandidate& track2) -> bool {
      return (track1.matchChi2 < track2.matchChi2);
    };

    //for (auto& [mchIndex, globalTracksVector] : newMatchingCandidates) { // o2-linter: disable=const-ref-in-for-loop (object is modified in loop)
      std::sort(newCandidatesVector.begin(), newCandidatesVector.end(), compareMatchingChi2);

      int ranking = 1;
      for (auto& candidate : newCandidatesVector) { // o2-linter: disable=const-ref-in-for-loop (object is modified in loop)
        candidate.matchRanking = ranking;
        ranking += 1;
      }
    //}
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void runChi2Matching(C const& collisions,
                       TMUON const& muonTracks,
                       TMFT const& mftTracks,
                       CMFT const& mftCovs,
                       std::string funcName,
                       float matchingPlaneZ,
                       int extrapMethod,
                       const MatchingCandidates& matchingCandidates,
                       MatchingCandidates& newMatchingCandidates)
  {
    newMatchingCandidates.clear();

    std::string funcNameEffective = funcName;
    float matchingPlaneZEffective = matchingPlaneZ;
    int extrapMethodEffective = extrapMethod;
    if (funcName == "prod") {
      funcNameEffective = "matchALL";
      matchingPlaneZEffective = MatchingPlaneDefaultZ;
      extrapMethodEffective = ExtrapolationMethodStandard;
    }

    if (mMatchingFunctionMap.count(funcNameEffective) < 1) {
      return;
    }
    auto matchingFunc = mMatchingFunctionMap.at(funcNameEffective);

    for (const auto& [mchIndex, candidatesVector] : matchingCandidates) {
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
      if (!mchTrack.has_collision()) {
        continue;
      }

      auto mchTrackParIt = mMchTrackPars.find(mchIndex);
      if (mchTrackParIt == mMchTrackPars.end()) {
        continue;
      }

      auto collision = collisions.rawIteratorAt(mchTrack.collisionId());

      for (const auto& candidate : candidatesVector) {
        auto const& mftTrack = mftTracks.rawIteratorAt(candidate.mftTrackId);
        if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
          continue;
        }
        auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

        auto mftTrackProp = fwdtrackutils::getTrackParCovFwd(mftTrack, mftTrackCov);
        auto mchTrackProp = mchTrackParIt->second;

        if (matchingPlaneZEffective < 0.) {
          mftTrackProp = propagateToMatchingPlaneMft(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZEffective, extrapMethodEffective);
          mchTrackProp = propagateToMatchingPlaneMch(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZEffective, extrapMethodEffective);
        }

        auto matchResult = matchingFunc(mchTrackProp, mftTrackProp);
        float matchChi2 = std::get<0>(matchResult) / std::get<1>(matchResult);
        float matchScore = chi2ToScore(std::get<0>(matchResult), std::get<1>(matchResult), 10.f * std::get<1>(matchResult));

        newMatchingCandidates[mchIndex].emplace_back(MatchingCandidate{
          candidate.mftTrackId,
          matchScore,
          matchChi2});
      }
    }

    auto compareMatchingChi2 = [](const MatchingCandidate& track1, const MatchingCandidate& track2) -> bool {
      return (track1.matchChi2 < track2.matchChi2);
    };

    for (auto& [mchIndex, globalTracksVector] : newMatchingCandidates) { // o2-linter: disable=const-ref-in-for-loop (object is modified in loop)
      std::sort(globalTracksVector.begin(), globalTracksVector.end(), compareMatchingChi2);

      int ranking = 1;
      for (auto& candidate : globalTracksVector) { // o2-linter: disable=const-ref-in-for-loop (object is modified in loop)
        candidate.matchRanking = ranking;
        ranking += 1;
      }
    }
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void runMlMatching(C const& collisions,
                     TMUON const& muonTracks,
                     TMFT const& mftTracks,
                     CMFT const& mftCovs,
                     o2::analysis::MlResponseMFTMuonMatch<float>& mlResponse,
                     float matchingPlaneZ,
                     int mchTrackIndex,
                     const std::vector<MatchingCandidate>& candidatesVector,
                     std::vector<MatchingCandidate>& newCandidatesVector)
  {
    newCandidatesVector.clear();
    //for (const auto& [mchIndex, candidatesVector] : matchingCandidates) {
      auto const& mchTrack = muonTracks.rawIteratorAt(mchTrackIndex);
      if (!mchTrack.has_collision()) {
        return;
      }

      auto collision = collisions.rawIteratorAt(mchTrack.collisionId());

      for (const auto& candidate : candidatesVector) {
        auto const& mftTrack = mftTracks.rawIteratorAt(candidate.mftTrackId);
        if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
          continue;
        }
        auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

        auto mftTrackProp = fwdtrackutils::getTrackParCovFwd(mftTrack, mftTrackCov);
        auto mchTrackProp = fwdtrackutils::getTrackParCovFwd(mchTrack, mchTrack);

        if (matchingPlaneZ < 0.) {
          mftTrackProp = propagateToZMftPar(mftTrackProp, matchingPlaneZ);
          mchTrackProp = propagateToZMchPar(mchTrackProp, matchingPlaneZ);
        }

        std::vector<float> output;
        std::vector<float> inputML = mlResponse.getInputFeatures(mchTrack, mftTrack, mchTrack, mftTrackProp, mchTrackProp, collision);
        mlResponse.isSelectedMl(inputML, 0, output);
        float matchScore = output[0];

        newCandidatesVector.emplace_back(MatchingCandidate{
          candidate.mftTrackId,
          matchScore,
          -1});
      }
    //}

    auto compareMatchingScore = [](const MatchingCandidate& track1, const MatchingCandidate& track2) -> bool {
      return (track1.matchScore > track2.matchScore);
    };

    //for (auto& [mchIndex, globalTracksVector] : newMatchingCandidates) { // o2-linter: disable=const-ref-in-for-loop (object is modified in loop)
      std::sort(newCandidatesVector.begin(), newCandidatesVector.end(), compareMatchingScore);

      int ranking = 1;
      for (auto& candidate : newCandidatesVector) { // o2-linter: disable=const-ref-in-for-loop (object is modified in loop)
        candidate.matchRanking = ranking;
        ranking += 1;
      }
    //}
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void runMlMatching(C const& collisions,
                     TMUON const& muonTracks,
                     TMFT const& mftTracks,
                     CMFT const& mftCovs,
                     o2::analysis::MlResponseMFTMuonMatch<float>& mlResponse,
                     float matchingPlaneZ,
                     const MatchingCandidates& matchingCandidates,
                     MatchingCandidates& newMatchingCandidates)
  {
    newMatchingCandidates.clear();
    for (const auto& [mchIndex, candidatesVector] : matchingCandidates) {
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
      if (!mchTrack.has_collision()) {
        continue;
      }

      auto collision = collisions.rawIteratorAt(mchTrack.collisionId());

      auto mchTrackParIt = mMchTrackPars.find(mchIndex);
      if (mchTrackParIt == mMchTrackPars.end()) {
        continue;
      }

      for (const auto& candidate : candidatesVector) {
        auto const& mftTrack = mftTracks.rawIteratorAt(candidate.mftTrackId);
        if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
          continue;
        }
        auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

        auto mftTrackProp = fwdtrackutils::getTrackParCovFwd(mftTrack, mftTrackCov);
        auto mchTrackProp = mchTrackParIt->second;

        if (matchingPlaneZ < 0.) {
          mftTrackProp = propagateToZMftPar(mftTrackProp, matchingPlaneZ);
          mchTrackProp = propagateToZMchPar(mchTrackProp, matchingPlaneZ);
        }

        std::vector<float> output;
        std::vector<float> inputML = mlResponse.getInputFeatures(mchTrack, mftTrack, mchTrack, mftTrackProp, mchTrackProp, collision);
        mlResponse.isSelectedMl(inputML, 0, output);
        float matchScore = output[0];

        newMatchingCandidates[mchIndex].emplace_back(MatchingCandidate{
          candidate.mftTrackId,
          matchScore,
          -1});
      }
    }

    auto compareMatchingScore = [](const MatchingCandidate& track1, const MatchingCandidate& track2) -> bool {
      return (track1.matchScore > track2.matchScore);
    };

    for (auto& [mchIndex, globalTracksVector] : newMatchingCandidates) { // o2-linter: disable=const-ref-in-for-loop (object is modified in loop)
      std::sort(globalTracksVector.begin(), globalTracksVector.end(), compareMatchingScore);

      int ranking = 1;
      for (auto& candidate : globalTracksVector) { // o2-linter: disable=const-ref-in-for-loop (object is modified in loop)
        candidate.matchRanking = ranking;
        ranking += 1;
      }
    }
  }

  template <class C, class BC, class TMUON, class TMFT, class CMFT>
  void processCollision(const CollisionInfo& collisionInfo,
                        C const& collisions,
                        BC const& bcs,
                        TMUON const& muonTracks,
                        TMFT const& mftTracks,
                        CMFT const& mftCovs,
                        aod::FwdTrkCls const& clusters)
  {
    auto collision = collisions.rawIteratorAt(collisionInfo.index);

    std::vector<int64_t> taggedMuons;
    getTaggedMuons(collisionInfo, collisions, muonTracks, taggedMuons);

    if (cfgEnableMCHRealign.value) {
      runMuonRealignment(muonTracks, clusters);
    }

    if (cfgCustomMatchingStrategy.value == 0) {
      if (hasActiveChi2Matching) {
        MatchingCandidates matchingCandidates;
        runChi2Matching(collisions, muonTracks, mftTracks, mftCovs, activeChi2FunctionName, activeChi2MatchingPlaneZ, activeChi2ExtrapMethod, collisionInfo.matchingCandidates, matchingCandidates);
        fillMatchingCandidatesForCollision(collision, muonTracks, mftTracks, mftCovs, matchingCandidates, taggedMuons);
      }
    } else {
      if (hasActiveMlMatching) {
        MatchingCandidates matchingCandidates;
        runMlMatching(collisions, muonTracks, mftTracks, mftCovs, activeMlResponse, activeMlMatchingPlaneZ, collisionInfo.matchingCandidates, matchingCandidates);
        fillMatchingCandidatesForCollision(collision, muonTracks, mftTracks, mftCovs, matchingCandidates, taggedMuons);
      }
    }
  }

  template <class TCOLLISION, class TMUON, class TMFT, class CMFT>
  void fillMatchingCandidatesForCollision(TCOLLISION const& collision,
                                          TMUON const& muonTracks,
                                          TMFT const& mftTracks,
                                          CMFT const& mftCovs,
                                          const MatchingCandidates& matchingCandidates,
                                          const std::vector<int64_t>& taggedMuons)
  {
    for (const auto& [mchIndex, candidates] : matchingCandidates) {
      if (candidates.empty()) {
        continue;
      }

      const auto& mchTrack = muonTracks.rawIteratorAt(mchIndex);
      if (!isGoodGlobalMuon(mchTrack, collision)) {
        continue;
      }

      bool isTagged = std::find(taggedMuons.begin(), taggedMuons.end(), mchIndex) != taggedMuons.end();

      std::vector<MatchingCandidate> storedCandidates;
      int nStored = 0;
      for (const auto& candidate : candidates) {
        if (cfgMaxCandidatesPerMchTrack.value >= 0 && nStored >= cfgMaxCandidatesPerMchTrack.value) {
          break;
        }

        int32_t candidateIndex = mMatchCandidateCounter;
        globalMuonMatchCandidates(
          mchIndex,
          candidate.mftTrackId,
          static_cast<float>(candidate.matchChi2),
          static_cast<float>(candidate.matchScore),
          static_cast<int32_t>(candidate.matchRanking),
          isTagged);
        mMatchCandidateCounter += 1;

        mMchTrackToCandidateIndices[mchIndex].push_back(candidateIndex);
        storedCandidates.push_back(candidate);
        nStored += 1;
      }

      if (!storedCandidates.empty()) {
        mMchTrackMatchingCandidates[mchIndex] = std::move(storedCandidates);
        mMchTrackIsTagged[mchIndex] = isTagged;
      }
    }
  }

  int32_t countStoredCandidatesForMchTrack(int64_t mchTrackIndex) const
  {
    const auto candidateIterator = mMchTrackMatchingCandidates.find(mchTrackIndex);
    if (candidateIterator == mMchTrackMatchingCandidates.end()) {
      return 0;
    }
    return static_cast<int32_t>(candidateIterator->second.size());
  }

  template <class TCOLLISION, class TMUON, class TMFT, class CMFT>
  void fillGmmCandidateFwdTracks(TCOLLISION const& collisions,
                                 TMUON const& muonTracks,
                                 TMFT const& mftTracks,
                                 CMFT const& mftCovs)
  {
    if (!cfgProduceCandidateFwdTracks.value) {
      return;
    }

    mFwdTrackToGmmCandTrkIndex.clear();

    // First pass: assign GMMCANDTRK row indices for MCH/MCH-MID base entries so that
    // MCHTrackId can be remapped consistently even when global muons appear first in FwdTracks.
    int32_t nextGmmCandTrkIndex = 0;
    for (const auto& track : muonTracks) {
      const int trackType = static_cast<int>(track.trackType());
      if (trackType > GlobalTrackTypeMax) {
        mFwdTrackToGmmCandTrkIndex[track.globalIndex()] = nextGmmCandTrkIndex;
        nextGmmCandTrkIndex += 1 + countStoredCandidatesForMchTrack(track.globalIndex());
      } else if (cfgIncludeGlobalMuonsInFwdTracks.value && trackType <= GlobalTrackTypeMax) {
        nextGmmCandTrkIndex += 1;
      }
    }

    // Second pass: fill GMMCANDTRK/GMMCANDTRKCOV in FwdTracks order.
    for (const auto& track : muonTracks) {
      const int trackType = static_cast<int>(track.trackType());

      if (trackType > GlobalTrackTypeMax) {
        const int64_t mchTrackIndex = track.globalIndex();
        const int32_t gmmMchTrackId = mFwdTrackToGmmCandTrkIndex.at(mchTrackIndex);

        auto mchTrackParIt = mMchTrackPars.find(mchTrackIndex);
        if (mchTrackParIt == mMchTrackPars.end()) {
          // fill muon tracks table with original parameters
          fillBaseGmmCandFwdTrack(track, gmmMchTrackId, -1, false, -1.f, -1.f);
        } else {
          // fill muon tracks table with realignment parameters
          fillBaseGmmCandFwdTrack(track, mchTrackParIt->second, gmmMchTrackId, -1, false, -1.f, -1.f);
        }

        const auto candidateIterator = mMchTrackMatchingCandidates.find(mchTrackIndex);
        if (candidateIterator != mMchTrackMatchingCandidates.end() && track.has_collision()) {
          const auto& collision = collisions.rawIteratorAt(track.collisionId());
          const bool isTagged = mMchTrackIsTagged[mchTrackIndex];
          for (const auto& candidate : candidateIterator->second) {
            const auto& mftTrack = mftTracks.rawIteratorAt(candidate.mftTrackId);
            fillCandidateFwdTrack(collision, track, gmmMchTrackId, mftTrack, mftCovs, candidate, isTagged);
          }
        }
        continue;
      }

      if (cfgIncludeGlobalMuonsInFwdTracks.value && trackType <= GlobalTrackTypeMax) {
        int32_t gmmMchTrackId = -1;
        const auto mchIterator = mFwdTrackToGmmCandTrkIndex.find(track.matchMCHTrackId());
        if (mchIterator != mFwdTrackToGmmCandTrkIndex.end()) {
          gmmMchTrackId = mchIterator->second;
        }
        fillBaseGmmCandFwdTrack(track,
                                gmmMchTrackId,
                                -1,
                                false,
                                track.chi2MatchMCHMFT(),
                                track.matchScoreMCHMFT());
      }
    }
  }

  template <class TMUON>
  void fillFwdTrkMatchCands(TMUON const& muonTracks)
  {
    std::vector<int32_t> empty{};
    for (const auto& muonTrack : muonTracks) {
      if (static_cast<int>(muonTrack.trackType()) <= GlobalTrackTypeMax) {
        fwdTrkMatchCands(empty);
        continue;
      }

      const int64_t mchTrackIndex = muonTrack.globalIndex();
      const auto matchIterator = mMchTrackToCandidateIndices.find(mchTrackIndex);
      if (matchIterator == mMchTrackToCandidateIndices.end() || matchIterator->second.empty()) {
        fwdTrkMatchCands(empty);
      } else {
        fwdTrkMatchCands(matchIterator->second);
      }
    }
  }


  void processData(MyEvents const& collisions,
                 aod::BCsWithTimestamps const& bcs,
                 MyMuons const& muonTracks,
                 MyMFTs const& mftTracks,
                 MyMFTCovariances const& mftCovs,
                 aod::FwdTrkCls const& clusters)
  {
    auto bc = bcs.begin();
    initCcdb(bc);
    return;

    LOGF(info, "Filling MFT cov");
    mftTrackCovs.clear();
    for (const auto& mftTrackCov : mftCovs) {
      mftTrackCovs[mftTrackCov.matchMFTTrackId()] = mftTrackCov.globalIndex();
    }

    mMatchCandidateCounter = 0;
    mMchTrackToCandidateIndices.clear();
    mMchTrackMatchingCandidates.clear();
    mMchTrackIsTagged.clear();
    mFwdTrackToGmmCandTrkIndex.clear();

    LOGF(info, "Filling coll");
    fillCollisions(collisions, bcs, muonTracks, mftTracks, mftCovs, fCollisionInfos);

    LOGF(info, "processing collisions");
    for (auto const& [collisionIndex, collisionInfo] : fCollisionInfos) {
      processCollision(collisionInfo, collisions, bcs, muonTracks, mftTracks, mftCovs, clusters);
    }

    LOGF(info, "Filling tables");
    // fill table with track/candidates index mapping
    fillFwdTrkMatchCands(muonTracks);
    // fill track tables
    fillGmmCandidateFwdTracks(collisions, muonTracks, mftTracks, mftCovs);
  }

  PROCESS_SWITCH(GlobalMuonMatching, processData, "processData", true);
};

// Extends the fwdtracksrealign table with expression columns
struct GlobalMuonMatchingSpawner {
  Spawns<aod::FwdTrksCovReAlign> realignFwdTrksCov;
  Spawns<aod::FwdTracksReAlign> realignFwdTrks;
  void init(InitContext const&) {}
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<GlobalMuonMatching>(cfgc)};
};
