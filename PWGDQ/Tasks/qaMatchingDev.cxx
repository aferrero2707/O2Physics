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
/// \file qaMatching.cxx
/// \brief Task to compute and evaluate DCA quantities
/// \author Nicolas Bizé <nicolas.bize@cern.ch>, SUBATECH
//
#include "PWGDQ/Core/MuonMatchingMlResponse.h"
#include "PWGDQ/Core/VarManager.h"
#include "PWGDQ/DataModel/ReducedInfoTables.h"

#include "Common/DataModel/EventSelection.h"

#include "Common/DataModel/CollisionAssociationTables.h"
#include "Common/DataModel/TrackSelectionTables.h"

#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "GlobalTracking/MatchGlobalFwd.h"
#include "MFTTracking/Constants.h"

#include <Math/ProbFunc.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace o2;
using namespace o2::framework;
using namespace o2::aod;

using MyEvents = soa::Join<aod::Collisions, aod::EvSels>;
//using MyEvents = soa::Join<aod::Collisions, aod::EvSels, aod::FT0Mults, aod::MFTMults, aod::PVMults, aod::CentFT0Ms, aod::CentFT0As, aod::CentFT0Cs>;
using MyMuons = soa::Join<aod::FwdTracks, aod::FwdTracksCov>;
using MyMuonsMC = soa::Join<aod::FwdTracks, aod::FwdTracksCov, aod::McFwdTrackLabels>;
//using MyMuonsMC = soa::Join<aod::FwdTracks, aod::FwdTracksCov, aod::McFwdTrackLabels, aod::FwdTracksDCA, aod::FwdTrkCompColls>;
using MyMFTs = aod::MFTTracks;
using MyMFTCovariances = aod::MFTTracksCov;
using MyMFTsMC = soa::Join<aod::MFTTracks, aod::McMFTTrackLabels>;

using MyMuon = MyMuons::iterator;
using MyMuonMC = MyMuonsMC::iterator;
using MyMFT = MyMFTs::iterator;
using MyMFTCovariance = MyMFTCovariances::iterator;

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

constexpr double firstMCHPlaneZ = -526.16;

static float chi2ToScore_(float chi2, int ndf, float chi2max)
{
  double p = ROOT::Math::chisquared_cdf_c(chi2, ndf);
  double pmin = ROOT::Math::chisquared_cdf_c(chi2max, ndf);
  double exponent = TMath::Log10(0.5f) / TMath::Log10(pmin);
  double result = TMath::Power(p, exponent);
  //float p = 1.f / (1.f - TMath::Log10(ROOT::Math::chisquared_cdf_c(chi2, ndf)));
  //std::cout << std::format("chi2={:0.3f}  NDF={}  P={:0.8f}  result={:0.3f}", chi2, ndf, p, result) << std::endl;
  return static_cast<float>(result);
}

static float chi2ToScore(float chi2, int ndf, float chi2max)
{
  double p = -TMath::Log10(ROOT::Math::chisquared_cdf_c(chi2, ndf));
  double pnorm = -TMath::Log10(ROOT::Math::chisquared_cdf_c(chi2max, ndf));
  double result = (1.f / (p / pnorm + 1.f));
  //float p = 1.f / (1.f - TMath::Log10(ROOT::Math::chisquared_cdf_c(chi2, ndf)));
  //std::cout << std::format("chi2={:0.3f}  NDF={}  P={:0.8f}  result={:0.3f}", chi2, ndf, p, result) << std::endl;
  return static_cast<float>(result);
}

/*static float chi2ToScore(float chi2, float chi2norm = 10.f)
{
  return (1.f / (chi2 / chi2norm + 1.f));
}

static float scoreToChi2(float score, float chi2norm = 10.f)
{
  return ((1.f / score - 1.f) * chi2norm);
}*/

//_________________________________________________________________________________________________
static o2::dataformats::GlobalFwdTrack MCHtoFwd(const o2::mch::TrackParam& mchParam)
{
  using SMatrix55Std = ROOT::Math::SMatrix<double, 5>;
  using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;

  // Convert a MCH Track parameters and covariances matrix to the
  // Forward track format. Must be called after propagation though the absorber

  o2::dataformats::GlobalFwdTrack convertedTrack;

  // Parameter conversion
  double alpha1, alpha3, alpha4, x2, x3, x4;

  alpha1 = mchParam.getNonBendingSlope();
  alpha3 = mchParam.getBendingSlope();
  alpha4 = mchParam.getInverseBendingMomentum();

  x2 = TMath::ATan2(-alpha3, -alpha1);
  x3 = -1. / TMath::Sqrt(alpha3 * alpha3 + alpha1 * alpha1);
  x4 = alpha4 * -x3 * TMath::Sqrt(1 + alpha3 * alpha3);

  auto K = alpha1 * alpha1 + alpha3 * alpha3;
  auto K32 = K * TMath::Sqrt(K);
  auto L = TMath::Sqrt(alpha3 * alpha3 + 1);

  // Covariances matrix conversion
  SMatrix55Std jacobian;
  SMatrix55Sym covariances;

  covariances(0, 0) = mchParam.getCovariances()(0, 0);
  covariances(0, 1) = mchParam.getCovariances()(0, 1);
  covariances(0, 2) = mchParam.getCovariances()(0, 2);
  covariances(0, 3) = mchParam.getCovariances()(0, 3);
  covariances(0, 4) = mchParam.getCovariances()(0, 4);

  covariances(1, 1) = mchParam.getCovariances()(1, 1);
  covariances(1, 2) = mchParam.getCovariances()(1, 2);
  covariances(1, 3) = mchParam.getCovariances()(1, 3);
  covariances(1, 4) = mchParam.getCovariances()(1, 4);

  covariances(2, 2) = mchParam.getCovariances()(2, 2);
  covariances(2, 3) = mchParam.getCovariances()(2, 3);
  covariances(2, 4) = mchParam.getCovariances()(2, 4);

  covariances(3, 3) = mchParam.getCovariances()(3, 3);
  covariances(3, 4) = mchParam.getCovariances()(3, 4);

  covariances(4, 4) = mchParam.getCovariances()(4, 4);

  jacobian(0, 0) = 1;

  jacobian(1, 2) = 1;

  jacobian(2, 1) = -alpha3 / K;
  jacobian(2, 3) = alpha1 / K;

  jacobian(3, 1) = alpha1 / K32;
  jacobian(3, 3) = alpha3 / K32;

  jacobian(4, 1) = -alpha1 * alpha4 * L / K32;
  jacobian(4, 3) = alpha3 * alpha4 * (1 / (TMath::Sqrt(K) * L) - L / K32);
  jacobian(4, 4) = L / TMath::Sqrt(K);

  // jacobian*covariances*jacobian^T
  covariances = ROOT::Math::Similarity(jacobian, covariances);

  // Set output
  convertedTrack.setX(mchParam.getNonBendingCoor());
  convertedTrack.setY(mchParam.getBendingCoor());
  convertedTrack.setZ(mchParam.getZ());
  convertedTrack.setPhi(x2);
  convertedTrack.setTanl(x3);
  convertedTrack.setInvQPt(x4);
  convertedTrack.setCharge(mchParam.getCharge());
  convertedTrack.setCovariances(covariances);

  return convertedTrack;
}

//_________________________________________________________________________________________________
template <class T>
static o2::mch::TrackParam FwdtoMCH(const T& fwdtrack)
{
  using SMatrix55Std = ROOT::Math::SMatrix<double, 5>;
  using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;

  // Convert Forward Track parameters and covariances matrix to the
  // MCH track format.

  // Parameter conversion
  double alpha1, alpha3, alpha4, x2, x3, x4;

  x2 = fwdtrack.getPhi();
  x3 = fwdtrack.getTanl();
  x4 = fwdtrack.getInvQPt();

  auto sinx2 = TMath::Sin(x2);
  auto cosx2 = TMath::Cos(x2);

  alpha1 = cosx2 / x3;
  alpha3 = sinx2 / x3;
  alpha4 = x4 / TMath::Sqrt(x3 * x3 + sinx2 * sinx2);

  auto K = TMath::Sqrt(x3 * x3 + sinx2 * sinx2);
  auto K3 = K * K * K;

  // Covariances matrix conversion
  SMatrix55Std jacobian;
  SMatrix55Sym covariances;

  covariances(0, 0) = fwdtrack.getCovariances()(0, 0);
  covariances(0, 1) = fwdtrack.getCovariances()(0, 1);
  covariances(0, 2) = fwdtrack.getCovariances()(0, 2);
  covariances(0, 3) = fwdtrack.getCovariances()(0, 3);
  covariances(0, 4) = fwdtrack.getCovariances()(0, 4);

  covariances(1, 1) = fwdtrack.getCovariances()(1, 1);
  covariances(1, 2) = fwdtrack.getCovariances()(1, 2);
  covariances(1, 3) = fwdtrack.getCovariances()(1, 3);
  covariances(1, 4) = fwdtrack.getCovariances()(1, 4);

  covariances(2, 2) = fwdtrack.getCovariances()(2, 2);
  covariances(2, 3) = fwdtrack.getCovariances()(2, 3);
  covariances(2, 4) = fwdtrack.getCovariances()(2, 4);

  covariances(3, 3) = fwdtrack.getCovariances()(3, 3);
  covariances(3, 4) = fwdtrack.getCovariances()(3, 4);

  covariances(4, 4) = fwdtrack.getCovariances()(4, 4);

  jacobian(0, 0) = 1;

  jacobian(1, 2) = -sinx2 / x3;
  jacobian(1, 3) = -cosx2 / (x3 * x3);

  jacobian(2, 1) = 1;

  jacobian(3, 2) = cosx2 / x3;
  jacobian(3, 3) = -sinx2 / (x3 * x3);

  jacobian(4, 2) = -x4 * sinx2 * cosx2 / K3;
  jacobian(4, 3) = -x3 * x4 / K3;
  jacobian(4, 4) = 1 / K;
  // jacobian*covariances*jacobian^T
  covariances = ROOT::Math::Similarity(jacobian, covariances);

  double cov[] = {covariances(0, 0), covariances(1, 0), covariances(1, 1), covariances(2, 0), covariances(2, 1), covariances(2, 2), covariances(3, 0), covariances(3, 1), covariances(3, 2), covariances(3, 3), covariances(4, 0), covariances(4, 1), covariances(4, 2), covariances(4, 3), covariances(4, 4)};
  double param[] = {fwdtrack.getX(), alpha1, fwdtrack.getY(), alpha3, alpha4};

  o2::mch::TrackParam convertedTrack(fwdtrack.getZ(), param, cov);
  return o2::mch::TrackParam(convertedTrack);
}


struct qaMatching {

  template <class T, int nr, int nc>
  using matrix = std::array<std::array<T, nc>, nr>;

  enum MuonMatchType {
    kMatchTypeTrueLeading = 0,
    kMatchTypeWrongLeading = 1,
    kMatchTypeDecayLeading = 2,
    kMatchTypeFakeLeading = 3,
    kMatchTypeTrueNonLeading = 4,
    kMatchTypeWrongNonLeading = 5,
    kMatchTypeDecayNonLeading = 6,
    kMatchTypeFakeNonLeading = 7,
    kMatchTypeUndefined
  };

  struct MatchingCandidate {
    int64_t collisionId{-1};
    int64_t globalTrackId{-1};
    int64_t muonTrackId{-1};
    int64_t mftTrackId{-1};
    double matchScore{-1};
    double matchChi2{-1};
    int matchRanking{-1};
    double matchScoreProd{-1};
    double matchChi2Prod{-1};
    int matchRankingProd{-1};
    MuonMatchType matchType{kMatchTypeUndefined};
  };

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
  Configurable<float> fMatchingChi2ScoreMftMchLow{"cfgMatchingChi2ScoreMftMchLow", chi2ToScore(50.f, 5, 50.f), ""};

  ////   Variables for selecting tagged muons
  Configurable<int> fMuonTaggingNCrossedMftPlanesLow{"cfgMuonTaggingNCrossedMftPlanesLow", 5, ""};
  Configurable<float> fMuonTaggingTrackChi2MchUp{"cfgMuonTaggingTrackChi2MchUp", 5.f, ""};
  Configurable<float> fMuonTaggingPMchLow{"cfgMuonTaggingPMchLow", 0.0f, ""};
  Configurable<float> fMuonTaggingPtMchLow{"cfgMuonTaggingPtMchLow", 0.7f, ""};
  Configurable<float> fMuonTaggingEtaMchLow{"cfgMuonTaggingEtaMchLow", -3.6f, ""};
  Configurable<float> fMuonTaggingEtaMchUp{"cfgMuonTaggingEtaMchUp", -2.5f, ""};
  Configurable<float> fMuonTaggingRabsLow{"cfgMuonTaggingRabsLow", 17.6f, ""};
  Configurable<float> fMuonTaggingRabsUp{"cfgMuonTaggingRabsUp", 89.5f, ""};
  Configurable<float> fMuonTaggingSigmaPdcaUp{"cfgMuonTaggingPdcaUp", 4.f, ""};
  Configurable<float> fMuonTaggingChi2DiffLow{"cfgMuonTaggingChi2DiffLow", 100.f, ""};

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

  // CCDB connection configurables
  struct : ConfigurableGroup {
    Configurable<std::string> fConfigCcdbUrl{"ccdb-url-", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
    Configurable<int64_t> fConfigNoLaterThan{"ccdb-no-later-than-", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), "latest acceptable timestamp of creation for the object"};
    Configurable<std::string> fConfigGrpPath{"grpPath-", "GLO/GRP/GRP", "Path of the grp file"};
    Configurable<std::string> fConfigGeoPath{"geoPath-", "GLO/Config/GeometryAligned", "Path of the geometry file"};
    Configurable<std::string> fConfigGrpMagPath{"grpmagPath-", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
  } fConfigCCDB;

  ///    Variables for histograms configuration
  Configurable<int> fNCandidatesMax{"nCandidatesMax", 5, ""};

  double mBzAtMftCenter{0};
  static constexpr float sAbsFront{-90.f};
  static constexpr float sAbsBack{-505.f};


  o2::globaltracking::MatchGlobalFwd mExtrap;

  using MatchingFunc_t = std::function<std::tuple<double, int>(const o2::dataformats::GlobalFwdTrack& mchtrack, const o2::track::TrackParCovFwd& mfttrack)>;
  std::map<std::string, MatchingFunc_t> mMatchingFunctionMap; ///< MFT-MCH Matching function

  // Chi2 matching interface
  static constexpr int sChi2FunctionsNum = 7;
  struct : ConfigurableGroup {
    std::array<Configurable<std::string>, sChi2FunctionsNum> fFunctionLabel{{
      {"cfgChi2FunctionLabel_0", std::string{"MatchALLMethod0a"}, "Text label identifying this chi2 matching method"},
      {"cfgChi2FunctionLabel_1", std::string{""/*"MatchALLMethod0b"*/}, "Text label identifying this chi2 matching method"},
      {"cfgChi2FunctionLabel_2", std::string{""/*"MatchXYPhiTanlMethod0a"*/}, "Text label identifying this chi2 matching method"},
      {"cfgChi2FunctionLabel_3", std::string{""/*"MatchXYPhiTanlMethod0b"*/}, "Text label identifying this chi2 matching method"},
      {"cfgChi2FunctionLabel_4", std::string{""/*"MatchXYPhiTanlMethod1a"*/}, "Text label identifying this chi2 matching method"},
      {"cfgChi2FunctionLabel_5", std::string{""/*"MatchXYPhiTanlMethod1b"*/}, "Text label identifying this chi2 matching method"},
      {"cfgChi2FunctionLabel_6", std::string{"MatchPhiTanlMethod2"/*"MatchXYSxSyMethod0a"*/}, "Text label identifying this chi2 matching method"},
    }};
    std::array<Configurable<std::string>, sChi2FunctionsNum> fFunctionName{{{"cfgChi2FunctionNames_0", std::string{"matchALL"}, "Name of the chi2 matching function"},
                                                                            {"cfgChi2FunctionNames_1", std::string{"matchALL"}, "Name of the chi2 matching function"},
                                                                            {"cfgChi2FunctionNames_2", std::string{"matchXYPhiTanl"/*"matchXYPhiTanl"*/}, "Name of the chi2 matching function"},
                                                                            {"cfgChi2FunctionNames_3", std::string{"matchXYPhiTanl"/*"matchXYPhiTanl"*/}, "Name of the chi2 matching function"},
                                                                            {"cfgChi2FunctionNames_4", std::string{"matchXYPhiTanl"/*"matchXYPhiTanl"*/}, "Name of the chi2 matching function"},
                                                                            {"cfgChi2FunctionNames_5", std::string{"matchXYPhiTanl"/*"matchXYPhiTanl"*/}, "Name of the chi2 matching function"},
                                                                            {"cfgChi2FunctionNames_6", std::string{"matchPhiTanl"/*"matchXYSxSy"*/}, "Name of the chi2 matching function"}}};
    std::array<Configurable<float>, sChi2FunctionsNum> fMatchingScoreCut{{
      {"cfgChi2FunctionMatchingScoreCut_0", 0.5f, "Minimum score value for selecting good matches"},
      {"cfgChi2FunctionMatchingScoreCut_1", 0.5f, "Minimum score value for selecting good matches"},
      {"cfgChi2FunctionMatchingScoreCut_2", 0.5f, "Minimum score value for selecting good matches"},
      {"cfgChi2FunctionMatchingScoreCut_3", 0.5f, "Minimum score value for selecting good matches"},
      {"cfgChi2FunctionMatchingScoreCut_4", 0.5f, "Minimum score value for selecting good matches"},
      {"cfgChi2FunctionMatchingScoreCut_5", 0.5f, "Minimum score value for selecting good matches"},
      {"cfgChi2FunctionMatchingScoreCut_6", 0.5f, "Minimum score value for selecting good matches"},
    }};
    std::array<Configurable<float>, sChi2FunctionsNum> fMatchingPlaneZ{{
      {"cfgChi2FunctionMatchingPlaneZ_0", static_cast<float>(o2::mft::constants::mft::LayerZCoordinate()[9]), "Z position of the matching plane"},
      {"cfgChi2FunctionMatchingPlaneZ_1", static_cast<float>(-89.f), "Z position of the matching plane"},
      {"cfgChi2FunctionMatchingPlaneZ_2", static_cast<float>(o2::mft::constants::mft::LayerZCoordinate()[9]), "Z position of the matching plane"},
      {"cfgChi2FunctionMatchingPlaneZ_3", static_cast<float>(-89.f), "Z position of the matching plane"},
      {"cfgChi2FunctionMatchingPlaneZ_4", static_cast<float>(o2::mft::constants::mft::LayerZCoordinate()[9]), "Z position of the matching plane"},
      {"cfgChi2FunctionMatchingPlaneZ_5", static_cast<float>(-446.f), "Z position of the matching plane"},
      {"cfgChi2FunctionMatchingPlaneZ_6", static_cast<float>(o2::mft::constants::mft::LayerZCoordinate()[0]), "Z position of the matching plane"},
    }};
    std::array<Configurable<int>, sChi2FunctionsNum> fMatchingExtrapMethod{{
      {"cfgMatchingExtrapMethod_0", static_cast<int>(0), "Method for MCH track extrapolation to maching plane"},
      {"cfgMatchingExtrapMethod_1", static_cast<int>(0), "Method for MCH track extrapolation to maching plane"},
      {"cfgMatchingExtrapMethod_2", static_cast<int>(0), "Method for MCH track extrapolation to maching plane"},
      {"cfgMatchingExtrapMethod_3", static_cast<int>(0), "Method for MCH track extrapolation to maching plane"},
      {"cfgMatchingExtrapMethod_4", static_cast<int>(1), "Method for MCH track extrapolation to maching plane"},
      {"cfgMatchingExtrapMethod_5", static_cast<int>(1), "Method for MCH track extrapolation to maching plane"},
      {"cfgMatchingExtrapMethod_6", static_cast<int>(2), "Method for MCH track extrapolation to maching plane"},
    }};
  } fConfigChi2MatchingOptions;

  // ML interface
  static constexpr int sMLModelsNum = 2;
  struct : ConfigurableGroup {
    std::array<Configurable<std::string>, sMLModelsNum> fModelLabel{{
      {"cfgMLModelLabel_0", std::string{""}, "Text label identifying this group of ML models"},
      {"cfgMLModelLabel_1", std::string{""}, "Text label identifying this group of ML models"},
    }};
    std::array<Configurable<std::vector<std::string>>, sMLModelsNum> fModelPathsCCDB{{{"cfgMLModelPathsCCDB_0", std::vector<std::string>{"Users/m/mcoquet/MLTest"}, "Paths of models on CCDB"},
                                                                                      {"cfgMLModelPathsCCDB_1", std::vector<std::string>{}, "Paths of models on CCDB"}}};
    std::array<Configurable<std::vector<std::string>>, sMLModelsNum> fInputFeatures{{{"cfgMLInputFeatures_0", std::vector<std::string>{"chi2MCHMFT"}, "Names of ML model input features"},
                                                                                     {"cfgMLInputFeatures_1", std::vector<std::string>{}, "Names of ML model input features"}}};
    std::array<Configurable<std::vector<std::string>>, sMLModelsNum> fModelNames{{{"cfgMLModelNames_0", std::vector<std::string>{"model.onnx"}, "ONNX file names for each pT bin (if not from CCDB full path)"},
                                                                                  {"cfgMLModelNames_1", std::vector<std::string>{}, "ONNX file names for each pT bin (if not from CCDB full path)"}}};
    std::array<Configurable<float>, sMLModelsNum> fMatchingScoreCut{{
      {"cfgMLModelMatchingScoreCut_0", 0.f, "Minimum score value for selecting good matches"},
      {"cfgMLModelMatchingScoreCut_1", 0.f, "Minimum score value for selecting good matches"},
    }};
    std::array<Configurable<float>, sMLModelsNum> fMatchingPlaneZ{{
      {"cfgMLModelMatchingPlaneZ_0", static_cast<float>(o2::mft::constants::mft::LayerZCoordinate()[9]), "Z position of the matching plane"},
      {"cfgMLModelMatchingPlaneZ_1", 0.f, "Z position of the matching plane"},
    }};
    std::array<Configurable<int>, sMLModelsNum> fMatchingExtrapMethod{{
      {"cfgMatchingExtrapMethod_0", static_cast<int>(0), "Method for MCH track extrapolation to maching plane"},
      {"cfgMatchingExtrapMethod_1", static_cast<int>(0), "Method for MCH track extrapolation to maching plane"},
    }};
  } fConfigMlOptions;

  std::vector<double> binsPtMl;
  std::array<double, 1> cutValues;
  std::vector<int> cutDirMl;
  std::map<std::string, o2::analysis::MlResponseMFTMuonMatch<float>> matchingMlResponses;
  std::map<std::string, std::string> matchingChi2Functions;
  std::map<std::string, double> matchingPlanesZ;
  std::map<std::string, double> matchingScoreCuts;
  std::map<std::string, int> matchingExtrapMethod;

  int mRunNumber{0}; // needed to detect if the run changed and trigger update of magnetic field

  Service<o2::ccdb::BasicCCDBManager> ccdbManager;
  o2::ccdb::CcdbApi fCCDBApi;

  o2::aod::rctsel::RCTFlagsChecker rctChecker{"CBT_muon_glo", false, false, true};

  // vector of all MFT-MCH(-MID) matching candidates associated to the same MCH(-MID) track,
  // to be sorted in descending order with respect to the matching score
  // the map key is the MCH(-MID) track global index
  // the elements are pairs of global muon track indexes and associated matching scores
  // for matching candidates computed with the chi2 method, the score is defined as 1/(1+chi2)
  //using MatchingCandidates = std::map<int64_t, std::vector<std::pair<int64_t, double>>>;
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
    // vector of MFT-MCH track index pairs belonging to the same MC muon particle
    std::vector<std::pair<int64_t, int64_t>> matchablePairs;
    // vector of MCH track indexes that are expected to have an associated MFT track
    std::vector<int64_t> taggedMuons;
  };

  using CollisionInfos = std::map<int64_t, CollisionInfo>;

  std::unordered_map<int64_t, int32_t> mftTrackCovs;

  std::vector<std::pair<int64_t, int64_t>> fMatchablePairs;
  MatchingCandidates fMatchingCandidates;
  std::vector<int64_t> fTaggedMuons;

  using MuonPair = std::pair<std::pair<int64_t, uint64_t>, std::pair<int64_t, uint64_t>>;
  using GlobalMuonPair = std::pair<std::pair<int64_t, std::vector<MatchingCandidate>>, std::pair<int64_t, std::vector<MatchingCandidate>>>;

  HistogramRegistry registry{"registry", {}};
  HistogramRegistry registryMatching{"registryMatching", {}};
  HistogramRegistry registryMatchingTrackChi2{"registryMatchingTrackChi2", {}};
  HistogramRegistry registryMatching0{"registryMatching_0", {}};
  HistogramRegistry registryMatching1{"registryMatching_1", {}};
  HistogramRegistry registryMatching2{"registryMatching_2", {}};
  HistogramRegistry registryMatching3{"registryMatching_3", {}};
  HistogramRegistry registryMatching4{"registryMatching_4", {}};
  HistogramRegistry registryMatching5{"registryMatching_5", {}};
  HistogramRegistry registryMatching6{"registryMatching_6", {}};
  HistogramRegistry registryMatching7{"registryMatching_7", {}};
  HistogramRegistry registryMatching8{"registryMatching_8", {}};
  HistogramRegistry registryMatching9{"registryMatching_9", {}};
  std::vector<HistogramRegistry*> registryMatchingVec{{
    &registryMatching0,
    &registryMatching1,
    &registryMatching2,
    &registryMatching3,
    &registryMatching4,
    &registryMatching5,
    &registryMatching6,
    &registryMatching7,
    &registryMatching8,
    &registryMatching9
  }};
  HistogramRegistry registryAlignment{"registryAlignment", {}};
  HistogramRegistry registryAlignment0{"registryAlignment_0", {}};
  HistogramRegistry registryAlignment1{"registryAlignment_1", {}};
  HistogramRegistry registryAlignment2{"registryAlignment_2", {}};
  HistogramRegistry registryAlignment3{"registryAlignment_3", {}};
  HistogramRegistry registryAlignment4{"registryAlignment_4", {}};
  HistogramRegistry registryAlignment5{"registryAlignment_5", {}};
  HistogramRegistry registryAlignment6{"registryAlignment_6", {}};
  HistogramRegistry registryAlignment7{"registryAlignment_7", {}};
  HistogramRegistry registryAlignment8{"registryAlignment_8", {}};
  HistogramRegistry registryAlignment9{"registryAlignment_9", {}};
  std::vector<HistogramRegistry*> registryAlignmentVec{{
    &registryAlignment0,
    &registryAlignment1,
    &registryAlignment2,
    &registryAlignment3,
    &registryAlignment4,
    &registryAlignment5,
    &registryAlignment6,
    &registryAlignment7,
    &registryAlignment8,
    &registryAlignment9
  }};
  HistogramRegistry registryDimuon{"registryDimuon", {}};

  std::unordered_map<std::string, o2::framework::HistPtr> matchingHistos;
  std::array<std::unordered_map<std::string, o2::framework::HistPtr>, 4> dcaHistos;
  matrix<o2::framework::HistPtr, 4, 4> dimuonHistos;

  struct EfficiencyPlotter {
    o2::framework::HistPtr p_num;
    o2::framework::HistPtr p_den;
    o2::framework::HistPtr pt_num;
    o2::framework::HistPtr pt_den;
    o2::framework::HistPtr phi_num;
    o2::framework::HistPtr phi_den;
    o2::framework::HistPtr eta_num;
    o2::framework::HistPtr eta_den;

    EfficiencyPlotter(std::string path, std::string title,
                      HistogramRegistry& registry)
    {
      AxisSpec pAxis = {100, 0, 100, "p (GeV/c)"};
      AxisSpec pTAxis = {100, 0, 10, "p_{T} (GeV/c)"};
      AxisSpec etaAxis = {100, -4, -2, "#eta"};
      AxisSpec phiAxis = {90, -180, 180, "#phi (degrees)"};

      std::string histName;
      std::string histTitle;

      // momentum dependence
      histName = path + "p_num";
      histTitle = title + " vs. p - num";
      p_num = registry.add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {pAxis}});

      histName = path + "p_den";
      histTitle = title + " vs. p - den";
      p_den = registry.add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {pAxis}});

      // pT dependence
      histName = path + "pt_num";
      histTitle = title + " vs. p_{T} - num";
      pt_num = registry.add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {pTAxis}});

      histName = path + "pt_den";
      histTitle = title + " vs. p_{T} - den";
      pt_den = registry.add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {pTAxis}});

      // eta dependence
      histName = path + "eta_num";
      histTitle = title + " vs. #eta - num";
      eta_num = registry.add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {etaAxis}});

      histName = path + "eta_den";
      histTitle = title + " vs. #eta - den";
      eta_den = registry.add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {etaAxis}});

      // phi dependence
      histName = path + "phi_num";
      histTitle = title + " vs. #phi - num";
      phi_num = registry.add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {phiAxis}});

      histName = path + "phi_den";
      histTitle = title + " vs. #phi - den";
      phi_den = registry.add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {phiAxis}});
    }

    template <class T>
    void Fill(const T& track, bool passed)
    {
      double phi = track.phi() * 180 / TMath::Pi();
      std::get<std::shared_ptr<TH1>>(p_den)->Fill(track.p());
      std::get<std::shared_ptr<TH1>>(pt_den)->Fill(track.pt());
      std::get<std::shared_ptr<TH1>>(eta_den)->Fill(track.eta());
      std::get<std::shared_ptr<TH1>>(phi_den)->Fill(phi);

      if (passed) {
        std::get<std::shared_ptr<TH1>>(p_num)->Fill(track.p());
        std::get<std::shared_ptr<TH1>>(pt_num)->Fill(track.pt());
        std::get<std::shared_ptr<TH1>>(eta_num)->Fill(track.eta());
        std::get<std::shared_ptr<TH1>>(phi_num)->Fill(phi);
      }
    }
  };

  struct MatchRankingHistos {
    o2::framework::HistPtr hist;
    o2::framework::HistPtr histVsP;
    o2::framework::HistPtr histVsPt;
    o2::framework::HistPtr histVsMcParticleDz;
    o2::framework::HistPtr histVsMftTrackMult;
    o2::framework::HistPtr histVsMftTrackType;
    o2::framework::HistPtr histVsDeltaChi2;
    o2::framework::HistPtr histVsProdRanking;
    //o2::framework::HistPtr histVsMftDQ;

    MatchRankingHistos(std::string histName, std::string histTitle, HistogramRegistry* registry)
    {
      AxisSpec pAxis = {100, 0, 100, "p (GeV/c)"};
      AxisSpec ptAxis = {100, 0, 10, "p_{T} (GeV/c)"};
      AxisSpec dzAxis = {100, 0, 50, "#Deltaz (cm)"};
      AxisSpec trackMultAxis = {100, 0, 1000, "MFT track mult."};
      AxisSpec trackTypeAxis = {2, 0, 2, "MFT track type"};
      AxisSpec dchi2Axis = {100, 0, 100, "#Delta#chi^{2}"};
      AxisSpec dqAxis = {3, -1.5, 1.5, "MFT #DeltaQ"};
      AxisSpec indexAxis = {6, 0, 6, "ranking index"};
      AxisSpec indexProdAxis = {6, 0, 6, "ranking index (production)"};

      hist = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {indexAxis}});
      histVsP = registry->add((histName + "VsP").c_str(), (histTitle + " vs. p").c_str(), {HistType::kTH2F, {pAxis, indexAxis}});
      histVsPt = registry->add((histName + "VsPt").c_str(), (histTitle + " vs. p_{T}").c_str(), {HistType::kTH2F, {ptAxis, indexAxis}});
      histVsMcParticleDz = registry->add((histName + "VsMcParticleDz").c_str(), (histTitle + " vs. MC particle #Deltaz").c_str(), {HistType::kTH2F, {dzAxis, indexAxis}});
      histVsMftTrackMult = registry->add((histName + "VsMftTrackMult").c_str(), (histTitle + " vs. MFT track multiplicity").c_str(), {HistType::kTH2F, {trackMultAxis, indexAxis}});
      histVsMftTrackType = registry->add((histName + "VsMftTrackType").c_str(), (histTitle + " vs. MFT track type").c_str(), {HistType::kTH2F, {trackTypeAxis, indexAxis}});
      std::get<std::shared_ptr<TH2>>(histVsMftTrackType)->GetXaxis()->SetBinLabel(1, "Kalman");
      std::get<std::shared_ptr<TH2>>(histVsMftTrackType)->GetXaxis()->SetBinLabel(2, "CA");
      histVsDeltaChi2 = registry->add((histName + "VsDeltaChi2").c_str(), (histTitle + " vs. #Delta#chi^{2}").c_str(), {HistType::kTH2F, {dchi2Axis, indexAxis}});
      histVsProdRanking = registry->add((histName + "VsProdRanking").c_str(), (histTitle + " vs. prod ranking").c_str(), {HistType::kTH2F, {indexProdAxis, indexAxis}});
      //histVsMftDQ = registry->add((histName + "VsMftDQ").c_str(), (histTitle + " vs. MFT #DeltaQ").c_str(), {HistType::kTH2F, {dqAxis, indexAxis}});
    }
  };

  struct MatchingPlotter {
    std::unique_ptr<MatchRankingHistos> fMatchRanking;
    std::unique_ptr<MatchRankingHistos> fMatchRankingGoodMCH;
    std::unique_ptr<MatchRankingHistos> fMatchRankingPaired;
    std::unique_ptr<MatchRankingHistos> fMatchRankingPairedGoodMCH;
    std::unique_ptr<MatchRankingHistos> fMatchRankingMuon;
    std::unique_ptr<MatchRankingHistos> fMatchRankingMuonGoodMCH;
    std::unique_ptr<MatchRankingHistos> fMatchRankingDecay;
    std::unique_ptr<MatchRankingHistos> fMatchRankingDecayGoodMCH;

    //-
    o2::framework::HistPtr fMissedMatches;
    o2::framework::HistPtr fMissedMatchesGoodMCH;
    o2::framework::HistPtr fMissedMatchesGoodMCHMFT;
    //-
    o2::framework::HistPtr fMatchRankingWrtProd;
    o2::framework::HistPtr fMatchRankingWrtProdVsP;
    o2::framework::HistPtr fMatchRankingWrtProdVsPt;

    //-
    o2::framework::HistPtr fDecayRankingGoodMatches;
    o2::framework::HistPtr fDecayRankingNonLeadingMatches;
    o2::framework::HistPtr fDecayRankingMissedMatches;

    //-
    o2::framework::HistPtr fScoreGapLeadingTrueMatches;
    o2::framework::HistPtr fScoreGapNonLeadingTrueMatches;

    //-
    o2::framework::HistPtr fMatchType;
    o2::framework::HistPtr fMatchTypeVsP;
    o2::framework::HistPtr fMatchTypeVsPt;

    //-
    o2::framework::HistPtr fMatchScoreVsChi2;
    //-
    o2::framework::HistPtr fMatchScoreVsType;
    o2::framework::HistPtr fMatchScoreVsTypeVsP;
    o2::framework::HistPtr fMatchScoreVsTypeVsPt;
    //-
    o2::framework::HistPtr fMatchChi2VsType;
    o2::framework::HistPtr fMatchChi2VsTypeVsP;
    o2::framework::HistPtr fMatchChi2VsTypeVsPt;
    /*//-
    o2::framework::HistPtr fLeadingMatchScore;
    o2::framework::HistPtr fLeadingMatchScoreVsP;
    o2::framework::HistPtr fLeadingMatchScoreVsPt;
    o2::framework::HistPtr fTrueMatchScore;
    o2::framework::HistPtr fTrueMatchScoreVsP;
    o2::framework::HistPtr fTrueMatchScoreVsPt;
    o2::framework::HistPtr fTrueMatchScoreMuon;
    o2::framework::HistPtr fTrueMatchScoreMuonVsP;
    o2::framework::HistPtr fTrueMatchScoreMuonVsPt;
    o2::framework::HistPtr fTrueMatchScoreDecay;
    o2::framework::HistPtr fTrueMatchScoreDecayVsP;
    o2::framework::HistPtr fTrueMatchScoreDecayVsPt;
    o2::framework::HistPtr fFakeMatchScore;
    o2::framework::HistPtr fFakeMatchScoreVsP;
    o2::framework::HistPtr fFakeMatchScoreVsPt;
    */
    //-
    o2::framework::HistPtr fMatchScoreVsProd;
    o2::framework::HistPtr fMatchChi2VsProd;
    o2::framework::HistPtr fTrueMatchScoreVsProd;
    o2::framework::HistPtr fTrueMatchChi2VsProd;

    //-
    EfficiencyPlotter fMatchingPurityPlotter;
    EfficiencyPlotter fPairingEfficiencyPlotter;
    EfficiencyPlotter fMatchingEfficiencyPlotter;
    EfficiencyPlotter fMatchingEfficiencyMuonPlotter;
    EfficiencyPlotter fFakeMatchingEfficiencyPlotter;

    HistogramRegistry* registry;

    MatchingPlotter(std::string path,
                    HistogramRegistry* reg)
      : fMatchingPurityPlotter(path + "matching-purity/", "Matching purity", *reg),
        fPairingEfficiencyPlotter(path + "pairing-efficiency/", "Pairing efficiency", *reg),
        fMatchingEfficiencyPlotter(path + "matching-efficiency/", "Matching efficiency", *reg),
        fMatchingEfficiencyMuonPlotter(path + "matching-efficiency-muon/", "Matching efficiency for muons", *reg),
        fFakeMatchingEfficiencyPlotter(path + "fake-matching-efficiency/", "Fake matching efficiency", *reg)
    {
      registry = reg;
      AxisSpec pAxis = {100, 0, 100, "p (GeV/c)"};
      AxisSpec ptAxis = {100, 0, 10, "p_{T} (GeV/c)"};
      AxisSpec dzAxis = {100, 0, 50, "#Deltaz (cm)"};
      AxisSpec indexAxis = {6, 0, 6, "ranking index"};

      std::string histName = path + "matchRanking";
      std::string histTitle = "True match ranking";

      fMatchRanking = std::make_unique<MatchRankingHistos>(path + "matchRanking", "True match ranking", registry);
      fMatchRankingGoodMCH = std::make_unique<MatchRankingHistos>(path + "matchRankingGoodMCH", "True match ranking (good MCH tracks)", registry);
      fMatchRankingPaired = std::make_unique<MatchRankingHistos>(path + "matchRankingPaired", "True match ranking (paired MCH tracks)", registry);
      fMatchRankingPairedGoodMCH = std::make_unique<MatchRankingHistos>(path + "matchRankingPairedGoodMCH", "True match ranking (good paired MCH tracks)", registry);
      fMatchRankingMuon = std::make_unique<MatchRankingHistos>(path + "matchRankingMuon", "True match ranking (paired MCH tracks, muons)", registry);
      fMatchRankingMuonGoodMCH = std::make_unique<MatchRankingHistos>(path + "matchRankingMuonGoodMCH", "True match ranking (good paired MCH tracks, muons)", registry);
      fMatchRankingDecay = std::make_unique<MatchRankingHistos>(path + "matchRankingDecay", "True match ranking (paired MCH tracks, intermediate decay)", registry);
      fMatchRankingDecayGoodMCH = std::make_unique<MatchRankingHistos>(path + "matchRankingDecayGoodMCH", "True match ranking (good paired MCH tracks, intermediate decay)", registry);

      //-
      AxisSpec missedMatchAxis = {5, 0, 5, ""};
      histName = path + "missedMatches";
      histTitle = "Missed matches";
      fMissedMatches = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {missedMatchAxis}});
      std::get<std::shared_ptr<TH1>>(fMissedMatches)->GetXaxis()->SetBinLabel(1, "not paired");
      std::get<std::shared_ptr<TH1>>(fMissedMatches)->GetXaxis()->SetBinLabel(2, "fake MCH");
      std::get<std::shared_ptr<TH1>>(fMissedMatches)->GetXaxis()->SetBinLabel(3, "not stored");
      histName = path + "missedMatchesGoodMCH";
      histTitle = "Missed matches - good MCH tracks";
      fMissedMatchesGoodMCH = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {missedMatchAxis}});
      std::get<std::shared_ptr<TH1>>(fMissedMatchesGoodMCH)->GetXaxis()->SetBinLabel(1, "not paired");
      std::get<std::shared_ptr<TH1>>(fMissedMatchesGoodMCH)->GetXaxis()->SetBinLabel(2, "fake MCH");
      std::get<std::shared_ptr<TH1>>(fMissedMatchesGoodMCH)->GetXaxis()->SetBinLabel(3, "not stored");
      histName = path + "missedMatchesGoodMCHMFT";
      histTitle = "Missed matches - good MFT and MCH tracks";
      fMissedMatchesGoodMCHMFT = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {missedMatchAxis}});
      std::get<std::shared_ptr<TH1>>(fMissedMatchesGoodMCHMFT)->GetXaxis()->SetBinLabel(1, "not paired");
      std::get<std::shared_ptr<TH1>>(fMissedMatchesGoodMCHMFT)->GetXaxis()->SetBinLabel(2, "fake MCH");
      std::get<std::shared_ptr<TH1>>(fMissedMatchesGoodMCHMFT)->GetXaxis()->SetBinLabel(3, "not stored");

      AxisSpec decayRankingAxis = {5, 0, 5, "decay ranking"};
      histName = path + "decayRankingGoodMatches";
      histTitle = "Decay ranking - good matches";
      fDecayRankingGoodMatches = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {decayRankingAxis}});
      histName = path + "decayRankingNonLeadingMatches";
      histTitle = "Decay ranking - non-leading matches";
      fDecayRankingNonLeadingMatches = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {decayRankingAxis}});
      histName = path + "decayRankingMissedMatches";
      histTitle = "Decay ranking - missed matches";
      fDecayRankingMissedMatches = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {decayRankingAxis}});

      AxisSpec scoreGapAxis = {100, 0, 1, "match score difference"};
      histName = path + "scoreGapLeadingTrueMatches";
      histTitle = "Score gap between leading and subleading matches - good matches";
      fScoreGapLeadingTrueMatches = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {scoreGapAxis}});
      histName = path + "scoreGapNonLeadingTrueMatches";
      histTitle = "Score gap between leading and subleading matches - non-leading matches";
      fScoreGapNonLeadingTrueMatches = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {scoreGapAxis}});

      //-
      AxisSpec chi2Axis = {100, 0, 100, "matching #chi^{2}/NDF"};
      AxisSpec scoreAxis = {100, 0, 1, "matching score"};
      int matchTypeMax = static_cast<int>(kMatchTypeUndefined);
      AxisSpec matchTypeAxis = {matchTypeMax, 0, matchTypeMax, "match type"};
      histName = path + "matchType";
      histTitle = "Match type";
      fMatchType = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {matchTypeAxis}});
      histName = path + "matchTypeVsP";
      histTitle = "Match type vs. p";
      fMatchTypeVsP = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {pAxis, matchTypeAxis}});
      histName = path + "matchTypeVsPt";
      histTitle = "Match type vs. p_{T}";
      fMatchTypeVsPt = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {ptAxis, matchTypeAxis}});

      histName = path + "matchScoreVsChi2";
      histTitle = "Match score vs. #chi^{2}";
      fMatchScoreVsChi2 = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {chi2Axis, scoreAxis}});
      //-
      histName = path + "matchChi2VsType";
      histTitle = "Match #chi^{2} vs. match type";
      fMatchChi2VsType = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {matchTypeAxis, chi2Axis}});
      histName = path + "matchChi2VsTypeVsP";
      histTitle = "Match #chi^{2} vs. match type vs. p";
      fMatchChi2VsTypeVsP = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH3F, {pAxis, matchTypeAxis, chi2Axis}});
      histName = path + "matchChi2VsTypeVsPt";
      histTitle = "Match #chi^{2} vs. match type vs. p_{T}";
      fMatchChi2VsTypeVsPt = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH3F, {ptAxis, matchTypeAxis, chi2Axis}});
      //-
      histName = path + "matchScoreVsType";
      histTitle = "Match score vs. match type";
      fMatchScoreVsType = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {matchTypeAxis, scoreAxis}});
      histName = path + "matchScoreVsTypeVsP";
      histTitle = "Match score vs. match type vs. p";
      fMatchScoreVsTypeVsP = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH3F, {pAxis, matchTypeAxis, scoreAxis}});
      histName = path + "matchScoreVsTypeVsPt";
      histTitle = "Match score vs. match type vs. p_{T}";
      fMatchScoreVsTypeVsPt = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH3F, {ptAxis, matchTypeAxis, scoreAxis}});
      /*//-
      histName = path + "leadingMatchScore";
      histTitle = "Leading match score";
      fLeadingMatchScore = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {scoreAxis}});
      histName = path + "leadingMatchScoreVsP";
      histTitle = "Leading match score vs. p";
      fLeadingMatchScoreVsP = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {pAxis, scoreAxis}});
      histName = path + "leadingMatchScoreVsPt";
      histTitle = "Leading match score vs. p_{T}";
      fLeadingMatchScoreVsPt = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {ptAxis, scoreAxis}});
      //-
      histName = path + "trueMatchScore";
      histTitle = "True match score";
      fTrueMatchScore = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {scoreAxis}});
      histName = path + "trueMatchScoreVsP";
      histTitle = "True match score vs. p";
      fTrueMatchScoreVsP = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {pAxis, scoreAxis}});
      histName = path + "trueMatchScoreVsPt";
      histTitle = "True match score vs. p_{T}";
      fTrueMatchScoreVsPt = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {ptAxis, scoreAxis}});
      //-
      histName = path + "trueMatchScoreMuon";
      histTitle = "True match score - muon";
      fTrueMatchScoreMuon = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {scoreAxis}});
      histName = path + "trueMatchScoreMuonVsP";
      histTitle = "True match score vs. p - muon";
      fTrueMatchScoreMuonVsP = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {pAxis, scoreAxis}});
      histName = path + "trueMatchScoreMuonVsPt";
      histTitle = "True match score vs. p_{T} - muon";
      fTrueMatchScoreMuonVsPt = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {ptAxis, scoreAxis}});
      //-
      histName = path + "trueMatchScoreDecay";
      histTitle = "True match score - decay";
      fTrueMatchScoreDecay = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {scoreAxis}});
      histName = path + "trueMatchScoreDecayVsP";
      histTitle = "True match score vs. p - decay";
      fTrueMatchScoreDecayVsP = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {pAxis, scoreAxis}});
      histName = path + "trueMatchScoreDecayVsPt";
      histTitle = "True match score vs. p_{T} - decay";
      fTrueMatchScoreDecayVsPt = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {ptAxis, scoreAxis}});

      histName = path + "fakeMatchScore";
      histTitle = "Fake match score";
      fFakeMatchScore = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH1F, {scoreAxis}});
      histName = path + "fakeMatchScoreVsP";
      histTitle = "Fake match score vs. p";
      fFakeMatchScoreVsP = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {pAxis, scoreAxis}});
      histName = path + "fakeMatchScoreVsPt";
      histTitle = "Fake match score vs. p_{T}";
      fFakeMatchScoreVsPt = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {ptAxis, scoreAxis}});
      */
      AxisSpec prodScoreAxis = {100, 0, 1, "matching score (prod)"};
      histName = path + "matchScoreVsProd";
      histTitle = "Match score vs. production";
      fMatchScoreVsProd = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {prodScoreAxis, scoreAxis}});
      histName = path + "trueMatchScoreVsProd";
      histTitle = "Match score vs. production - true match";
      fTrueMatchScoreVsProd = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {prodScoreAxis, scoreAxis}});

      AxisSpec prodChi2Axis = {100, 0, 100, "matching #chi^{2}/NDF (prod)"};
      histName = path + "matchChi2VsProd";
      histTitle = "Match #chi^{2} vs. production";
      fMatchChi2VsProd = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {prodChi2Axis, chi2Axis}});
      histName = path + "trueMatchChi2VsProd";
      histTitle = "Match #chi^{2} vs. production - true match";
      fTrueMatchChi2VsProd = registry->add(histName.c_str(), histTitle.c_str(), {HistType::kTH2F, {{100, 0, 10, "matching #chi^{2} (prod)"}, {100, 0, 10, "matching #chi^{2}"}}});
    }
  };

  struct TrackResolutionHistos {
    // tracks residuals versus momentum
    o2::framework::HistPtr mftMchTrackDx_directParticle;
    o2::framework::HistPtr mftMchTrackDy_directParticle;
    o2::framework::HistPtr mftMchTrackDphi_directParticle;
    o2::framework::HistPtr mftMchTrackDtanl_directParticle;
    o2::framework::HistPtr mftMchTrackDsx_directParticle;
    o2::framework::HistPtr mftMchTrackDsy_directParticle;

    // normalized tracks residuals versus momentum
    o2::framework::HistPtr mftMchTrackNDx_directParticle;
    o2::framework::HistPtr mftMchTrackNDy_directParticle;
    o2::framework::HistPtr mftMchTrackNDphi_directParticle;
    o2::framework::HistPtr mftMchTrackNDtanl_directParticle;
    o2::framework::HistPtr mftMchTrackNDsx_directParticle;
    o2::framework::HistPtr mftMchTrackNDsy_directParticle;

    // tracks residuals versus momentum
    o2::framework::HistPtr mftMchTrackDx_goodRanking;
    o2::framework::HistPtr mftMchTrackDy_goodRanking;

    o2::framework::HistPtr partMftTrackDx_goodRanking;
    o2::framework::HistPtr partMftTrackDx_goodRanking_Qpn[4][2];
    o2::framework::HistPtr partMftTrackDy_goodRanking;
    o2::framework::HistPtr partMftTrackDy_goodRanking_Qpn[4][2];
    o2::framework::HistPtr partMchTrackDx_goodRanking;
    o2::framework::HistPtr partMchTrackDy_goodRanking;
    o2::framework::HistPtr mftTrackDr_goodRanking;
    o2::framework::HistPtr mftTrackDtheta_goodRanking;
    o2::framework::HistPtr mftTrackSigmax_goodRanking;
    o2::framework::HistPtr mftTrackSigmay_goodRanking;
    o2::framework::HistPtr mftTrackSigmaPhi_goodRanking;
    o2::framework::HistPtr mftTrackSigmaTanl_goodRanking;

    o2::framework::HistPtr mftMchTrackDx_badRanking;
    o2::framework::HistPtr mftMchTrackDy_badRanking;
    o2::framework::HistPtr partMftTrackDx_badRanking;
    o2::framework::HistPtr partMftTrackDy_badRanking;
    o2::framework::HistPtr partMchTrackDx_badRanking;
    o2::framework::HistPtr partMchTrackDy_badRanking;
    o2::framework::HistPtr mftTrackDr_badRanking;
    o2::framework::HistPtr mftTrackDtheta_badRanking;
    o2::framework::HistPtr mftTrackSigmax_badRanking;
    o2::framework::HistPtr mftTrackSigmay_badRanking;
    o2::framework::HistPtr mftTrackSigmaPhi_badRanking;
    o2::framework::HistPtr mftTrackSigmaTanl_badRanking;

    o2::framework::HistPtr mftMchTrackDx_kalman;
    o2::framework::HistPtr mftMchTrackDy_kalman;
    o2::framework::HistPtr partMftTrackDx_kalman;
    o2::framework::HistPtr partMftTrackDy_kalman;

    o2::framework::HistPtr mftMchTrackDx_ca;
    o2::framework::HistPtr mftMchTrackDy_ca;
    o2::framework::HistPtr partMftTrackDx_ca;
    o2::framework::HistPtr partMftTrackDy_ca;

    o2::framework::HistPtr mftDpVsP;
    o2::framework::HistPtr mftDpOverPVsP;
    o2::framework::HistPtr mftDpOverSigmaPVsP;
    o2::framework::HistPtr mftDQVsP;
    o2::framework::HistPtr mchDpVsP;
    o2::framework::HistPtr mchDpOverPVsP;

    TrackResolutionHistos(std::string histPath, HistogramRegistry* registry)
    {
      AxisSpec dxAxis = {200, -10, 10, "#Deltax (cm)"};
      AxisSpec dyAxis = {200, -10, 10, "#Deltay (cm)"};
      AxisSpec dsxAxis = {200, -10, 10, "#Deltaslope_x"};
      AxisSpec dsyAxis = {200, -10, 10, "#Deltaslope_y"};
      AxisSpec dphiAxis = {200, -10, 10, "#Delta#phi (degrees)"};
      AxisSpec dtanlAxis = {200, -10, 10, "#Deltatanl"};

      AxisSpec ndxAxis = {200, -10, 10, "#Deltax/#sigmax"};
      AxisSpec ndyAxis = {200, -10, 10, "#Deltay/#sigmay"};
      AxisSpec ndsxAxis = {200, -10, 10, "#Deltaslope_x/#sigmaslope_x"};
      AxisSpec ndsyAxis = {200, -10, 10, "#Deltaslope_y/#sigmaslope_y"};
      AxisSpec ndphiAxis = {200, -10, 10, "#Delta#phi/#sigma#phi"};
      AxisSpec ndtanlAxis = {200, -10, 10, "#Deltatanl/#sigmatanl"};

      AxisSpec drAxis = {100, 0, 10, "#DeltaR (cm)"};
      AxisSpec dthetaAxis = {100, 0, 10, "#Delta#theta (degrees)"};
      AxisSpec sigmaxAxis = {1000, 0, 1.0, "#sigma_{x} (cm)"};
      AxisSpec sigmayAxis = {1000, 0, 1.0, "#sigma_{y} (cm)"};
      AxisSpec sigmaPhiAxis = {100, 0, 1, "#sigma_{#phi} (degrees)"};
      AxisSpec sigmaTanlAxis = {100, 0, 0.1, "#sigma_{tanl}"};
      AxisSpec dcaxMCHAxis = {400, -10.0, 10.0, "DCA_{x} (cm)"};
      AxisSpec dcayMCHAxis = {400, -10.0, 10.0, "DCA_{y} (cm)"};
      AxisSpec dpAxis = {200, -10, 10, "p (GeV/c)"};
      AxisSpec dppAxis = {200, -0.1, 0.1, "#Deltap/p"};
      AxisSpec dQAxis = {3, -1.5f, 1.5f, "#DeltaQ"};
      AxisSpec pAxis = {100, 0, 100, "p (GeV/c)"};

      mftMchTrackDx_directParticle = registry->add((histPath + "mftMchTrackDx_directParticle").c_str(), "MFT-MCH #Deltax vs. p (direct particle)", {HistType::kTH2F, {pAxis, dxAxis}});
      mftMchTrackDy_directParticle = registry->add((histPath + "mftMchTrackDy_directParticle").c_str(), "MFT-MCH #Deltay vs. p (direct particle)", {HistType::kTH2F, {pAxis, dyAxis}});
      mftMchTrackDsx_directParticle = registry->add((histPath + "mftMchTrackDsx_directParticle").c_str(), "MFT-MCH #Deltaslope_x vs. p (direct particle)", {HistType::kTH2F, {pAxis, dsxAxis}});
      mftMchTrackDsy_directParticle = registry->add((histPath + "mftMchTrackDsy_directParticle").c_str(), "MFT-MCH #Deltaslope_y vs. p (direct particle)", {HistType::kTH2F, {pAxis, dsyAxis}});
      mftMchTrackDphi_directParticle = registry->add((histPath + "mftMchTrackDphi_directParticle").c_str(), "MFT-MCH #Delta#phi vs. p (direct particle)", {HistType::kTH2F, {pAxis, dphiAxis}});
      mftMchTrackDtanl_directParticle = registry->add((histPath + "mftMchTrackDtanl_directParticle").c_str(), "MFT-MCH #Deltatanl vs. p (direct particle)", {HistType::kTH2F, {pAxis, dtanlAxis}});

      mftMchTrackNDx_directParticle = registry->add((histPath + "mftMchTrackNDx_directParticle").c_str(), "MFT-MCH #Deltax/#sigmax vs. p (direct particle)", {HistType::kTH2F, {pAxis, ndxAxis}});
      mftMchTrackNDy_directParticle = registry->add((histPath + "mftMchTrackNDy_directParticle").c_str(), "MFT-MCH #Deltay/#sigmay vs. p (direct particle)", {HistType::kTH2F, {pAxis, ndyAxis}});
      mftMchTrackNDsx_directParticle = registry->add((histPath + "mftMchTrackNDsx_directParticle").c_str(), "MFT-MCH #Deltaslope_x/#signmaslope_x vs. p (direct particle)", {HistType::kTH2F, {pAxis, ndsxAxis}});
      mftMchTrackNDsy_directParticle = registry->add((histPath + "mftMchTrackNDsy_directParticle").c_str(), "MFT-MCH #Deltaslope_y/#signmaslope_y vs. p (direct particle)", {HistType::kTH2F, {pAxis, ndsyAxis}});
      mftMchTrackNDphi_directParticle = registry->add((histPath + "mftMchTrackNDphi_directParticle").c_str(), "MFT-MCH #Delta#phi/#signa#phi vs. p (direct particle)", {HistType::kTH2F, {pAxis, ndphiAxis}});
      mftMchTrackNDtanl_directParticle = registry->add((histPath + "mftMchTrackNDtanl_directParticle").c_str(), "MFT-MCH #Deltatanl/#sigmatanl vs. p (direct particle)", {HistType::kTH2F, {pAxis, ndtanlAxis}});

      mftMchTrackDx_goodRanking = registry->add((histPath + "mftMchTrackDx_goodRanking").c_str(), "MFT-MCH #Deltax vs. p (good ranking)", {HistType::kTH2F, {pAxis, dxAxis}});
      mftMchTrackDy_goodRanking = registry->add((histPath + "mftMchTrackDy_goodRanking").c_str(), "MFT-MCH #Deltay vs. p (good ranking)", {HistType::kTH2F, {pAxis, dyAxis}});
      partMftTrackDx_goodRanking = registry->add((histPath + "partMftTrackDx_goodRanking").c_str(), "Particle-MFT #Deltax vs. p (good ranking)", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_goodRanking = registry->add((histPath + "partMftTrackDy_goodRanking").c_str(), "Particle-MFT #Deltay vs. p (good ranking)", {HistType::kTH2F, {pAxis, dyAxis}});
      partMftTrackDx_goodRanking_Qpn[0][0] = registry->add((histPath + "partMftTrackDx_goodRanking_Q0p").c_str(), "Particle-MFT #Deltax vs. p (good ranking) - Q0 p", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_goodRanking_Qpn[0][0] = registry->add((histPath + "partMftTrackDy_goodRanking_Q0p").c_str(), "Particle-MFT #Deltay vs. p (good ranking) - Q0 p", {HistType::kTH2F, {pAxis, dyAxis}});
      partMftTrackDx_goodRanking_Qpn[0][1] = registry->add((histPath + "partMftTrackDx_goodRanking_Q0n").c_str(), "Particle-MFT #Deltax vs. p (good ranking) - Q0 n", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_goodRanking_Qpn[0][1] = registry->add((histPath + "partMftTrackDy_goodRanking_Q0n").c_str(), "Particle-MFT #Deltay vs. p (good ranking) - Q0 n", {HistType::kTH2F, {pAxis, dyAxis}});
      partMftTrackDx_goodRanking_Qpn[1][0] = registry->add((histPath + "partMftTrackDx_goodRanking_Q1p").c_str(), "Particle-MFT #Deltax vs. p (good ranking) - Q1 p", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_goodRanking_Qpn[1][0] = registry->add((histPath + "partMftTrackDy_goodRanking_Q1p").c_str(), "Particle-MFT #Deltay vs. p (good ranking) - Q1 p", {HistType::kTH2F, {pAxis, dyAxis}});
      partMftTrackDx_goodRanking_Qpn[1][1] = registry->add((histPath + "partMftTrackDx_goodRanking_Q1n").c_str(), "Particle-MFT #Deltax vs. p (good ranking) - Q1 n", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_goodRanking_Qpn[1][1] = registry->add((histPath + "partMftTrackDy_goodRanking_Q1n").c_str(), "Particle-MFT #Deltay vs. p (good ranking) - Q1 n", {HistType::kTH2F, {pAxis, dyAxis}});
      partMftTrackDx_goodRanking_Qpn[2][0] = registry->add((histPath + "partMftTrackDx_goodRanking_Q2p").c_str(), "Particle-MFT #Deltax vs. p (good ranking) - Q2 p", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_goodRanking_Qpn[2][0] = registry->add((histPath + "partMftTrackDy_goodRanking_Q2p").c_str(), "Particle-MFT #Deltay vs. p (good ranking) - Q2 p", {HistType::kTH2F, {pAxis, dyAxis}});
      partMftTrackDx_goodRanking_Qpn[2][1] = registry->add((histPath + "partMftTrackDx_goodRanking_Q2n").c_str(), "Particle-MFT #Deltax vs. p (good ranking) - Q2 n", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_goodRanking_Qpn[2][1] = registry->add((histPath + "partMftTrackDy_goodRanking_Q2n").c_str(), "Particle-MFT #Deltay vs. p (good ranking) - Q2 n", {HistType::kTH2F, {pAxis, dyAxis}});
      partMftTrackDx_goodRanking_Qpn[3][0] = registry->add((histPath + "partMftTrackDx_goodRanking_Q3p").c_str(), "Particle-MFT #Deltax vs. p (good ranking) - Q3 p", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_goodRanking_Qpn[3][0] = registry->add((histPath + "partMftTrackDy_goodRanking_Q3p").c_str(), "Particle-MFT #Deltay vs. p (good ranking) - Q3 p", {HistType::kTH2F, {pAxis, dyAxis}});
      partMftTrackDx_goodRanking_Qpn[3][1] = registry->add((histPath + "partMftTrackDx_goodRanking_Q3n").c_str(), "Particle-MFT #Deltax vs. p (good ranking) - Q3 n", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_goodRanking_Qpn[3][1] = registry->add((histPath + "partMftTrackDy_goodRanking_Q3n").c_str(), "Particle-MFT #Deltay vs. p (good ranking) - Q3 n", {HistType::kTH2F, {pAxis, dyAxis}});

      partMchTrackDx_goodRanking = registry->add((histPath + "partMchTrackDx_goodRanking").c_str(), "Particle-MCH #Deltax vs. p (good ranking)", {HistType::kTH2F, {pAxis, dxAxis}});
      partMchTrackDy_goodRanking = registry->add((histPath + "partMchTrackDy_goodRanking").c_str(), "Particle-MCH #Deltay vs. p (good ranking)", {HistType::kTH2F, {pAxis, dyAxis}});

      mftTrackDr_goodRanking = registry->add((histPath + "mftTrackDr_goodRanking").c_str(), "MFT #DeltaR vs. p (good ranking)", {HistType::kTH2F, {pAxis, drAxis}});
      mftTrackDtheta_goodRanking = registry->add((histPath + "mftTrackDtheta_goodRanking").c_str(), "MFT #Delta#theta vs. p (good ranking)", {HistType::kTH2F, {pAxis, dthetaAxis}});

      mftTrackSigmax_goodRanking = registry->add((histPath + "mftTrackSigmax_goodRanking").c_str(), "MFT #sigma_{x} vs. p (good ranking)", {HistType::kTH2F, {pAxis, sigmaxAxis}});
      mftTrackSigmay_goodRanking = registry->add((histPath + "mftTrackSigmay_goodRanking").c_str(), "MFT #sigma_{y} vs. p (good ranking)", {HistType::kTH2F, {pAxis, sigmayAxis}});
      mftTrackSigmaPhi_goodRanking = registry->add((histPath + "mftTrackSigmaPhi_goodRanking").c_str(), "MFT #sigma_{#phi} vs. p (good ranking)", {HistType::kTH2F, {pAxis, sigmaPhiAxis}});
      mftTrackSigmaTanl_goodRanking = registry->add((histPath + "mftTrackSigmaTanl_goodRanking").c_str(), "MFT #sigma_{tanl} vs. p (good ranking)", {HistType::kTH2F, {pAxis, sigmaTanlAxis}});

      mftMchTrackDx_badRanking = registry->add((histPath + "mftMchTrackDx_badRanking").c_str(), "MFT-MCH #Deltax vs. p (bad ranking)", {HistType::kTH2F, {pAxis, dxAxis}});
      mftMchTrackDy_badRanking = registry->add((histPath + "mftMchTrackDy_badRanking").c_str(), "MFT-MCH #Deltay vs. p (bad ranking)", {HistType::kTH2F, {pAxis, dyAxis}});

      partMftTrackDx_badRanking = registry->add((histPath + "partMftTrackDx_badRanking").c_str(), "Particle-MFT #Deltax vs. p (bad ranking)", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_badRanking = registry->add((histPath + "partMftTrackDy_badRanking").c_str(), "Particle-MFT #Deltay vs. p (bad ranking)", {HistType::kTH2F, {pAxis, dyAxis}});

      partMchTrackDx_badRanking = registry->add((histPath + "partMchTrackDx_badRanking").c_str(), "Particle-MCH #Deltax vs. p (bad ranking)", {HistType::kTH2F, {pAxis, dxAxis}});
      partMchTrackDy_badRanking = registry->add((histPath + "partMchTrackDy_badRanking").c_str(), "Particle-MCH #Deltay vs. p (bad ranking)", {HistType::kTH2F, {pAxis, dyAxis}});

      mftTrackDr_badRanking = registry->add((histPath + "mftTrackDr_badRanking").c_str(), "MFT #DeltaR vs. p (bad ranking)", {HistType::kTH2F, {pAxis, drAxis}});
      mftTrackDtheta_badRanking = registry->add((histPath + "mftTrackDtheta_badRanking").c_str(), "MFT #Delta#theta vs. p (bad ranking)", {HistType::kTH2F, {pAxis, dthetaAxis}});

      mftTrackSigmax_badRanking = registry->add((histPath + "mftTrackSigmax_badRanking").c_str(), "MFT #sigma_{x} vs. p (bad ranking)", {HistType::kTH2F, {pAxis, sigmaxAxis}});
      mftTrackSigmay_badRanking = registry->add((histPath + "mftTrackSigmay_badRanking").c_str(), "MFT #sigma_{y} vs. p (bad ranking)", {HistType::kTH2F, {pAxis, sigmayAxis}});
      mftTrackSigmaPhi_badRanking = registry->add((histPath + "mftTrackSigmaPhi_badRanking").c_str(), "MFT #sigma_{#phi} vs. p (bad ranking)", {HistType::kTH2F, {pAxis, sigmaPhiAxis}});
      mftTrackSigmaTanl_badRanking = registry->add((histPath + "mftTrackSigmaTanl_badRanking").c_str(), "MFT #sigma_{tanl} vs. p (bad ranking)", {HistType::kTH2F, {pAxis, sigmaTanlAxis}});

      mftMchTrackDx_kalman = registry->add((histPath + "mftMchTrackDx_kalman").c_str(), "MFT-MCH #Deltax vs. p (kalman filter)", {HistType::kTH2F, {pAxis, dxAxis}});
      mftMchTrackDy_kalman = registry->add((histPath + "mftMchTrackDy_kalman").c_str(), "MFT-MCH #Deltay vs. p (kalman filter)", {HistType::kTH2F, {pAxis, dyAxis}});

      partMftTrackDx_kalman = registry->add((histPath + "partMftTrackDx_kalman").c_str(), "Particle-MFT #Deltax vs. p (kalman filter)", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_kalman = registry->add((histPath + "partMftTrackDy_kalman").c_str(), "Particle-MFT #Deltay vs. p (kalman filter)", {HistType::kTH2F, {pAxis, dyAxis}});

      mftMchTrackDx_ca = registry->add((histPath + "mftMchTrackDx_ca").c_str(), "MFT-MCH #Deltax vs. p (cellular automation)", {HistType::kTH2F, {pAxis, dxAxis}});
      mftMchTrackDy_ca = registry->add((histPath + "mftMchTrackDy_ca").c_str(), "MFT-MCH #Deltay vs. p (cellular automation)", {HistType::kTH2F, {pAxis, dyAxis}});

      partMftTrackDx_ca = registry->add((histPath + "partMftTrackDx_ca").c_str(), "Particle-MFT #Deltax vs. p (cellular automation)", {HistType::kTH2F, {pAxis, dxAxis}});
      partMftTrackDy_ca = registry->add((histPath + "partMftTrackDy_ca").c_str(), "Particle-MFT #Deltay vs. p (cellular automation)", {HistType::kTH2F, {pAxis, dyAxis}});

      mftDpVsP = registry->add((histPath + "mftDpVsP").c_str(), "MFT momentum resolution vs. p", {HistType::kTH2F, {pAxis, dpAxis}});
      mftDpOverPVsP = registry->add((histPath + "mftDpOverPVsP").c_str(), "MFT #Deltap/p vs. p", {HistType::kTH2F, {pAxis, {200, -1, 1, "#Deltap/p"}}});
      mftDpOverSigmaPVsP = registry->add((histPath + "mftDpOverSigmaPVsP").c_str(), "MFT #Deltap/#sigma_{p} vs. p", {HistType::kTH2F, {pAxis, {200, -1, 1, "#Deltap/#sigma_{p}"}}});
      mftDQVsP = registry->add((histPath + "mftDQVsP").c_str(), "MFT #DeltaQ vs. p", {HistType::kTH2F, {pAxis, dQAxis}});
      mchDpVsP = registry->add((histPath + "mchDpVsP").c_str(), "MCH momentum resolution vs. p", {HistType::kTH2F, {pAxis, dpAxis}});
      mchDpOverPVsP = registry->add((histPath + "mchDpOverPVsP").c_str(), "MCH #Deltap/p vs. p", {HistType::kTH2F, {pAxis, dppAxis}});
    }
  };

  std::unique_ptr<MatchingPlotter> fChi2MatchingPlotter;
  std::unique_ptr<MatchingPlotter> fTrackChi2MatchingPlotter;
  std::map<std::string, std::unique_ptr<HistogramRegistry>> fMatchingHistogramRegistries;
  std::map<std::string, std::unique_ptr<MatchingPlotter>> fMatchingPlotters;
  std::unique_ptr<MatchingPlotter> fTaggedMuonsMatchingPlotter;
  std::unique_ptr<MatchingPlotter> fSelectedMuonsMatchingPlotter;

  std::map<std::string, std::unique_ptr<TrackResolutionHistos>> fTrackResolutionHistos;
  //std::unique_ptr<TrackResolutionHistos> fTrackResolutionHistosTaggedMuons;
  //std::unique_ptr<TrackResolutionHistos> fTrackResolutionHistosSelectedMuons;

  CollisionInfos fCollisionInfos;

  template <typename BC>
  void initCCDB(BC const& bc)
  {
    if (mRunNumber == bc.runNumber())
      return;

    mRunNumber = bc.runNumber();
    std::map<std::string, std::string> metadata;
    auto soreor = o2::ccdb::BasicCCDBManager::getRunDuration(fCCDBApi, mRunNumber);
    auto ts = soreor.first;
    auto grpmag = fCCDBApi.retrieveFromTFileAny<o2::parameters::GRPMagField>(grpmagPath, metadata, ts);
    o2::base::Propagator::initFieldFromGRP(grpmag);
    LOGF(info, "Set field for muons");
    VarManager::SetupMuonMagField();
    if (!o2::base::GeometryManager::isGeometryLoaded()) {
      ccdbManager->get<TGeoManager>(geoPath);
    }
    o2::mch::TrackExtrap::setField();
    auto* fieldB = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    if (fieldB) {
      double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
      mBzAtMftCenter = fieldB->getBz(centerMFT);
      // std::cout << "fieldB: " << (void*)fieldB << std::endl;
    }
  }

  void createMatchingHistosMC()
  {
    AxisSpec chi2Axis = {1000, 0, 1000, "chi^{2}"};
    AxisSpec chi2AxisSmall = {200, 0, 100, "chi^{2}"};
    AxisSpec pAxis = {1000, 0, 100, "p (GeV/c)"};
    AxisSpec pTAxis = {100, 0, 10, "p_{T} (GeV/c)"};
    AxisSpec etaAxis = {100, -4, -2, "#eta"};
    AxisSpec phiAxis = {90, -180, 180, "#phi (degrees)"};
    std::string histPath = "matching/MC/";

    AxisSpec trackPositionXAtMFTAxis = {100, -15, 15, "MFT x (cm)"};
    AxisSpec trackPositionYAtMFTAxis = {100, -15, 15, "MFT y (cm)"};
    registry.add((histPath + "pairedMCHTracksAtMFT").c_str(), "Paired MCH tracks position at MFT end", {HistType::kTH2F, {trackPositionXAtMFTAxis, trackPositionYAtMFTAxis}});
    registry.add((histPath + "pairedMFTTracksAtMFT").c_str(), "Paired MFT tracks position at MFT end", {HistType::kTH2F, {trackPositionXAtMFTAxis, trackPositionYAtMFTAxis}});
    registry.add((histPath + "selectedMCHTracksAtMFTFront").c_str(), "Selected MCH tracks position at MFT front", {HistType::kTH2F, {trackPositionXAtMFTAxis, trackPositionYAtMFTAxis}});
    registry.add((histPath + "selectedMCHTracksAtMFTFrontTrue").c_str(), "Selected MCH tracks position at MFT front - true", {HistType::kTH2F, {trackPositionXAtMFTAxis, trackPositionYAtMFTAxis}});
    registry.add((histPath + "selectedMCHTracksAtMFTFrontFake").c_str(), "Selected MCH tracks position at MFT front - fake", {HistType::kTH2F, {trackPositionXAtMFTAxis, trackPositionYAtMFTAxis}});
    registry.add((histPath + "selectedMCHTracksAtMFTBack").c_str(), "Selected MCH tracks position at MFT back", {HistType::kTH2F, {trackPositionXAtMFTAxis, trackPositionYAtMFTAxis}});
    registry.add((histPath + "selectedMCHTracksAtMFTBackTrue").c_str(), "Selected MCH tracks position at MFT back - true", {HistType::kTH2F, {trackPositionXAtMFTAxis, trackPositionYAtMFTAxis}});
    registry.add((histPath + "selectedMCHTracksAtMFTBackFake").c_str(), "Selected MCH tracks position at MFT back - fake", {HistType::kTH2F, {trackPositionXAtMFTAxis, trackPositionYAtMFTAxis}});

    registry.add((histPath + "matchingChi2CorrelationMuon").c_str(), "Matching #chi2 correlation, muons (-77.5 cm vs. -466 cm)", {HistType::kTH2F, {chi2AxisSmall, chi2AxisSmall}});
    registry.add((histPath + "matchingChi2CorrelationFake").c_str(), "Matching #chi2 correlation, fake matches (-77.5 cm vs. -466 cm)", {HistType::kTH2F, {chi2AxisSmall, chi2AxisSmall}});

    AxisSpec pairableType = {2, 0, 2, ""};
    auto pairableTypeHist = registry.add((histPath + "pairableType").c_str(), "Pairable MCH tracks type", {HistType::kTH1F, {pairableType}});
    std::get<std::shared_ptr<TH1>>(pairableTypeHist)->GetXaxis()->SetBinLabel(1, "direct");
    std::get<std::shared_ptr<TH1>>(pairableTypeHist)->GetXaxis()->SetBinLabel(2, "decay");

    fChi2MatchingPlotter = std::make_unique<MatchingPlotter>(histPath + "Prod/", &registryMatching);
    fTrackChi2MatchingPlotter = std::make_unique<MatchingPlotter>(histPath + "TrackChi2/", &registryMatchingTrackChi2);
    int registryIndex = 0;
    for (const auto& [label, func] : matchingChi2Functions) {
      //fMatchingHistogramRegistries[label] = std::make_unique<HistogramRegistry>((std::string("registryMatching") + label).c_str());
      //fMatchingPlotters[label] = std::make_unique<MatchingPlotter>(histPath + label + "/", fMatchingHistogramRegistries[label].get());
      fMatchingPlotters[label] = std::make_unique<MatchingPlotter>(histPath + label + "/", registryMatchingVec[registryIndex]);
      registryIndex += 1;
    }
    for (const auto& [label, response] : matchingMlResponses) {
      //fMatchingHistogramRegistries[label] = std::make_unique<HistogramRegistry>((std::string("registryMatching") + label).c_str());
      //fMatchingPlotters[label] = std::make_unique<MatchingPlotter>(histPath + label + "/", fMatchingHistogramRegistries[label].get());
      fMatchingPlotters[label] = std::make_unique<MatchingPlotter>(histPath + label + "/", (registryMatchingVec[registryIndex]));
      registryIndex += 1;
    }

    fTaggedMuonsMatchingPlotter = std::make_unique<MatchingPlotter>(histPath + "Tagged/", &registryMatching);
    fSelectedMuonsMatchingPlotter = std::make_unique<MatchingPlotter>(histPath + "Selected/", &registryMatching);
  }

  void createAlignmentHistos()
  {
    AxisSpec dxAxis = {200, -10, 10, "#Delta x (cm)"};
    AxisSpec dyAxis = {200, -10, 10, "#Delta y (cm)"};
    AxisSpec dcaxMCHAxis = {400, -10.0, 10.0, "DCA_{x} (cm)"};
    AxisSpec dcayMCHAxis = {400, -10.0, 10.0, "DCA_{y} (cm)"};
    AxisSpec pAxis = {100, 0, 100, "p (GeV/c)"};
    std::array<std::string, 4> quadrants = {"Q0", "Q1", "Q2", "Q3"};
    std::string histPath = "resolution/MC/";

    registryAlignment.add((histPath + "trackDxAtMFTVsP").c_str(), "Track #Delta x vs. p", {HistType::kTH2F, {pAxis, dxAxis}});
    registryAlignment.add((histPath + "trackDyAtMFTVsP").c_str(), "Track #Delta y vs. p", {HistType::kTH2F, {pAxis, dyAxis}});
    registryAlignment.add((histPath + "trackDxAtMFTVsP_alt").c_str(), "Track #Delta x vs. p (alt method)", {HistType::kTH2F, {pAxis, dxAxis}});
    registryAlignment.add((histPath + "trackDyAtMFTVsP_alt").c_str(), "Track #Delta y vs. p (alt method)", {HistType::kTH2F, {pAxis, dyAxis}});

    registryAlignment.add((histPath + "trackDxAtMFTVsP_fake").c_str(), "Track #Delta x vs. p (fake pairs)", {HistType::kTH2F, {pAxis, dxAxis}});
    registryAlignment.add((histPath + "trackDyAtMFTVsP_fake").c_str(), "Track #Delta y vs. p (fake pairs)", {HistType::kTH2F, {pAxis, dyAxis}});
    registryAlignment.add((histPath + "trackDxAtMFTVsP_alt_fake").c_str(), "Track #Delta x vs. p (alt method, fake pairs)", {HistType::kTH2F, {pAxis, dxAxis}});
    registryAlignment.add((histPath + "trackDyAtMFTVsP_alt_fake").c_str(), "Track #Delta y vs. p (alt method, fake pairs)", {HistType::kTH2F, {pAxis, dyAxis}});

    for (size_t j = 0; j < quadrants.size(); j++) {
      const auto& quadrant = quadrants[j];
      dcaHistos[j]["DCA_x"] = registryAlignment.add((histPath + quadrant + "/DCA_x").c_str(), std::format("DCA(x) - {}", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistos[j]["DCA_y"] = registryAlignment.add((histPath + quadrant + "/DCA_y").c_str(), std::format("DCA(y) - {}", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});
      dcaHistos[j]["DCA_x_shifted"] = registryAlignment.add((histPath + quadrant + "/DCA_x_shifted").c_str(), std::format("DCA(x) - {} (shifted)", quadrant).c_str(), {HistType::kTH1F, {dcaxMCHAxis}});
      dcaHistos[j]["DCA_y_shifted"] = registryAlignment.add((histPath + quadrant + "/DCA_y_shifted").c_str(), std::format("DCA(y) - {} (shifted)", quadrant).c_str(), {HistType::kTH1F, {dcayMCHAxis}});
    }

    AxisSpec invMassAxis = {400, 1, 5, "M_{#mu^{+}#mu^{-}} (GeV/c^{2})"};
    // MCH-MID tracks with MCH acceptance cuts
    registryAlignment.add((histPath + "invariantMass").c_str(), "#mu^{+}#mu^{-} invariant mass", {HistType::kTH1F, {invMassAxis}});
    registryAlignment.add((histPath + "invariantMass_shifted").c_str(), "#mu^{+}#mu^{-} invariant mass (shifted)", {HistType::kTH1F, {invMassAxis}});

    //---------

    //fTrackResolutionHistosProd = std::make_unique<TrackResolutionHistos>(histPath + "Prod/", &registryAlignment);
    int registryIndex = 0;
    for (const auto& [label, func] : matchingChi2Functions) {
      fTrackResolutionHistos[label] = std::make_unique<TrackResolutionHistos>(histPath + label + "/", registryAlignmentVec[registryIndex]);
      registryIndex += 1;
    }
    for (const auto& [label, response] : matchingMlResponses) {
      fTrackResolutionHistos[label] = std::make_unique<TrackResolutionHistos>(histPath + label + "/", registryAlignmentVec[registryIndex]);
      registryIndex += 1;
    }

    //fTrackResolutionHistosTaggedMuons = std::make_unique<TrackResolutionHistos>(histPath + "Tagged/", &registryAlignment);
    //fTrackResolutionHistosSelectedMuons = std::make_unique<TrackResolutionHistos>(histPath + "Selected/", &registryAlignment);
  }

  void CreateDimuonHistos()
  {
    AxisSpec invMassAxis = {400, 1, 5, "M_{#mu^{+}#mu^{-}} (GeV/c^{2})"};
    AxisSpec invMassCorrelationAxis = {400, 0, 8, "M_{#mu^{+}#mu^{-}} (GeV/c^{2})"};
    AxisSpec invMassAxisFull = {5000, 0, 100, "M_{#mu^{+}#mu^{-}} (GeV/c^{2})"};
    int matchTypeCombMax = (static_cast<int>(kMatchTypeTrueNonLeading) - 1) * 10 + static_cast<int>(kMatchTypeTrueNonLeading) - 1;
    AxisSpec matchTypeAxis = {matchTypeCombMax + 1, 0, matchTypeCombMax + 1, "match type"};

    // MCH-MID tracks with MCH acceptance cuts
    registryDimuon.add("dimuon/invariantMass_MuonKine_MuonCuts", "#mu^{+}#mu^{-} invariant mass (muon cuts)", {HistType::kTH1F, {invMassAxis}});
    // MCH-MID tracks with MFT acceptance cuts
    registryDimuon.add("dimuon/invariantMass_MuonKine_GlobalMuonCuts", "#mu^{+}#mu^{-} invariant mass (global muon cuts)", {HistType::kTH1F, {invMassAxis}});
    // MCH-MID tracks with MFT acceptance cuts vs. muon tracks match type
    registryDimuon.add("dimuon/invariantMass_MuonKine_GlobalMuonCuts_vs_match_type", "#mu^{+}#mu^{-} invariant mass vs. match tye (global muon cuts)", {HistType::kTH2F, {invMassAxis, matchTypeAxis}});
    // MCH-MID tracks with MFT acceptance cuts, good matches
    registryDimuon.add("dimuon/invariantMass_MuonKine_GlobalMuonCuts_GoodMatches", "#mu^{+}#mu^{-} invariant mass (global muon cuts, good matches)", {HistType::kTH1F, {invMassAxis}});
    // MCH-MID tracks with MFT acceptance cuts, good matches + paired muons
    registryDimuon.add("dimuon/invariantMass_MuonKine_GlobalMuonCuts_GoodMatches_vs_match_type", "#mu^{+}#mu^{-} invariant mass vs. match tye (global muon cuts, good matches)", {HistType::kTH2F, {invMassAxis, matchTypeAxis}});

    // scaled kinematics (Hiroshima method)
    // MFT-MCH-MID tracks with MFT acceptance cuts
    registryDimuon.add("dimuon/invariantMass_ScaledMftKine_GlobalMuonCuts", "#mu^{+}#mu^{-} invariant mass (global muon cuts, rescaled MFT)", {HistType::kTH1F, {invMassAxis}});
    // MCH-MID tracks with MFT acceptance cuts vs. muon tracks match type
    registryDimuon.add("dimuon/invariantMass_ScaledMftKine_GlobalMuonCuts_vs_match_type", "#mu^{+}#mu^{-} invariant mass vs. match tye (global muon cuts, rescaled MFT)", {HistType::kTH2F, {invMassAxis, matchTypeAxis}});
    // MFT-MCH-MID tracks with MFT acceptance cuts, good matches
    registryDimuon.add("dimuon/invariantMass_ScaledMftKine_GlobalMuonCuts_GoodMatches", "#mu^{+}#mu^{-} invariant mass (global muon cuts, rescaled MFT, good matches)", {HistType::kTH1F, {invMassAxis}});
    // MFT-MCH-MID tracks with MFT acceptance cuts vs. muon tracks match type, good matches
    registryDimuon.add("dimuon/invariantMass_ScaledMftKine_GlobalMuonCuts_GoodMatches_vs_match_type", "#mu^{+}#mu^{-} invariant mass vs. match tye (global muon cuts, rescaled MFT, good matches)", {HistType::kTH2F, {invMassAxis, matchTypeAxis}});
}

  void InitMatchingFunctions()
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

      SMatrix55Sym H_k, V_k;
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

      // Update Parameters
      r_k_kminus1 = m_k - H_k * GlobalMuonTrackParameters; // Residuals of prediction

      auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

      // return chi2 and NDF
      return {matchChi2Track, 5};
    };

    //________________________________________________________________________________
    mMatchingFunctionMap["matchXYPhiTanl"] = [](const o2::dataformats::GlobalFwdTrack& mchTrack, const o2::track::TrackParCovFwd& mftTrack) -> std::tuple<double, int> {

      // Match two tracks evaluating positions & angles

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
      //std::cout << std::format("Matching tracks {} and {} with \"matchXYPhiTanl\"");
      /*std::cout << "MCH covariances:" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float cov = GlobalMuonTrackCovariances(i, j);
          std::cout << std::format("  {:0.3f}", cov);
        }
        std::cout << std::endl;
      }
      std::cout << "MFT covariances:" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float cov = V_k(i, j);
          std::cout << std::format("  {:0.3f}", cov);
        }
        std::cout << std::endl;
      }
      std::cout << "invResCov before inversion:" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float cov = invResCov(i, j);
          std::cout << std::format("  {:0.3f}", cov);
        }
        std::cout << std::endl;
      }*/
      invResCov.Invert();
      /*std::cout << "invResCov after inversion:" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float cov = invResCov(i, j);
          std::cout << std::format("  {:0.3f}", cov);
        }
        std::cout << std::endl;
      }*/

      // Residuals of prediction
      r_k_kminus1 = m_k - H_k * GlobalMuonTrackParameters;
      /*std::cout << "r_k_kminus1:" << std::endl;
      for (int i = 0; i < 4; i++) {
        std::cout << std::format("  {:0.3f}", r_k_kminus1(i));
      }
      std::cout << std::endl;*/

      /*std::cout << "Chi2 computation:" << std::endl;
      double sum = 0;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float product = r_k_kminus1(i) * invResCov(i, j) * r_k_kminus1(j);
          sum += product;
          std::cout << std::format("i={} j={} vT({}) * A({}, {}) * v({}) = {:0.3f} * {:0.3f} * {:0.3f} = {:0.3f}  ->  {:0.3f}",
              i, j, i, i, j, j, r_k_kminus1(i), invResCov(i, j), r_k_kminus1(j), product, sum) << std::endl;
        }
      }*/
      auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);
      //std::cout << std::format("Chi2: {:0.3f} / 4 = {:0.3f}", matchChi2Track, matchChi2Track / 4.f) << std::endl;

      // return chi2 and NDF
      return {matchChi2Track, 4};
    };

    //________________________________________________________________________________
    mMatchingFunctionMap["matchXYSxSy"] = [](const o2::dataformats::GlobalFwdTrack& mchTrack, const o2::track::TrackParCovFwd& mftTrack) -> std::tuple<double, int> {

      // Match two tracks evaluating positions & angles
      auto mchTrack2 = FwdtoMCH(mchTrack);
      auto mftTrack2 = FwdtoMCH(mftTrack);

      SMatrix45 H_k;
      SMatrix44 V_k;
      SVector4 m_k(mftTrack2.getNonBendingCoor(), mftTrack2.getNonBendingSlope(),
                   mftTrack2.getBendingCoor(), mftTrack2.getBendingSlope()),
              r_k_kminus1;
      SVector5 GlobalMuonTrackParameters(mchTrack2.getNonBendingCoor(), mchTrack2.getNonBendingSlope(),
                                         mchTrack2.getBendingCoor(), mchTrack2.getBendingSlope(),
                                         mchTrack2.getInverseBendingMomentum());
      SMatrix55Sym GlobalMuonTrackCovariances;
      GlobalMuonTrackCovariances(0, 0) = mchTrack2.getCovariances()(0, 0);
      GlobalMuonTrackCovariances(0, 1) = mchTrack2.getCovariances()(0, 1);
      GlobalMuonTrackCovariances(1, 1) = mchTrack2.getCovariances()(1, 1);
      GlobalMuonTrackCovariances(0, 2) = mchTrack2.getCovariances()(0, 2);
      GlobalMuonTrackCovariances(1, 2) = mchTrack2.getCovariances()(1, 2);
      GlobalMuonTrackCovariances(2, 2) = mchTrack2.getCovariances()(2, 2);
      GlobalMuonTrackCovariances(0, 3) = mchTrack2.getCovariances()(0, 3);
      GlobalMuonTrackCovariances(1, 3) = mchTrack2.getCovariances()(1, 3);
      GlobalMuonTrackCovariances(2, 3) = mchTrack2.getCovariances()(2, 3);
      GlobalMuonTrackCovariances(3, 3) = mchTrack2.getCovariances()(3, 3);
      GlobalMuonTrackCovariances(0, 4) = mchTrack2.getCovariances()(0, 4);
      GlobalMuonTrackCovariances(1, 4) = mchTrack2.getCovariances()(1, 4);
      GlobalMuonTrackCovariances(2, 4) = mchTrack2.getCovariances()(2, 4);
      GlobalMuonTrackCovariances(3, 4) = mchTrack2.getCovariances()(3, 4);
      GlobalMuonTrackCovariances(4, 4) = mchTrack2.getCovariances()(4, 4);

      V_k(0, 0) = mftTrack2.getCovariances()(0, 0);
      V_k(1, 1) = mftTrack2.getCovariances()(1, 1);
      V_k(2, 2) = mftTrack2.getCovariances()(2, 2);
      V_k(3, 3) = mftTrack2.getCovariances()(3, 3);

      H_k(0, 0) = 1.0;
      H_k(1, 1) = 1.0;
      H_k(2, 2) = 1.0;
      H_k(3, 3) = 1.0;

      // Covariance of residuals
      SMatrix44 invResCov = (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
      //std::cout << std::format("Matching tracks {} and {} with \"matchXYPhiTanl\"");
      /*std::cout << "MCH covariances:" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float cov = GlobalMuonTrackCovariances(i, j);
          std::cout << std::format("  {:0.3f}", cov);
        }
        std::cout << std::endl;
      }
      std::cout << "MFT covariances:" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float cov = V_k(i, j);
          std::cout << std::format("  {:0.3f}", cov);
        }
        std::cout << std::endl;
      }
      std::cout << "invResCov before inversion:" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float cov = invResCov(i, j);
          std::cout << std::format("  {:0.3f}", cov);
        }
        std::cout << std::endl;
      }*/
      invResCov.Invert();
      /*std::cout << "invResCov after inversion:" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float cov = invResCov(i, j);
          std::cout << std::format("  {:0.3f}", cov);
        }
        std::cout << std::endl;
      }*/

      // Residuals of prediction
      r_k_kminus1 = m_k - H_k * GlobalMuonTrackParameters;
      /*std::cout << "r_k_kminus1:" << std::endl;
      for (int i = 0; i < 4; i++) {
        std::cout << std::format("  {:0.3f}", r_k_kminus1(i));
      }
      std::cout << std::endl;*/

      /*std::cout << "Chi2 computation:" << std::endl;
      double sum = 0;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          float product = r_k_kminus1(i) * invResCov(i, j) * r_k_kminus1(j);
          sum += product;
          std::cout << std::format("i={} j={} vT({}) * A({}, {}) * v({}) = {:0.3f} * {:0.3f} * {:0.3f} = {:0.3f}  ->  {:0.3f}",
              i, j, i, i, j, j, r_k_kminus1(i), invResCov(i, j), r_k_kminus1(j), product, sum) << std::endl;
        }
      }*/
      auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);
      //std::cout << std::format("Chi2: {:0.3f} / 4 = {:0.3f}", matchChi2Track, matchChi2Track / 4.f) << std::endl;

      // return chi2 and NDF
      return {matchChi2Track, 4};
    };

    //________________________________________________________________________________
    mMatchingFunctionMap["matchXY"] = [](const o2::dataformats::GlobalFwdTrack& mchTrack, const o2::track::TrackParCovFwd& mftTrack) -> std::tuple<double, int> {

      // Calculate Matching Chi2 - X and Y positions

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

      // Residuals of prediction
      r_k_kminus1 = m_k - H_k * GlobalMuonTrackParameters;
      auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

      // return reduced chi2
      return {matchChi2Track, 2};
    };

    //________________________________________________________________________________
    mMatchingFunctionMap["matchPhiTanl"] = [](const o2::dataformats::GlobalFwdTrack& mchTrack, const o2::track::TrackParCovFwd& mftTrack) -> std::tuple<double, int> {

      // Calculate Matching Chi2 - X and Y positions

      SMatrix25 H_k;
      SMatrix22 V_k;
      SMatrix22 G_k;
      SVector5 GlobalMuonTrackParameters = mchTrack.getParameters();
      SVector2 m_k(mftTrack.getPhi(), mftTrack.getTanl()),
          g_k(GlobalMuonTrackParameters(2), GlobalMuonTrackParameters(3)), r_k_kminus1;
      SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
      V_k(0, 0) = mftTrack.getCovariances()(2, 2);
      V_k(1, 1) = mftTrack.getCovariances()(3, 3);
      H_k(0, 0) = 1.0;
      H_k(1, 1) = 1.0;
      G_k(0, 0) = GlobalMuonTrackCovariances(2, 2);
      G_k(0, 1) = GlobalMuonTrackCovariances(2, 3);
      G_k(1, 0) = GlobalMuonTrackCovariances(3, 2);
      G_k(1, 1) = GlobalMuonTrackCovariances(3, 3);

      // Covariance of residuals
      SMatrix22 invResCov = (V_k + G_k);
      invResCov.Invert();

      // Residuals of prediction
      r_k_kminus1 = m_k - g_k;
      auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

      // return reduced chi2
      return {matchChi2Track, 2};
    };
  }

  void init(o2::framework::InitContext&)
  {
    // Load geometry
    ccdbManager->setURL(ccdburl);
    ccdbManager->setCaching(true);
    ccdbManager->setLocalObjectValidityChecking();
    fCCDBApi.init(ccdburl);
    mRunNumber = 0;

    if (!o2::base::GeometryManager::isGeometryLoaded()) {
      LOGF(info, "Load geometry from CCDB");
      ccdbManager->get<TGeoManager>(geoPath);
    }

    // Matching functions
    InitMatchingFunctions();
    for (size_t funcId = 0; funcId < sChi2FunctionsNum; funcId++) {
      auto label = fConfigChi2MatchingOptions.fFunctionLabel[funcId].value;
      auto funcName = fConfigChi2MatchingOptions.fFunctionName[funcId].value;
      auto scoreMin = fConfigChi2MatchingOptions.fMatchingScoreCut[funcId].value;
      auto matchingPlaneZ = fConfigChi2MatchingOptions.fMatchingPlaneZ[funcId].value;
      auto extrapMethod = fConfigChi2MatchingOptions.fMatchingExtrapMethod[funcId].value;

      if (label == "" || funcName == "")
        continue;

      matchingChi2Functions[label] = funcName;

      matchingScoreCuts[label] = scoreMin;
      matchingPlanesZ[label] = matchingPlaneZ;
      matchingExtrapMethod[label] = extrapMethod;
    }

    // Matching ML models
    // TODO : for now we use hard coded values since the current models use 1 pT bin
    binsPtMl = {-1e-6, 1000.0};
    cutValues = {0.0};
    cutDirMl = {cuts_ml::CutNot};
    o2::framework::LabeledArray<double> mycutsMl(cutValues.data(), 1, 1, std::vector<std::string>{"pT bin 0"}, std::vector<std::string>{"score"});

    for (size_t modelId = 0; modelId < sMLModelsNum; modelId++) {
      auto label = fConfigMlOptions.fModelLabel[modelId].value;
      auto modelPaths = fConfigMlOptions.fModelPathsCCDB[modelId].value;
      auto inputFeatures = fConfigMlOptions.fInputFeatures[modelId].value;
      auto modelNames = fConfigMlOptions.fModelNames[modelId].value;
      auto scoreMin = fConfigMlOptions.fMatchingScoreCut[modelId].value;
      auto matchingPlaneZ = fConfigMlOptions.fMatchingPlaneZ[modelId].value;
      auto extrapMethod = fConfigMlOptions.fMatchingExtrapMethod[modelId].value;

      if (label == "" || modelPaths.empty() || inputFeatures.empty() || modelNames.empty())
        break;

      matchingMlResponses[label].configure(binsPtMl, mycutsMl, cutDirMl, 1);
      matchingMlResponses[label].setModelPathsCCDB(modelNames, fCCDBApi, modelPaths, fConfigCCDB.fConfigNoLaterThan.value);
      matchingMlResponses[label].cacheInputFeaturesIndices(inputFeatures);
      matchingMlResponses[label].init();

      matchingScoreCuts[label] = scoreMin;
      matchingPlanesZ[label] = matchingPlaneZ;
      matchingExtrapMethod[label] = extrapMethod;
    }

    int nTrackTypes = static_cast<int>(o2::aod::fwdtrack::ForwardTrackTypeEnum::MCHStandaloneTrack) + 1;
    AxisSpec trackTypeAxis = {static_cast<int>(nTrackTypes), 0.0, static_cast<double>(nTrackTypes), "track type"};
    registry.add("nTracksPerType", "Number of tracks per type", {HistType::kTH1F, {trackTypeAxis}});

    AxisSpec tracksMultiplicityAxis = {10000, 0, 10000, "tracks multiplicity"};
    registry.add("tracksMultiplicityMFT", "MFT tracks multiplicity", {HistType::kTH1F, {tracksMultiplicityAxis}});
    registry.add("tracksMultiplicityMCH", "MCH tracks multiplicity", {HistType::kTH1F, {tracksMultiplicityAxis}});
    AxisSpec dchi2Axis = {100, 0, 100, "#Delta#chi^{2}"};
    registry.add("matchCandidatesDeltaChi2", "#Delta#chi^{2} of match candidates", {HistType::kTH1F, {dchi2Axis}});

    createMatchingHistosMC();
    createAlignmentHistos();
    CreateDimuonHistos();
  }

  template <class T, class C>
  bool pDCACut(const T& mchTrack, const C& collision, double nSigmaPDCA)
  {
    static const double sigmaPDCA23 = 80.;
    static const double sigmaPDCA310 = 54.;
    static const double relPRes = 0.0004;
    static const double slopeRes = 0.0005;

    double thetaAbs = TMath::ATan(mchTrack.rAtAbsorberEnd() / 505.) * TMath::RadToDeg();

    // propagate muon track to vertex
    auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToVertex);

    // double pUncorr = mchTrack.p();
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

  template <class T, class C>
  bool IsGoodMuon(const T& mchTrack, const C& collision,
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
    if (!pDCACut(mchTrack, collision, nSigmaPdcaCut)) {
      return false;
    }

    return true;
  }

  template <class T, class C>
  bool IsGoodMuon(const T& muonTrack, const C& collision)
  {
    return IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMchLow, fEtaMchUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp);
  }

  template <class T, class C>
  bool IsGoodGlobalMuon(const T& muonTrack, const C& collision)
  {
    if (static_cast<int>(muonTrack.trackType()) != 3)
      return false;

    return IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp);
  }

  template <class T>
  bool IsGoodMFT(const T& mftTrack,
                 double chi2Cut,
                 int nClustersCut)
  {
    // std::cout << std::format("Checking MFT track") << std::endl;
    // std::cout << std::format("    chi2={}", mftTrack.chi2()) << std::endl;
    // std::cout << std::format("    nClusters={}", mftTrack.nClusters()) << std::endl;
    //  chi2 cut
    if (mftTrack.chi2() > chi2Cut)
      return false;

    // number of clusters cut
    if (mftTrack.nClusters() < nClustersCut)
      return false;

    return true;
  }

  template <class T>
  bool IsGoodMFT(const T& mftTrack)
  {
    return IsGoodMFT(mftTrack, fTrackChi2MftUp, fTrackNClustMftLow);
  }

  template <class TMUON>
  bool IsGoodGlobalMatching(const TMUON& muonTrack,
                            double matchingScore,
                            double matchingScoreCut)
  {
    if (static_cast<int>(muonTrack.trackType()) > 2)
      return false;

    // MFT-MCH matching score cut
    if (matchingScore < matchingScoreCut)
      return false;

    return true;
  }

  template <class TMUON>
  bool IsGoodGlobalMatching(const TMUON& muonTrack, double matchingScore)
  {
    return IsGoodGlobalMatching(muonTrack, matchingScore, fMatchingChi2ScoreMftMchLow);
  }

  template <class TMUON>
  bool IsTrueGlobalMatching(const TMUON& muonTrack, const std::vector<std::pair<int64_t, int64_t>>& matchablePairs)
  {
    if (static_cast<int>(muonTrack.trackType()) > 2)
      return false;

    int64_t mchTrackId = static_cast<int64_t>(muonTrack.matchMCHTrackId());
    int64_t mftTrackId = static_cast<int64_t>(muonTrack.matchMFTTrackId());

    std::pair<int64_t, int64_t> trackIndexes = std::make_pair(mchTrackId, mftTrackId);

    return (std::find(matchablePairs.begin(), matchablePairs.end(), trackIndexes) != matchablePairs.end());
  }

  bool IsMatchableMCH(int64_t mchTrackId, const std::vector<std::pair<int64_t, int64_t>>& matchablePairs)
  {
    for (auto [id1, id2] : matchablePairs) {
      if (mchTrackId == id1)
        return true;
    }
    return false;
  }

  std::optional<std::pair<int64_t, int64_t>> GetMatchablePairForMCH(int64_t mchTrackId, const std::vector<std::pair<int64_t, int64_t>>& matchablePairs)
  {
    for (auto pair : matchablePairs) {
      if (mchTrackId == pair.first)
        return pair;
    }
    return {};
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

  template <typename T, typename C>
  o2::dataformats::GlobalFwdTrack FwdToTrackPar(const T& track, const C& cov)
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

  template <typename T, typename TMCH, typename TMFT>
  float GetTrackChi2OverNDF(const T& track, const TMCH&, const TMFT&)
  {
    int nClusters = 0;
    if (static_cast<int>(track.trackType()) <= 2) {
      // global muon track
      auto const& mchTrack = track.template matchMCHTrack_as<TMCH>();
      auto const& mftTrack = track.template matchMFTTrack_as<TMFT>();
      nClusters = mchTrack.nClusters() + mftTrack.nClusters();
    } else {
      nClusters = track.nClusters();
    }
    //std::cout << std::format("Track type {}, nClusters: {}", static_cast<int>(track.trackType()), nClusters) << std::endl;
    return (track.chi2() / (nClusters - 5));
  }

  template <typename T>
  float GetTrackCharge(const T& track)
  {
    return (track.signed1Pt() >= 0 ? 1.f : -1.f);
  }

  template <typename T>
  float GetParticleCharge(const T& mcParticle)
  {
    if (std::abs(mcParticle.pdgCode()) >= 11 && std::abs(mcParticle.pdgCode()) <= 18) {
      // leptons, negative particles and positive anti-particles
      return mcParticle.pdgCode() < 0 ? 1.f : -1.f;
    } else {
      // other particles
      return mcParticle.pdgCode() < 0 ? -1.f : 1.f;
    }
  }

  template <typename T>
  o2::dataformats::GlobalFwdTrack ParticleToTrackPar(const T& mcParticle)
  {
    o2::mch::TrackParam track;

    double z = mcParticle.vz();
    double x = mcParticle.vx();
    double y = mcParticle.vy();
    double xSlope = mcParticle.px() / mcParticle.pz();
    double ySlope = mcParticle.py() / mcParticle.pz();
    double pz = mcParticle.pz();

    track.setZ(z);
    track.setBendingCoor(y);
    track.setNonBendingCoor(x);
    track.setBendingSlope(ySlope);
    track.setNonBendingSlope(xSlope);

    double pyz = -pz * TMath::Sqrt(1.0 + ySlope * ySlope);
    double charge = GetParticleCharge(mcParticle);

    track.setInverseBendingMomentum(charge / pyz);

    //std::cout << std::format("[ParticleToTrackPar] pMC: {}  pTrack: {}", mcParticle.p(), track.p()) << std::endl;

    return mExtrap.MCHtoFwd(track);
  }

  o2::dataformats::GlobalFwdTrack PropagateToZMCH(const o2::dataformats::GlobalFwdTrack& muon, const double z)
  {
    auto mchTrack = mExtrap.FwdtoMCH(muon);

    float absFront = -90.f;
    float absBack = -505.f;

    if (muon.getZ() < absBack && z > absFront) {
      // extrapolation through the absorber in the upstream direction
      //std::cout << std::format("Calling extrapToVertexWithoutBranson({})", z) << std::endl;
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrack, z);
      //std::cout << std::format("  mchTrack.getZ()={}", mchTrack.getZ()) << std::endl;
    } if (muon.getZ() < absBack && z <= absFront && z > absBack) {
      // extrapolation inside the absorber in the upstream direction
      // first extrapolate through the whole absorber, correcting for energy loss
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrack, absFront + 0.1f);
      // then extrapolate back to the given z
      o2::mch::TrackExtrap::extrapToZCov(mchTrack, z);
    } else {
      // all other cases
      //std::cout << std::format("Calling extrapToZCov({})", z) << std::endl;
      o2::mch::TrackExtrap::extrapToZCov(mchTrack, z);
      //std::cout << std::format("  mchTrack.getZ()={}", mchTrack.getZ()) << std::endl;
    }

    auto proptrack = mExtrap.MCHtoFwd(mchTrack);
    o2::dataformats::GlobalFwdTrack propmuon;
    propmuon.setParameters(proptrack.getParameters());
    propmuon.setZ(proptrack.getZ());
    propmuon.setCovariances(proptrack.getCovariances());

    return propmuon;
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
    track.setParameters(fwdtrack.getParameters());
    track.setZ(fwdtrack.getZ());
    track.setCovariances(fwdtrack.getCovariances());

    return PropagateToZMCH(track, z);

    /*auto mchTrack = mExtrap.FwdtoMCH(track);

    float absFront = -90.f;
    float absBack = -505.f;

    if (fwdtrack.getZ() < absBack && z > absFront) {
      // extrapolation through the absorber in the upstream direction
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrack, z);
    } else {
      // all other cases
      o2::mch::TrackExtrap::extrapToZCov(mchTrack, z);
    }

    auto proptrack = mExtrap.MCHtoFwd(mchTrack);
    o2::dataformats::GlobalFwdTrack propmuon;
    propmuon.setParameters(proptrack.getParameters());
    propmuon.setZ(proptrack.getZ());
    propmuon.setCovariances(proptrack.getCovariances());

    return propmuon;*/
  }

  o2::dataformats::GlobalFwdTrack PropagateToZMFT(const o2::dataformats::GlobalFwdTrack& mftTrack, const double z)
  {
    o2::dataformats::GlobalFwdTrack fwdtrack{mftTrack};
    fwdtrack.propagateToZ(z, mBzAtMftCenter);
    return fwdtrack;
  }

  template <typename TMFT, typename CMFT>
  o2::dataformats::GlobalFwdTrack PropagateToZMFT(const TMFT& mftTrack, const CMFT& mftCov, const double z)
  {
    o2::dataformats::GlobalFwdTrack fwdtrack = FwdToTrackPar(mftTrack, mftCov);
    return PropagateToZMFT(fwdtrack, z);
  }

  // method 0: standard extrapolation
  // method 1: MFT extrapolation using MCH momentum
  // method 2: MCH track extrapolation constrained to the first MFT track point, MFT extrapolation using MCH momentum
  // method 3: MCH track extrapolation constrained to the collision point, MFT extrapolation using MCH momentum
  template <typename TMCH, typename TMFT, typename CMFT, typename C>
  o2::dataformats::GlobalFwdTrack PropagateToMatchingPlaneMCH(const TMCH& mchTrack, const TMFT& mftTrack, const CMFT& mftTrackCov, const C& collision, const double z, int method)
  {
    if (method == 0 || method == 1) {
      // simple extrapolation upstream through the absorber
      return PropagateToZMCH(mchTrack, z);
    }

    if (method == 2) {
      // extrapolation to the first MFT point and then back to the matching plane
      auto mftTrackPar = FwdToTrackPar(mftTrack, mftTrackCov);
      //std::cout << std::format("[PropagateToMatchingPlaneMCH] extrapolating to MFT: x={:0.3f} y={:0.3f} z={:0.3f}", mftTrackPar.getX(), mftTrackPar.getY(), mftTrackPar.getZ()) << std::endl;
      auto mchTrackAtMFT = PropagateToVertexMCH(FwdToTrackPar(mchTrack, mchTrack),
                                                mftTrackPar.getX(), mftTrackPar.getY(), mftTrackPar.getZ(),
                                                mftTrackPar.getSigma2X(), mftTrackPar.getSigma2Y());
      //std::cout << std::format("[PropagateToMatchingPlaneMCH] extrapolating to z={:0.3f}", z) << std::endl;
      return PropagateToZMCH(mchTrackAtMFT, z);
    }

    if (method == 3) {
      // extrapolation to the vertex and then back to the matching plane
      auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToVertex);
      return PropagateToZMCH(mchTrackAtVertex, z);
    }

    if (method == 4) {
      // extrapolation to the MFT DCA and then back to the matching plane
      auto mftTrackDCA = PropagateToZMFT(FwdToTrackPar(mftTrack, mftTrackCov), collision.posZ());
      auto mchTrackAtDCA = PropagateToVertexMCH(FwdToTrackPar(mchTrack, mchTrack),
                                                mftTrackDCA.getX(), mftTrackDCA.getY(), mftTrackDCA.getZ(),
                                                mftTrackDCA.getSigma2X(), mftTrackDCA.getSigma2Y());
      return PropagateToZMCH(mchTrackAtDCA, z);
    }

    return {};
  }

  template <typename TMCH, typename TMFT, typename CMFT, typename C>
  o2::dataformats::GlobalFwdTrack PropagateToMatchingPlaneMFT(const TMCH& mchTrack, const TMFT& mftTrack, const CMFT& mftTrackCov, const C& collision, const double z, int method)
  {
    if (method == 0) {
      // extrapolation with MFT tools
      return PropagateToZMFT(mftTrack, mftTrackCov, z);
    }

    if (method > 0) {
      // extrapolation with MCH tools
      auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToVertex);
      double pMCH = mchTrackAtVertex.getP();
      double px = pMCH * sin(M_PI / 2 - atan(mftTrack.tgl())) * cos(mftTrack.phi());
      double py = pMCH * sin(M_PI / 2 - atan(mftTrack.tgl())) * sin(mftTrack.phi());
      double pt = std::sqrt(std::pow(px, 2) + std::pow(py, 2));
      double sign = mchTrack.sign();

      o2::dataformats::GlobalFwdTrack track = FwdToTrackPar(mftTrack, mftTrackCov);

      // update momentum in track parameters and errors
      auto newCov = track.getCovariances();
      newCov(4, 4) = mchTrackAtVertex.getSigma2InvQPt();
      track.setCovariances(newCov);
      track.setInvQPt(sign / pt);

      auto trackExt = mExtrap.FwdtoMCH(track);
      o2::mch::TrackExtrap::extrapToZCov(trackExt, z);

      o2::dataformats::GlobalFwdTrack propmuon;
      auto proptrack = mExtrap.MCHtoFwd(trackExt);
      propmuon.setParameters(proptrack.getParameters());
      propmuon.setZ(proptrack.getZ());
      propmuon.setCovariances(proptrack.getCovariances());

      return propmuon;
    }

    return {};
  }

  o2::dataformats::GlobalFwdTrack PropagateToVertexMCH(const o2::dataformats::GlobalFwdTrack& muon,
                                                       const double vx, const double vy, const double vz,
                                                       const double covVx, const double covVy)
  {
    auto mchTrack = mExtrap.FwdtoMCH(muon);

    o2::mch::TrackExtrap::extrapToVertex(mchTrack, vx, vy, vz, covVx, covVy);

    auto proptrack = mExtrap.MCHtoFwd(mchTrack);
    o2::dataformats::GlobalFwdTrack propmuon;
    propmuon.setParameters(proptrack.getParameters());
    propmuon.setZ(proptrack.getZ());
    propmuon.setCovariances(proptrack.getCovariances());

    return propmuon;
  }

  template <class TMCH, class C>
  o2::dataformats::GlobalFwdTrack PropagateToVertexMCH(const TMCH& muon,
                                                       const C& collision)
  {
    return PropagateToVertexMCH(FwdToTrackPar(muon, muon),
                                collision.posX(),
                                collision.posY(),
                                collision.posZ(),
                                collision.covXX(),
                                collision.covYY());
  }

  o2::dataformats::GlobalFwdTrack PropagateToVertexMFT(o2::dataformats::GlobalFwdTrack muon,
                                                       const double vx, const double vy, const double vz,
                                                       const double covVx, const double covVy)
  {
    o2::dataformats::GlobalFwdTrack propmuon;
    auto geoMan = o2::base::GeometryManager::meanMaterialBudget(muon.getX(), muon.getY(), muon.getZ(), vx, vy, vz);
    auto x2x0 = static_cast<float>(geoMan.meanX2X0);
    muon.propagateToVtxhelixWithMCS(vz, {vx, vy}, {covVx, covVy}, mBzAtMftCenter, x2x0);
    propmuon.setParameters(muon.getParameters());
    propmuon.setZ(muon.getZ());
    propmuon.setCovariances(muon.getCovariances());

    return propmuon;
  }

  template <class TMFT, class C>
  o2::dataformats::GlobalFwdTrack PropagateToVertexMFT(const TMFT& muon,
                                                       const C& collision)
  {
    return PropagateToVertexMFT(FwdToTrackPar(muon),
                                collision.posX(),
                                collision.posY(),
                                collision.posZ(),
                                collision.covXX(),
                                collision.covYY());
  }

  void UpdateTrackMomentum(o2::mch::TrackParam& track, const o2::mch::TrackParam& track4mom)
  {
    double pRatio = track.p() / track4mom.p();
    double newInvBendMom = track.getInverseBendingMomentum() * pRatio;
    track.setInverseBendingMomentum(newInvBendMom);
    track.setCharge(track4mom.getCharge());
  }

  template <typename TMCH, typename TMFT, class C>
  o2::dataformats::GlobalFwdTrack PropagateToVertexMFT(const TMFT& muon,
                                                       const TMCH& mchTrack,
                                                       const C& collision)
  {
    // extrapolation with MCH tools
    auto mchTrackAtMFT = mExtrap.FwdtoMCH(FwdToTrackPar(mchTrack));
    o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrackAtMFT, muon.z());

    auto muonTrackProp = mExtrap.FwdtoMCH(FwdToTrackPar(muon));
    UpdateTrackMomentum(muonTrackProp, mchTrackAtMFT);
    o2::mch::TrackExtrap::extrapToVertex(muonTrackProp,
                                         collision.posX(),
                                         collision.posY(),
                                         collision.posZ(),
                                         collision.covXX(),
                                         collision.covYY());

    return mExtrap.MCHtoFwd(muonTrackProp);
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

  template <class T>
  int GetQuadrant(const T& track)
  {
    double phi = track.phi() * 180 / TMath::Pi();
    return GetQuadrant(phi);
  }

  void doTransformMCH(o2::mch::TrackParam& track, bool inverseTransform = true)
  {
    float sign = inverseTransform ? -1.f : 1.f;

    double zCH1 = firstMCHPlaneZ;

    double alpha1 = track.getNonBendingSlope();
    double alpha3 = track.getBendingSlope();
    double phi = TMath::ATan2(-alpha3, -alpha1) * 180 / TMath::Pi();
    int quadrant = GetQuadrant(phi);

    // double deltaX[4]{ 0.1003, 0.2825, 0.1807, -0.0588 };
    double deltaX[4]{0.1513 * sign, 0.2954 * sign, 0.1795 * sign, -0.0847 * sign};
    // double deltaY[4]{ 0.0806, 0.1589, 0.8761, 0.7888 };
    double deltaY[4]{0.1328 * sign, 0.1719 * sign, 0.8538 * sign, 0.8049 * sign};
    double deltaThetaX[4]{-0.0030 * sign, -0.0482 * sign, -0.0499 * sign, -0.0233 * sign};
    double deltaThetaY[4]{0.0018 * sign, 0.0018 * sign, 0.0731 * sign, 0.0669 * sign};

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
    // std::cout << std::format("[TOTO] MFT y={:0.3f}  xShift={:0.3f}",
    //     y, xShiftMCH) << std::endl;
    double xPositionCorrected = x - xShiftAtCH1 + xSlopeCorrected * (z - zCH1);
    track.setNonBendingCoor(xPositionCorrected);

    double ySlopeCorrected = ySlope - deltaSlopeY[quadrant];
    track.setBendingSlope(ySlopeCorrected);
    double yShiftAtCH1 = deltaY[quadrant];
    // std::cout << std::format("[TOTO] MFT y={:0.3f}  xShift={:0.3f}",
    //     y, xShiftMCH) << std::endl;
    double yPositionCorrected = y - yShiftAtCH1 + ySlopeCorrected * (z - zCH1);
    track.setBendingCoor(yPositionCorrected);
  }

  void TransformMCH(o2::dataformats::GlobalFwdTrack& track)
  {
    auto mchTrack = mExtrap.FwdtoMCH(track);

    doTransformMCH(mchTrack);

    auto transformedTrack = mExtrap.MCHtoFwd(mchTrack);
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

    auto mchTrack = mExtrap.FwdtoMCH(track);

    doTransformMCH(mchTrack);

    auto transformedTrack = mExtrap.MCHtoFwd(mchTrack);
    fwdtrack.setParameters(transformedTrack.getParameters());
    fwdtrack.setZ(transformedTrack.getZ());
    fwdtrack.setCovariances(transformedTrack.getCovariances());
  }

  template <class MCP>
  void GetMotherParticles(MCP const& mcParticle, std::vector<std::pair<int64_t, int64_t>>& motherParticlesVec)
  {
    const auto& motherParticles = mcParticle.template mothers_as<aod::McParticles>();
    if (motherParticles.empty()) {
      return;
    }

    const auto& motherParticle = motherParticles[0];
    motherParticlesVec.emplace_back(std::make_pair(static_cast<int64_t>(motherParticle.pdgCode()), static_cast<int64_t>(motherParticle.globalIndex())));
    GetMotherParticles(motherParticle, motherParticlesVec);
  }

  template <class T>
  std::vector<std::pair<int64_t, int64_t>> GetMotherParticles(T const& track)
  {
    std::vector<std::pair<int64_t, int64_t>> result;
    if (!track.has_mcParticle())
      return result;

    const auto& mcParticle = track.mcParticle();
    result.emplace_back(std::make_pair(static_cast<int64_t>(mcParticle.pdgCode()), static_cast<int64_t>(mcParticle.globalIndex())));

    GetMotherParticles(mcParticle, result);

    return result;
  }

  template <class TMCH, class TMFTs>
  int GetDecayRanking(TMCH const& mchTrack, TMFTs const& mftTracks)
  {
    auto mchMotherParticles = GetMotherParticles(mchTrack);

    int decayRanking = 0;
    // search for an MFT track that is associated to one of the MCH mother particles
    for (const auto& mftTrack : mftTracks) {
      // skip tracks that do not have an associated MC particle
      if (!mftTrack.has_mcParticle())
        continue;
      // get the index associated to the MC particle
      auto mftMcParticle = mftTrack.mcParticle();
      int64_t mftMcTrackIndex = mftMcParticle.globalIndex();

      int ranking = 1;
      for (const auto& mother : mchMotherParticles) {
        if (mother.second == mftMcTrackIndex) {
          decayRanking = ranking;
          break;
        }
        ranking += 1;
      }

      if (decayRanking > 0) {
        break;
      }
    }
    return decayRanking;
  }

  template <class TMUON, class TMFT>
  void FillMatchablePairs(CollisionInfo& collisionInfo,
                         TMUON const& muonTracks,
                         TMFT const& mftTracks)
  {
    collisionInfo.matchablePairs.clear();
    for (const auto& muonTrack : muonTracks) {
      // only consider MCH standalone or MCH-MID matches
      if (static_cast<int>(muonTrack.trackType()) <= 2)
        continue;

      // only consider tracks associated to the current collision
      if (!muonTrack.has_collision())
        continue;
      if (muonTrack.collisionId() != collisionInfo.index)
        continue;

      // skip tracks that do not have an associated MC particle
      if (!muonTrack.has_mcParticle())
        continue;
      // get the index associated to the MC particle
      auto muonMcParticle = muonTrack.mcParticle();
      if (std::abs(muonMcParticle.pdgCode()) != 13)
        continue;

      int64_t muonMcTrackIndex = muonMcParticle.globalIndex();

      for (const auto& mftTrack : mftTracks) {
        // skip tracks that do not have an associated MC particle
        if (!mftTrack.has_mcParticle())
          continue;
        // get the index associated to the MC particle
        auto mftMcParticle = mftTrack.mcParticle();
        int64_t mftMcTrackIndex = mftMcParticle.globalIndex();

        if (muonMcTrackIndex == mftMcTrackIndex) {
          collisionInfo.matchablePairs.emplace_back(std::make_pair(static_cast<int64_t>(muonTrack.globalIndex()),
                                                                   static_cast<int64_t>(mftTrack.globalIndex())));
          // std::cout << std::format("Added matchable pair: {} / {}", muonTrack.globalIndex(), mftTrack.globalIndex()) << std::endl;
          // auto const& pairedMchTrack = muonTracks.rawIteratorAt(muonTrack.globalIndex());
          // auto const& pairedMftTrack = mftTracks.rawIteratorAt(mftTrack.globalIndex());
          // std::cout << std::format("  MCH chi2: {}", pairedMchTrack.chi2()) << std::endl;
          // std::cout << std::format("  MFT chi2: {}", pairedMftTrack.chi2()) << std::endl;

        /*} else {
          // check if the muon particle is a decay product of the MFT particle
          for (auto& motherParticle : muonMcParticle.template mothers_as<aod::McParticles>()) {
            if (motherParticle.globalIndex() == mftMcTrackIndex) {
              matchablePairs.emplace_back(std::make_pair(static_cast<int64_t>(muonTrack.globalIndex()),
                                                         static_cast<int64_t>(mftTrack.globalIndex())));
              // std::cout << std::format("Added matchable pair from decay: {} / {}", muonTrack.globalIndex(), mftTrack.globalIndex()) << std::endl;
              break;
            }
          }*/
        }
      }
    }
  }

  template <class TMUON>
  int GetTrueMatchIndex(TMUON const& muonTracks,
                        const std::vector<MatchingCandidate>& matchCandidatesVector,
                        const std::vector<std::pair<int64_t, int64_t>>& matchablePairs)
  {
    // find the index of the matching candidate that corresponds to the true match
    // index=1 corresponds to the leading candidate
    // index=0 means no candidate was found that corresponds to the true match
    int trueMatchIndex = 0;
    for (size_t i = 0; i < matchCandidatesVector.size(); i++) {
      auto const& muonTrack = muonTracks.rawIteratorAt(matchCandidatesVector[i].globalTrackId);

      if (IsTrueGlobalMatching(muonTrack, matchablePairs)) {
        trueMatchIndex = i + 1;
        break;
      }
    }
    return trueMatchIndex;
  }

  template <class TMCH, class TMFT>
  bool IsMuon(const TMCH& mchTrack,
              const TMFT& mftTrack)
  {
    // skip tracks that do not have an associated MC particle
    if (!mchTrack.has_mcParticle()) return false;
    if (!mftTrack.has_mcParticle()) return false;

    // get the index associated to the MC particles
    auto mchMcParticle = mchTrack.mcParticle();
    auto mftMcParticle = mftTrack.mcParticle();
    if (mchMcParticle.globalIndex() != mftMcParticle.globalIndex()) return false;

    if (std::abs(mchMcParticle.pdgCode()) != 13) return false;

    return true;
  }

  template <class TMUON, class TMUONS, class TMFTS>
  bool IsMuon(const TMUON& muonTrack,
              TMUONS const& muonTracks,
              TMFTS const& mftTracks)
  {
    if (static_cast<int>(muonTrack.trackType()) >= 2)
      return false;

    auto const& mchTrack = muonTrack.template matchMCHTrack_as<TMUONS>();
    auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFTS>();

    return IsMuon(mchTrack, mftTrack);
  }

  template <class TMUON, class TMUONS, class TMFTS>
  MuonMatchType GetMatchType(const TMUON& muonTrack,
                             TMUONS const& muonTracks,
                             TMFTS const& mftTracks,
                             const std::vector<std::pair<int64_t, int64_t>>& matchablePairs,
                             int ranking)
  {
    if (static_cast<int>(muonTrack.trackType()) > 2)
      return kMatchTypeUndefined;

    auto const& mchTrack = muonTrack.template matchMCHTrack_as<TMUONS>();
    auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFTS>();

    bool isPaired = IsMatchableMCH(mchTrack.globalIndex(), matchablePairs);
    bool isMuon = IsMuon(muonTrack, muonTracks, mftTracks);
    int decayRanking = GetDecayRanking(mchTrack, mftTracks);

    MuonMatchType result{kMatchTypeUndefined};
    if (isPaired) {
      if (isMuon) {
        result = (ranking == 1) ? kMatchTypeTrueLeading : kMatchTypeTrueNonLeading;
      } else {
        result = (ranking == 1) ? kMatchTypeWrongLeading : kMatchTypeWrongNonLeading;
      }
    } else if (decayRanking == 2) {
      result = (ranking == 1) ? kMatchTypeDecayLeading : kMatchTypeDecayNonLeading;
    } else {
      result = (ranking == 1) ? kMatchTypeFakeLeading : kMatchTypeFakeNonLeading;
    }

    if (result == kMatchTypeUndefined) {
    std::cout << std::format("[GetMatchType] isPaired={} isMuon={} decayRanking={} result={}",
        isPaired, isMuon, decayRanking, static_cast<int>(result)) << std::endl;
    }

    return result;
  }

  // for each MCH standalone track, collect the associated matching candidates
  template <class TMUON, class C>
  void GetSelectedMuons(const CollisionInfo& collisionInfo,
                        C const& collisions,
                        TMUON const& muonTracks,
                        std::vector<int64_t>& selectedMuons)
  {
    selectedMuons.clear();
    for (auto muonTrack : muonTracks) {

      // only consider MCH-MID matches
      if (static_cast<int>(muonTrack.trackType()) != 3) {
        continue;
      }

      // only select MCH-MID tracks from the current collision
      if (!muonTrack.has_collision())
        continue;
      if (static_cast<int64_t>(muonTrack.collisionId()) != collisionInfo.index)
        continue;

      const auto& collision = collisions.rawIteratorAt(muonTrack.collisionId());

      // select MCH tracks with strict quality cuts
      if (!IsGoodMuon(muonTrack, collision,
                      fMuonTaggingTrackChi2MchUp,
                      fMuonTaggingPMchLow,
                      fMuonTaggingPtMchLow,
                      {fMuonTaggingEtaMchLow, fMuonTaggingEtaMchUp},
                      {fMuonTaggingRabsLow, fMuonTaggingRabsUp},
                      fMuonTaggingSigmaPdcaUp)) {
        continue;
      }
      // std::cout << "[TOTO1] " << (int)muonTrack.globalIndex() << "  " << (int)muonTrack.trackType() << "  " << muonTrack.chi2MatchMCHMFT() << std::endl;

      // auto const& mchTrack = muonTrack.template matchMCHTrack_as<TMUON>();
      // int64_t mchTrackIndex = mchTrack.globalIndex();
      // std::cout << "[TOTO1] MCH " << (int)mchTrack.globalIndex() << "  " << (int)mchTrack.trackType() << std::endl;

      // propagate MCH track to the vertex
      auto mchTrackAtVertex = VarManager::PropagateMuon(muonTrack, collision, VarManager::kToVertex);

      // propagate the track from the vertex to the first MFT plane
      const auto& extrapToMFTfirst = PropagateToZMCH(mchTrackAtVertex, o2::mft::constants::mft::LayerZCoordinate()[0]);
      double rFront = std::sqrt(extrapToMFTfirst.getX() * extrapToMFTfirst.getX() + extrapToMFTfirst.getY() * extrapToMFTfirst.getY());
      double rMinFront = 3.f;
      double rMaxFront = 9.f;
      if (rFront < rMinFront || rFront > rMaxFront)
        continue;

      // propagate the track from the vertex to the last MFT plane
      const auto& extrapToMFTlast = PropagateToZMCH(mchTrackAtVertex, o2::mft::constants::mft::LayerZCoordinate()[9]);
      double rBack = std::sqrt(extrapToMFTlast.getX() * extrapToMFTlast.getX() + extrapToMFTlast.getY() * extrapToMFTlast.getY());
      double rMinBack = 5.f;
      double rMaxBack = 12.f;
      if (rBack < rMinBack || rBack > rMaxBack)
        continue;

      /*
      // propagate the extrapolated track to each of the MFT planes
      int nCrossedPlanes = 0;
      for (auto zPlaneMFT : o2::mft::constants::mft::LayerZCoordinate()) {
        const auto& extrapToMFT = PropagateToZMCH(mchTrackAtVertex, zPlaneMFT);
        if (true) {
          nCrossedPlanes += 1;
        }
      }
      // check that the track crosses enough MFT planes
      if (nCrossedPlanes < fMuonTaggingNCrossedMftPlanesLow) {
        continue;
      }
      */

      int64_t muonTrackIndex = muonTrack.globalIndex();
      selectedMuons.emplace_back(muonTrackIndex);
    }
  }

  // for each MCH standalone track, collect the associated matching candidates
  template <class TMUON>
  void GetTaggedMuons(const CollisionInfo& collisionInfo,
                      TMUON const& muonTracks,
                      const std::vector<int64_t>& selectedMuons,
                      std::vector<int64_t>& taggedMuons)
  {
    taggedMuons.clear();
    for (auto [mchIndex, globalTracksVector] : collisionInfo.matchingCandidates) {

      // check if the current muon is selected
      if (std::find(selectedMuons.begin(), selectedMuons.end(), mchIndex) == selectedMuons.end())
        continue;

      // if there is only one candidate, mark the muon as select
      if (globalTracksVector.size() == 1) {
        taggedMuons.emplace_back(mchIndex);
        continue;
      }

      auto const& muonTrack0 = muonTracks.rawIteratorAt(globalTracksVector[0].globalTrackId);
      auto const& muonTrack1 = muonTracks.rawIteratorAt(globalTracksVector[1].globalTrackId);

      double chi2diff = muonTrack1.chi2MatchMCHMFT() - muonTrack0.chi2MatchMCHMFT();
      if (chi2diff < fMuonTaggingChi2DiffLow)
        continue;

      //auto mchMotherParticles = GetMotherParticles(muonTrack0);
      //std::cout << "  MCH particles: ";
      //for (const auto& mother : mchMotherParticles) {
      //  std::cout << std::format("[{}, {}] ", mother.first, mother.second);
      //}
      //std::cout << std::endl;

      taggedMuons.emplace_back(mchIndex);
    }
  }

  void GetMuonPairs(const CollisionInfo& collisionInfo,
                    std::vector<MuonPair>& muonPairs,
                    std::vector<GlobalMuonPair>& globalMuonPairs)
  {
    // outer loop over muon tracks
    for (auto mchIndex1 : collisionInfo.mchTracks) {

      // inner loop over muon tracks
      for (auto mchIndex2 : collisionInfo.mchTracks) {
        // avoid double-counting of muon pairs
        if (mchIndex2 <= mchIndex1) continue;

        MuonPair muonPair{{collisionInfo.index, mchIndex1}, {collisionInfo.index, mchIndex2}};
        muonPairs.emplace_back(muonPair);
      }
    }

    // outer loop over global muon tracks
    for (auto& [mchIndex1, matchingCandidates1] : collisionInfo.matchingCandidates) {

      // inner loop over global muon tracks
      for (auto& [mchIndex2, matchingCandidates2] : collisionInfo.matchingCandidates) {
        // avoid double-counting of muon pairs
        if (mchIndex2 <= mchIndex1) continue;

        GlobalMuonPair muonPair{{collisionInfo.index, matchingCandidates1}, {collisionInfo.index, matchingCandidates2}};
        globalMuonPairs.emplace_back(muonPair);
      }
    }
  }

  double GetMuMuInvariantMass(const o2::mch::TrackParam& track1, const o2::mch::TrackParam& track2)
  {
    ROOT::Math::PxPyPzMVector muon1{
      track1.px(),
      track1.py(),
      track1.pz(),
      o2::constants::physics::MassMuon};

    ROOT::Math::PxPyPzMVector muon2{
      track2.px(),
      track2.py(),
      track2.pz(),
      o2::constants::physics::MassMuon};

    auto dimuon = muon1 + muon2;

    // std::cout << std::format("[TOTO] P1=({:0.2f} [{:0.2f},{:0.2f},{:0.2f}])  P2=({:0.2f} [{:0.2f},{:0.2f},{:0.2f}])  M={:0.2f}",
    //     track1.getP(), track1.getPx(), track1.getPy(), track1.getPz(),
    //     track2.getP(), track2.getPx(), track2.getPy(), track2.getPz(),
    //     dimuon.M()) << std::endl;

    return dimuon.M();
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

    // std::cout << std::format("[TOTO] P1=({:0.2f} [{:0.2f},{:0.2f},{:0.2f}])  P2=({:0.2f} [{:0.2f},{:0.2f},{:0.2f}])  M={:0.2f}",
    //     track1.getP(), track1.getPx(), track1.getPy(), track1.getPz(),
    //     track2.getP(), track2.getPx(), track2.getPy(), track2.getPz(),
    //     dimuon.M()) << std::endl;

    return dimuon.M();
  }

  template <class EVT, class BC, class TMUON, class TMFT>
  void FillCollisions(EVT const& collisions,
                      BC const& bcs,
                      TMUON const& muonTracks,
                      TMFT const& mftTracks,
                      CollisionInfos& collisionInfos)
  {
    collisionInfos.clear();

    std::vector<int64_t> collisionIds;
    for (const auto& collision : collisions) {
      collisionIds.push_back(collision.globalIndex());
    }

    for(size_t cid = 1; cid < collisionIds.size()-1; cid++) {
      const auto& collision = collisions.rawIteratorAt(collisionIds[cid]);
      int64_t collisionIndex = collision.globalIndex();
      const auto& collisionPrev = collisions.rawIteratorAt(collisionIds[cid-1]);
      const auto& collisionNext = collisions.rawIteratorAt(collisionIds[cid+1]);
      auto bc = bcs.rawIteratorAt(collision.bcId());
      auto bcPrev = bcs.rawIteratorAt(collisionPrev.bcId());
      auto bcNext = bcs.rawIteratorAt(collisionNext.bcId());
      int64_t deltaPrev = bc.globalBC() - bcPrev.globalBC();
      int64_t deltaNext = bcNext.globalBC() - bc.globalBC();
      //std::cout << std::format("Collision #{}  bc={}  deltas={},{}", cid, bc.globalBC(), deltaPrev, deltaNext) << std::endl;

      //if (deltaPrev < 50 || deltaNext < 50) {
      //  continue;
      //}

      // fill collision information for global muon tracks (MFT-MCH-MID matches)
      for (auto muonTrack : muonTracks) {
        if (!muonTrack.has_collision())
          continue;

        //auto collision = collisions.rawIteratorAt(muonTrack.collisionId());
        //int64_t collisionIndex = collision.globalIndex();
        //auto bc = bcs.rawIteratorAt(collision.bcId());

        if (collisionIndex != muonTrack.collisionId()) {
          continue;
        }

        // std::cout << std::format("Collision indexes: {} / {}", collisionIndex, muonTrack.collisionId()) << std::endl;

        auto& collisionInfo = collisionInfos[collisionIndex];
        collisionInfo.index = collisionIndex;
        collisionInfo.bc = bc.globalBC();
        collisionInfo.zVertex = collision.posZ();

        if (collisionInfo.matchablePairs.empty()) {
          FillMatchablePairs(collisionInfo, muonTracks, mftTracks);
        }

        if (static_cast<int>(muonTrack.trackType()) > 2) {
          // standalone MCH or MCH-MID tracks
          int64_t mchTrackIndex = muonTrack.globalIndex();
          // if (!IsGoodMuon(muonTrack, collision)) continue;
          collisionInfo.mchTracks.push_back(mchTrackIndex);
        } else {
          // global muon tracks (MFT-MCH or MFT-MCH-MID)
          int64_t muonTrackIndex = muonTrack.globalIndex();
          double matchChi2 = muonTrack.chi2MatchMCHMFT() / 5.f;
          double matchScore = chi2ToScore(muonTrack.chi2MatchMCHMFT(), 5, 50.f);
          auto const& mchTrack = muonTrack.template matchMCHTrack_as<TMUON>();
          int64_t mchTrackIndex = mchTrack.globalIndex();
          auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();
          int64_t mftTrackIndex = mftTrack.globalIndex();

          // check if a vector of global muon candidates is already available for the current MCH index
          // if not, initialize a new one and add the current global muon track
          // bool globalMuonTrackFound = false;
          auto matchingCandidateIterator = collisionInfo.matchingCandidates.find(mchTrackIndex);
          if (matchingCandidateIterator != collisionInfo.matchingCandidates.end()) {
            //matchingCandidateIterator->second.push_back(std::make_pair(muonTrackIndex, matchScore));
            matchingCandidateIterator->second.emplace_back(MatchingCandidate{
              collisionIndex,
              muonTrackIndex,
              mchTrackIndex,
              mftTrackIndex,
              matchScore,
              matchChi2,
              -1,
              matchScore,
              matchChi2,
              -1,
              kMatchTypeUndefined
            });
            // globalMuonTrackFound = true;
          } else {
            //collisionInfo.matchingCandidates[mchTrackIndex].push_back(std::make_pair(muonTrackIndex, matchScore));
            collisionInfo.matchingCandidates[mchTrackIndex].emplace_back(MatchingCandidate{
              collisionIndex,
              muonTrackIndex,
              mchTrackIndex,
              mftTrackIndex,
              matchScore,
              matchChi2,
              -1,
              matchScore,
              matchChi2,
              -1,
              kMatchTypeUndefined
            });
          }
        }
      }

      // fill collision information for MFT standalone tracks
      for (auto mftTrack : mftTracks) {
        if (!mftTrack.has_collision())
          continue;

        //auto collision = collisions.rawIteratorAt(mftTrack.collisionId());
        //int64_t collisionIndex = collision.globalIndex();
        //auto bc = bcs.rawIteratorAt(collision.bcId());

        if (collisionIndex != mftTrack.collisionId()) {
          continue;
        }

        int64_t mftTrackIndex = mftTrack.globalIndex();

        auto& collisionInfo = collisionInfos[collisionIndex];
        collisionInfo.index = collisionIndex;
        collisionInfo.bc = bc.globalBC();
        collisionInfo.zVertex = collision.posZ();

        collisionInfo.mftTracks.push_back(mftTrackIndex);
      }
    }

    // sort the vectors of matching candidates in ascending order based on the matching score value
    auto compareMatchingScore = [](const MatchingCandidate& track1, const MatchingCandidate& track2) -> bool {
      return (track1.matchScore > track2.matchScore);
    };

    for (auto& [collisionIndex, collisionInfo] : collisionInfos) {
      for (auto& [mchIndex, globalTracksVector] : collisionInfo.matchingCandidates) {
        std::sort(globalTracksVector.begin(), globalTracksVector.end(), compareMatchingScore);

        int ranking = 1;
        for (auto& candidate : globalTracksVector) {
          const auto& muonTrack = muonTracks.rawIteratorAt(candidate.globalTrackId);

          candidate.matchRanking = ranking;
          candidate.matchRankingProd = ranking;
          candidate.matchType = GetMatchType(muonTrack, muonTracks, mftTracks, collisionInfo.matchablePairs, ranking);
          ranking += 1;
        }
      }
    }
  }

  template <class C, class TMUON, class TMFT>
  void FillMatchingPlotsMC(C const& collision,
                           const CollisionInfo& collisionInfo,
                           TMUON const& muonTracks,
                           TMFT const& mftTracks,
                           const MatchingCandidates& matchingCandidates,
                           const MatchingCandidates& matchingCandidatesProd,
                           const std::vector<std::pair<int64_t, int64_t>>& matchablePairs,
                           double matchingScoreCut,
                           MatchingPlotter* plotter,
                           bool verbose=false)
  {
    int mftTrackMult = collisionInfo.mftTracks.size();

    // ====================================
    // Matching candidates hierarchy

    for (const auto& [mchIndex, globalTracksVector] : matchingCandidates) {
      // check if the MCH track belongs to a matchable pair
      bool isPairedMCH = IsMatchableMCH(static_cast<int64_t>(mchIndex), matchablePairs);

      // get the standalone MCH track
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
      double mchMom = mchTrack.p();
      double mchPt = mchTrack.pt();

      // MCH track quality flag
      bool isGoodMCH = IsGoodGlobalMuon(mchTrack, collision);

      auto matchablePair = GetMatchablePairForMCH(static_cast<int64_t>(mchIndex), matchablePairs);
      bool hasMatchablePair = matchablePair.has_value();
      bool isGoodPair = false;
      bool isMuon = false;
      int decayRanking = 0;
      int mftTrackType = -1;
      float dchi2 = (globalTracksVector.size() >= 2) ?
          globalTracksVector[1].matchChi2 - globalTracksVector[0].matchChi2 :
          -1;
      float mftDQ = 0;
      if (hasMatchablePair) {
        const auto& mchMcParticle = mchTrack.mcParticle();
        int64_t mchMcTrackIndex = mchMcParticle.globalIndex();

        auto const& pairedMftTrack = mftTracks.rawIteratorAt(matchablePair.value().second);
        auto mftMcParticle = pairedMftTrack.mcParticle();
        int64_t mftMcTrackIndex = mftMcParticle.globalIndex();

        mftTrackType = pairedMftTrack.isCA() ? 1 : 0;

        isMuon = ((mchMcTrackIndex == mftMcTrackIndex) && (std::abs(mchMcParticle.pdgCode()) == 13));

        isGoodPair = isGoodMCH && IsGoodMFT(pairedMftTrack);

        decayRanking = GetDecayRanking(mchTrack, mftTracks);

        mftDQ = (GetParticleCharge(pairedMftTrack.mcParticle()) - GetTrackCharge(pairedMftTrack)) / 2;
      }

      // std::cout << std::format("Checking matchable MCH track #{}", mchIndex) << std::endl;

      // find the index of the matching candidate that corresponds to the true match
      // index=1 corresponds to the leading candidate
      // index=0 means no candidate was found that corresponds to the true match
      int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, matchablePairs);
      int trueMatchIndexProd = GetTrueMatchIndex(muonTracks, matchingCandidatesProd.at(mchIndex), matchablePairs);

      float mcParticleDz = -1000;
      if (mchTrack.has_mcParticle()) {
        const auto& mchMcParticle = mchTrack.mcParticle();
        //float mchMcParticleZ = mchMcParticle.vz();
        mcParticleDz = collision.posZ() - mchMcParticle.vz();
      }

      std::get<std::shared_ptr<TH1>>(plotter->fMatchRanking->hist)->Fill(trueMatchIndex);
      std::get<std::shared_ptr<TH2>>(plotter->fMatchRanking->histVsP)->Fill(mchMom, trueMatchIndex);
      std::get<std::shared_ptr<TH2>>(plotter->fMatchRanking->histVsPt)->Fill(mchPt, trueMatchIndex);
      std::get<std::shared_ptr<TH2>>(plotter->fMatchRanking->histVsMcParticleDz)->Fill(mcParticleDz, trueMatchIndex);
      std::get<std::shared_ptr<TH2>>(plotter->fMatchRanking->histVsMftTrackMult)->Fill(mftTrackMult, trueMatchIndex);
      std::get<std::shared_ptr<TH2>>(plotter->fMatchRanking->histVsMftTrackType)->Fill(mftTrackType, trueMatchIndex);
      std::get<std::shared_ptr<TH2>>(plotter->fMatchRanking->histVsProdRanking)->Fill(trueMatchIndexProd, trueMatchIndex);
      if (dchi2 >= 0) std::get<std::shared_ptr<TH2>>(plotter->fMatchRanking->histVsDeltaChi2)->Fill(dchi2, trueMatchIndex);
      //std::get<std::shared_ptr<TH2>>(plotter->fMatchRanking->histVsMftDQ)->Fill(mftDQ, trueMatchIndex);

      if (isGoodMCH) {
        std::get<std::shared_ptr<TH1>>(plotter->fMatchRankingGoodMCH->hist)->Fill(trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingGoodMCH->histVsP)->Fill(mchMom, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingGoodMCH->histVsPt)->Fill(mchPt, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingGoodMCH->histVsMcParticleDz)->Fill(mcParticleDz, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingGoodMCH->histVsMftTrackMult)->Fill(mftTrackMult, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingGoodMCH->histVsMftTrackType)->Fill(mftTrackType, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingGoodMCH->histVsProdRanking)->Fill(trueMatchIndexProd, trueMatchIndex);
        if (dchi2 >= 0) std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingGoodMCH->histVsDeltaChi2)->Fill(dchi2, trueMatchIndex);
        //std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingGoodMCH->histVsMftDQ)->Fill(mftDQ, trueMatchIndex);
     }

      if (isPairedMCH) {
        std::get<std::shared_ptr<TH1>>(plotter->fMatchRankingPaired->hist)->Fill(trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPaired->histVsP)->Fill(mchMom, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPaired->histVsPt)->Fill(mchPt, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPaired->histVsMcParticleDz)->Fill(mcParticleDz, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPaired->histVsMftTrackMult)->Fill(mftTrackMult, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPaired->histVsMftTrackType)->Fill(mftTrackType, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPaired->histVsProdRanking)->Fill(trueMatchIndexProd, trueMatchIndex);
        if (dchi2 >= 0) std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPaired->histVsDeltaChi2)->Fill(dchi2, trueMatchIndex);
        //std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPaired->histVsMftDQ)->Fill(mftDQ, trueMatchIndex);
      }
      if (decayRanking > 1) {
        std::get<std::shared_ptr<TH1>>(plotter->fMatchRankingDecay->hist)->Fill(trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecay->histVsP)->Fill(mchMom, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecay->histVsPt)->Fill(mchPt, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecay->histVsMcParticleDz)->Fill(mcParticleDz, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecay->histVsMftTrackMult)->Fill(mftTrackMult, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecay->histVsMftTrackType)->Fill(mftTrackType, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecay->histVsProdRanking)->Fill(trueMatchIndexProd, trueMatchIndex);
        if (dchi2 >= 0) std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecay->histVsDeltaChi2)->Fill(dchi2, trueMatchIndex);
        //std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecay->histVsMftDQ)->Fill(mftDQ, trueMatchIndex);
      }

      if (isGoodMCH && isPairedMCH) {
        std::get<std::shared_ptr<TH1>>(plotter->fMatchRankingPairedGoodMCH->hist)->Fill(trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPairedGoodMCH->histVsP)->Fill(mchMom, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPairedGoodMCH->histVsPt)->Fill(mchPt, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPairedGoodMCH->histVsMcParticleDz)->Fill(mcParticleDz, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPairedGoodMCH->histVsMftTrackMult)->Fill(mftTrackMult, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPairedGoodMCH->histVsMftTrackType)->Fill(mftTrackType, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPairedGoodMCH->histVsProdRanking)->Fill(trueMatchIndexProd, trueMatchIndex);
        if (dchi2 >= 0) std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPairedGoodMCH->histVsDeltaChi2)->Fill(dchi2, trueMatchIndex);
        //std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingPairedGoodMCH->histVsMftDQ)->Fill(mftDQ, trueMatchIndex);
      }

      if (isGoodMCH && decayRanking > 1) {
        std::get<std::shared_ptr<TH1>>(plotter->fMatchRankingDecayGoodMCH->hist)->Fill(trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecayGoodMCH->histVsP)->Fill(mchMom, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecayGoodMCH->histVsPt)->Fill(mchPt, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecayGoodMCH->histVsMcParticleDz)->Fill(mcParticleDz, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecayGoodMCH->histVsMftTrackMult)->Fill(mftTrackMult, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecayGoodMCH->histVsMftTrackType)->Fill(mftTrackType, trueMatchIndex);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecayGoodMCH->histVsProdRanking)->Fill(trueMatchIndexProd, trueMatchIndex);
        if (dchi2 >= 0) std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecayGoodMCH->histVsDeltaChi2)->Fill(dchi2, trueMatchIndex);
        //std::get<std::shared_ptr<TH2>>(plotter->fMatchRankingDecayGoodMCH->histVsMftDQ)->Fill(mftDQ, trueMatchIndex);
      }

      if (trueMatchIndex == 0) {
        // missed matches
        if (!isPairedMCH) {
          // the MCH track does not have a corresponding MFT track for matching
          if (mchTrack.has_mcParticle()) {
            // the MCH track is not fake
            std::get<std::shared_ptr<TH1>>(plotter->fMissedMatches)->Fill(0);
            if (isGoodMCH) {
              std::get<std::shared_ptr<TH1>>(plotter->fMissedMatchesGoodMCH)->Fill(0);
            }
          } else {
            // the MCH track is fake
            std::get<std::shared_ptr<TH1>>(plotter->fMissedMatches)->Fill(1);
            if (isGoodMCH) {
              std::get<std::shared_ptr<TH1>>(plotter->fMissedMatchesGoodMCH)->Fill(1);
            }
          }
        } else {
          // the correct match is not among the stored candidates
          std::get<std::shared_ptr<TH1>>(plotter->fMissedMatches)->Fill(2);
          if (isGoodMCH) {
            std::get<std::shared_ptr<TH1>>(plotter->fMissedMatchesGoodMCH)->Fill(2);
          }
          if (isGoodPair) {
            std::get<std::shared_ptr<TH1>>(plotter->fMissedMatchesGoodMCHMFT)->Fill(2);
          }
        }
      }

      if (globalTracksVector.size() > 1 && trueMatchIndex > 0) {
        // we have a matchable pair with at least two candidates, so we can check the score difference
        // between the leading and the sub-leading
        auto leadingScore = globalTracksVector[0].matchScore;
        auto subleadingScore = globalTracksVector[1].matchScore;
        if (trueMatchIndex == 1) {
          std::get<std::shared_ptr<TH1>>(plotter->fScoreGapLeadingTrueMatches)->Fill(leadingScore - subleadingScore);
        } else {
          std::get<std::shared_ptr<TH1>>(plotter->fScoreGapNonLeadingTrueMatches)->Fill(leadingScore - subleadingScore);
        }
      }

      if (trueMatchIndex == 0) {
        // missed matches
        std::get<std::shared_ptr<TH1>>(plotter->fDecayRankingMissedMatches)->Fill(decayRanking);
      } else if (trueMatchIndex == 1) {
        // good matches
        std::get<std::shared_ptr<TH1>>(plotter->fDecayRankingGoodMatches)->Fill(decayRanking);
      } else {
        // non-leading matches
        std::get<std::shared_ptr<TH1>>(plotter->fDecayRankingNonLeadingMatches)->Fill(decayRanking);
      }
    }

    // ====================================
    // Matching properties

    for (auto [mchIndex, globalTracksVector] : matchingCandidates) {
      if (globalTracksVector.size() < 1)
        continue;

      // get the standalone MCH track
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

      // check if the MCH track belongs to a matchable pair
      bool isPairedMCH = IsMatchableMCH(static_cast<int64_t>(mchIndex), matchablePairs);

      // get leading matching candidate
      auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[0].globalTrackId);

      auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();

      // skip global muon tracks that do not pass the MCH and MFT quality cuts
      if (!IsGoodGlobalMuon(mchTrack, collision))
        continue;


      //int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, matchablePairs);
      //double matchingScore = (trueMatchIndex > 0) ? globalTracksVector[trueMatchIndex - 1].matchScore : globalTracksVector[0].matchScore;
      //double leadingMatchingScore = globalTracksVector[0].matchScore;
      double mchMom = mchTrack.p();
      double mchPt = mchTrack.pt();
      //auto leadingMatchType = globalTracksVector[0].matchType;

      // check the matching quality, but set the minimum matching score to zero
      //bool isGoodMatchNoScore = IsGoodGlobalMatching(muonTrack, matchingScore, 0);
      // bool isGoodMatch = IsGoodGlobalMatching(muonTrack, matchingScore, matchingScoreCut);
      //bool isTrueMatch = IsTrueGlobalMatching(muonTrack, matchablePairs);

      int decayRanking = GetDecayRanking(mchTrack, mftTracks);
      bool isMuon = (decayRanking == 1) ?
          ((mchTrack.mcParticle().globalIndex() == mftTrack.mcParticle().globalIndex()) &&
           (std::abs(mchTrack.mcParticle().pdgCode()) == 13)) :
          false;


      // matching score analysis
      for (const auto& candidate : globalTracksVector) {
        int matchType = static_cast<int>(candidate.matchType);
        std::get<std::shared_ptr<TH1>>(plotter->fMatchType)->Fill(matchType);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchTypeVsP)->Fill(mchMom, matchType);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchTypeVsPt)->Fill(mchPt, matchType);

        std::get<std::shared_ptr<TH2>>(plotter->fMatchScoreVsChi2)->Fill(candidate.matchChi2, candidate.matchScore);

        std::get<std::shared_ptr<TH2>>(plotter->fMatchChi2VsType)->Fill(matchType, candidate.matchChi2);
        std::get<std::shared_ptr<TH3>>(plotter->fMatchChi2VsTypeVsP)->Fill(mchMom, matchType, candidate.matchChi2);
        std::get<std::shared_ptr<TH3>>(plotter->fMatchChi2VsTypeVsPt)->Fill(mchPt, matchType, candidate.matchChi2);

        std::get<std::shared_ptr<TH2>>(plotter->fMatchScoreVsType)->Fill(matchType, candidate.matchScore);
        std::get<std::shared_ptr<TH3>>(plotter->fMatchScoreVsTypeVsP)->Fill(mchMom, matchType, candidate.matchScore);
        std::get<std::shared_ptr<TH3>>(plotter->fMatchScoreVsTypeVsPt)->Fill(mchPt, matchType, candidate.matchScore);
      }
      /*
      // score of the leading match among the candidates
      std::get<std::shared_ptr<TH1>>(plotter->fLeadingMatchScore)->Fill(leadingMatchingScore);
      std::get<std::shared_ptr<TH2>>(plotter->fLeadingMatchScoreVsP)->Fill(mchMom, leadingMatchingScore);
      std::get<std::shared_ptr<TH2>>(plotter->fLeadingMatchScoreVsPt)->Fill(mchPt, leadingMatchingScore);
      if (trueMatchIndex > 0) {
        // score of the true match among the candidates
        std::get<std::shared_ptr<TH1>>(plotter->fTrueMatchScore)->Fill(matchingScore);
        std::get<std::shared_ptr<TH2>>(plotter->fTrueMatchScoreVsP)->Fill(mchMom, matchingScore);
        std::get<std::shared_ptr<TH2>>(plotter->fTrueMatchScoreVsPt)->Fill(mchPt, matchingScore);
        if (isMuon) {
          std::get<std::shared_ptr<TH1>>(plotter->fTrueMatchScoreMuon)->Fill(matchingScore);
          std::get<std::shared_ptr<TH2>>(plotter->fTrueMatchScoreMuonVsP)->Fill(mchMom, matchingScore);
          std::get<std::shared_ptr<TH2>>(plotter->fTrueMatchScoreMuonVsPt)->Fill(mchPt, matchingScore);
        } else if (decayRanking > 1) {
          std::get<std::shared_ptr<TH1>>(plotter->fTrueMatchScoreDecay)->Fill(matchingScore);
          std::get<std::shared_ptr<TH2>>(plotter->fTrueMatchScoreDecayVsP)->Fill(mchMom, matchingScore);
          std::get<std::shared_ptr<TH2>>(plotter->fTrueMatchScoreDecayVsPt)->Fill(mchPt, matchingScore);
        }
      } else if (!isPairedMCH) {
        // score of the leading candidates if the match is fake,
        // i.e. the MCH track does not have any MFT couterpart
        std::get<std::shared_ptr<TH1>>(plotter->fFakeMatchScore)->Fill(matchingScore);
        std::get<std::shared_ptr<TH2>>(plotter->fFakeMatchScoreVsP)->Fill(mchMom, matchingScore);
        std::get<std::shared_ptr<TH2>>(plotter->fFakeMatchScoreVsPt)->Fill(mchPt, matchingScore);
      }*/
    }

    for (auto [mchIndex, globalTracksVector] : matchingCandidates) {
      if (globalTracksVector.size() < 1)
        continue;

      // get the standalone MCH track
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

      int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, matchablePairs);

      // loop over candidates
      int candidateIndex = 1;
      for (const auto& candidate : globalTracksVector) {
        auto const& muonTrack = muonTracks.rawIteratorAt(candidate.globalTrackId);

        float matchScore = candidate.matchScore;
        float matchChi2 = candidate.matchChi2;

        float matchChi2Prod = muonTrack.chi2MatchMCHMFT() / 5.f;
        float matchScoreProd = chi2ToScore(muonTrack.chi2MatchMCHMFT(), 5, 50.f);

        std::get<std::shared_ptr<TH2>>(plotter->fMatchScoreVsProd)->Fill(matchScoreProd, matchScore);
        std::get<std::shared_ptr<TH2>>(plotter->fMatchChi2VsProd)->Fill(matchChi2Prod, matchChi2);

        if (candidateIndex == trueMatchIndex) {
          std::get<std::shared_ptr<TH2>>(plotter->fTrueMatchScoreVsProd)->Fill(matchScoreProd, matchScore);
          std::get<std::shared_ptr<TH2>>(plotter->fTrueMatchChi2VsProd)->Fill(matchChi2Prod, matchChi2);
        }

        candidateIndex += 1;
      }
    }

    // ====================================
    // Matching purity
    if (verbose) std::cout << std::format("  Filling matching purity plots with score cut {}", matchingScoreCut) << std::endl;
    for (auto [mchIndex, globalTracksVector] : matchingCandidates) {
      if (globalTracksVector.size() < 1)
        continue;

      // get the leading matching candidate
      auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[0].globalTrackId);
      double matchingScore = globalTracksVector[0].matchScore;

      // get the standalone MCH and MFT tracks
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
      auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();

      // skip global muon tracks that do not pass the MCH and MFT quality cuts
      if (!IsGoodGlobalMuon(mchTrack, collision))
        continue;
      if (!IsGoodMFT(mftTrack))
        continue;

      // skip  candidates that do not pass the matching quality cuts
      if (!IsGoodGlobalMatching(muonTrack, matchingScore, matchingScoreCut))
        continue;

      // check if the matching candidate is a true one
      bool isTrueMatch = IsTrueGlobalMatching(muonTrack, matchablePairs);

      if (verbose) std::cout << std::format("    MCH track #{} -> Muon track #{}, isTrueMatch={}", mchIndex, globalTracksVector[0].globalTrackId, isTrueMatch) << std::endl;
      // fill matching purity plots
      plotter->fMatchingPurityPlotter.Fill(mchTrack, isTrueMatch);
    }

    // ====================================
    // Matching efficiencies

    // outer loop on matchable pairs
    for (auto [matchableMchIndex, matchableMftIndex] : matchablePairs) {
      // get the standalone MCH track
      // std::cout << std::format("Retrieving paired tracks: {} / {}", matchableMchIndex, matchableMftIndex) << std::endl;
      auto const& mchTrack = muonTracks.rawIteratorAt(matchableMchIndex);
      auto const& pairedMftTrack = mftTracks.rawIteratorAt(matchableMftIndex);
      // std::cout << std::format("... done.") << std::endl;

      // skip  track pairs that do not pass the MCH and MFT quality cuts
      // we only consider matchable pairs that fulfill the track quality requirements
      // std::cout << std::format("Checking tracks...") << std::endl;
      if (!IsGoodGlobalMuon(mchTrack, collision))
        continue;
      // std::cout << std::format("... MCH ok.") << std::endl;
      //if (!IsGoodMFT(pairedMftTrack)) continue;
      // std::cout << std::format("... MFT ok.") << std::endl;

      bool goodMatchFound = false;
      bool isTrueMatch = false;
      bool isMuon = IsMuon(mchTrack, pairedMftTrack);

      // check if we have some matching candidates for the current matchable MCH track
      if (matchingCandidates.count(matchableMchIndex) > 0) {
        // std::cout << std::format("Getting matching candidates for MCH track {}", matchableMchIndex) << std::endl;
        const auto& globalTracksVector = matchingCandidates.at(static_cast<int64_t>(matchableMchIndex));
        // std::cout << std::format("Number of matching candidates: {}", globalTracksVector.size()) << std::endl;
        if (!globalTracksVector.empty()) {
          // get the leading matching candidate
          // std::cout << std::format("Getting leading matching candidate: {}", globalTracksVector[0].globalTrackId) << std::endl;
          auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[0].globalTrackId);
          double matchingScore = globalTracksVector[0].matchScore;
          // std::cout << std::format("... done.") << std::endl;

          // get the standalone MFT track
          // std::cout << std::format("Getting MFT track") << std::endl;
          auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();
          auto mftIndex = mftTrack.globalIndex();
          // std::cout << std::format("... done.") << std::endl;

          // a good match must pass the MFT and matching quality cuts
          // the MCH track quality is already checked in the outer loop
          // std::cout << std::format("Checking matched track quality") << std::endl;
          //goodMatchFound = IsGoodMFT(mftTrack) && IsGoodGlobalMatching(muonTrack, matchingScore, matchingScoreCut);
          goodMatchFound = IsGoodGlobalMatching(muonTrack, matchingScore, matchingScoreCut);
          isTrueMatch = (mftIndex == matchableMftIndex);
          // std::cout << std::format("... done: {}", goodMatchFound) << std::endl;
        }
      }

      // fill matching efficiency plots
      plotter->fPairingEfficiencyPlotter.Fill(mchTrack, goodMatchFound);
      plotter->fMatchingEfficiencyPlotter.Fill(mchTrack, (goodMatchFound && isTrueMatch));
      plotter->fFakeMatchingEfficiencyPlotter.Fill(mchTrack, (goodMatchFound && !isTrueMatch));
      if (isMuon) {
        plotter->fMatchingEfficiencyMuonPlotter.Fill(mchTrack, (goodMatchFound && isTrueMatch));
      }
    }
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void FillResolutionPlotsMC(C const& collision,
                            TMUON const& muonTracks,
                            TMFT const& mftTracks,
                            CMFT const& mftCovs,
                            std::string label,
                            const MatchingCandidates& matchingCandidates,
                            const std::vector<std::pair<int64_t, int64_t>>& matchablePairs,
                            TrackResolutionHistos* histos)
  {
    // ====================================
    // Matching candidates hierarchy

    // extrapolation parameters
    auto matchingPlaneZ = matchingPlanesZ[label];
    if (matchingPlaneZ >= 0) {
      return;
    }
    auto extrapMethod = matchingExtrapMethod[label];

    for (const auto& [mchIndex, globalTracksVector] : matchingCandidates) {
      // check if the MCH track belongs to a matchable pair
      bool isPairedMCH = IsMatchableMCH(static_cast<int64_t>(mchIndex), matchablePairs);
      if (!isPairedMCH) {
        continue;
      }

      if (globalTracksVector.size() < 2) {
        continue;
      }

      // get the standalone MCH track
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
      double mchMom = mchTrack.p();

      // MCH track quality flag
      bool isGoodMCH = IsGoodGlobalMuon(mchTrack, collision);
      if (!isGoodMCH) {
        continue;
      }

      int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, matchablePairs);
      if (trueMatchIndex < 1) {
        continue;
      }
      auto matchablePair = GetMatchablePairForMCH(static_cast<int64_t>(mchIndex), matchablePairs);
      if (!matchablePair.has_value()) {
        continue;
      }

      // get muon track corresponding to the true match
      auto const& muonTrack = muonTracks.rawIteratorAt(globalTracksVector[trueMatchIndex - 1].globalTrackId);

      // get MFT standalone track
      auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();

      // only consider muons that cross both MFT and MCH
      if (std::abs(mftTrack.mcParticle().pdgCode()) != 13) {
        continue;
      }
      if (std::abs(mchTrack.mcParticle().pdgCode()) != 13) {
        continue;
      }
      int decayRanking = GetDecayRanking(mchTrack, mftTracks);
      if (decayRanking != 1) {
        continue;
      }

      // get muon track corresponding to the highest ranked wrong match
      auto const& muonTrackWrongMatch = (trueMatchIndex == 1) ? muonTracks.rawIteratorAt(globalTracksVector[1].globalTrackId) : muonTracks.rawIteratorAt(globalTracksVector[0].globalTrackId);

      // get MFT track covariance parameters
      if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
        continue;
      }
      auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

      // get MFT standalone track and covariance parameters
      auto const& mftTrackWrongMatch = muonTrackWrongMatch.template matchMFTTrack_as<TMFT>();
      if (mftTrackCovs.count(mftTrackWrongMatch.globalIndex()) < 1) {
        continue;
      }
      auto const& mftTrackCovWrongMatch = mftCovs.rawIteratorAt(mftTrackCovs[mftTrackWrongMatch.globalIndex()]);

      // get tracks parameters in O2 format
      auto mftTrackProp = PropagateToMatchingPlaneMFT(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZ, extrapMethod);
      auto mftTrackPropWrongMatch = PropagateToMatchingPlaneMFT(mchTrack, mftTrackWrongMatch, mftTrackCovWrongMatch, collision, matchingPlaneZ, extrapMethod);
      auto mchTrackProp = PropagateToMatchingPlaneMCH(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZ, extrapMethod);
      auto particleProp = PropagateToZMCH(ParticleToTrackPar(mftTrack.mcParticle()), matchingPlaneZ);

      // MFT-MCH residuals
      float dx = mchTrackProp.getX() - mftTrackProp.getX();
      float dy = mchTrackProp.getY() - mftTrackProp.getY();
      float dphi = mchTrackProp.getPhi() - mftTrackProp.getPhi();
      float dtanl = mchTrackProp.getTanl() - mftTrackProp.getTanl();

      float ndx = dx / TMath::Sqrt(mchTrackProp.getSigma2X() + mftTrackProp.getSigma2X());
      float ndy = dy / TMath::Sqrt(mchTrackProp.getSigma2Y() + mftTrackProp.getSigma2Y());
      float ndphi = dphi / TMath::Sqrt(mchTrackProp.getSigma2Phi() + mftTrackProp.getSigma2Phi());
      float ndtanl = dtanl / TMath::Sqrt(mchTrackProp.getSigma2Tanl() + mftTrackProp.getSigma2Tanl());

      float mftDx = mftTrackPropWrongMatch.getX() - mftTrackProp.getX();
      float mftDy = mftTrackPropWrongMatch.getY() - mftTrackProp.getY();
      float mftDr = std::sqrt(mftDx * mftDx + mftDy * mftDy);
      float mftTheta = 90.f + std::atanf(mftTrackProp.getTanl()) * 180.f / TMath::Pi();
      float mftThetaWrongMatch = 90.f + std::atanf(mftTrackPropWrongMatch.getTanl()) * 180.f / TMath::Pi();
      float mftDtheta = std::fabs(mftThetaWrongMatch - mftTheta);
      //std::cout << std::format("tanl MFT:   {}  {}", mftTrackProp.getTanl(), mftTrackPropWrongMatch.getTanl()) << std::endl;
      //std::cout << std::format("lambda MFT: {}  {}", -1.f * std::atanf(mftTrackProp.getTanl()) * 180.f / TMath::Pi(), -1.f * std::atanf(mftTrackProp.getTanl()) * 180.f / TMath::Pi()) << std::endl;
      //std::cout << std::format("theta MFT:  {}  {}  {}", mftTheta, mftThetaWrongMatch, mftDtheta) << std::endl;
      float mftSigmax = TMath::Sqrt(mftTrackProp.getSigma2X());
      float mftSigmay = TMath::Sqrt(mftTrackProp.getSigma2Y());
      float mftSigmaPhi = TMath::Sqrt(mftTrackProp.getSigma2Phi()) * 180.f / TMath::Pi();
      float mftSigmaTanl = TMath::Sqrt(mftTrackProp.getSigma2Tanl());
      float mftSigmaxWrongMatch = TMath::Sqrt(mftTrackProp.getSigma2X());
      float mftSigmayWrongMatch = TMath::Sqrt(mftTrackProp.getSigma2Y());
      float mftSigmaPhiWrongMatch = TMath::Sqrt(mftTrackProp.getSigma2Phi()) * 180.f / TMath::Pi();
      float mftSigmaTanlWrongMatch = TMath::Sqrt(mftTrackProp.getSigma2Tanl());

      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDx_directParticle)->Fill(mchMom, dx);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDy_directParticle)->Fill(mchMom, dy);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDsx_directParticle)->Fill(mchMom, dx);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDsy_directParticle)->Fill(mchMom, dx);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDphi_directParticle)->Fill(mchMom, dphi);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDtanl_directParticle)->Fill(mchMom, dtanl);

      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackNDx_directParticle)->Fill(mchMom, ndx);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackNDy_directParticle)->Fill(mchMom, ndy);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackNDsx_directParticle)->Fill(mchMom, ndx);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackNDsy_directParticle)->Fill(mchMom, ndx);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackNDphi_directParticle)->Fill(mchMom, ndphi);
      std::get<std::shared_ptr<TH2>>(histos->mftMchTrackNDtanl_directParticle)->Fill(mchMom, ndtanl);

      if (trueMatchIndex == 1) {
        std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDx_goodRanking)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDy_goodRanking)->Fill(mchMom, dy);
        std::get<std::shared_ptr<TH2>>(histos->mftTrackDr_goodRanking)->Fill(mchMom, mftDr);
        std::get<std::shared_ptr<TH2>>(histos->mftTrackDtheta_goodRanking)->Fill(mchMom, mftDtheta);
        std::get<std::shared_ptr<TH2>>(histos->mftTrackSigmax_goodRanking)->Fill(mchMom, mftSigmax);
        std::get<std::shared_ptr<TH2>>(histos->mftTrackSigmay_goodRanking)->Fill(mchMom, mftSigmay);
        std::get<std::shared_ptr<TH2>>(histos->mftTrackSigmaPhi_goodRanking)->Fill(mchMom, mftSigmaPhi);
        std::get<std::shared_ptr<TH2>>(histos->mftTrackSigmaTanl_goodRanking)->Fill(mchMom, mftSigmaTanl);
      } else {
        std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDx_badRanking)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDy_badRanking)->Fill(mchMom, dy);
        if (trueMatchIndex == 2) {
          std::get<std::shared_ptr<TH2>>(histos->mftTrackDr_badRanking)->Fill(mchMom, mftDr);
          std::get<std::shared_ptr<TH2>>(histos->mftTrackDtheta_badRanking)->Fill(mchMom, mftDtheta);
          std::get<std::shared_ptr<TH2>>(histos->mftTrackSigmax_badRanking)->Fill(mchMom, mftSigmaxWrongMatch);
          std::get<std::shared_ptr<TH2>>(histos->mftTrackSigmay_badRanking)->Fill(mchMom, mftSigmayWrongMatch);
          std::get<std::shared_ptr<TH2>>(histos->mftTrackSigmaPhi_badRanking)->Fill(mchMom, mftSigmaPhiWrongMatch);
          std::get<std::shared_ptr<TH2>>(histos->mftTrackSigmaTanl_badRanking)->Fill(mchMom, mftSigmaTanlWrongMatch);
        }
      }
      if (mftTrack.isCA()) {
        std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDx_ca)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDy_ca)->Fill(mchMom, dy);
      } else {
        std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDx_kalman)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->mftMchTrackDy_kalman)->Fill(mchMom, dy);
      }

      // Particle-MFT residuals
      dx = mftTrackProp.getX() - particleProp.getX();
      dy = mftTrackProp.getY() - particleProp.getY();
      if (trueMatchIndex == 1) {
        int quadrant = GetQuadrant(mftTrack);
        int pn = GetParticleCharge(mftTrack.mcParticle()) > 0 ? 0 : 1;
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDx_goodRanking)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDy_goodRanking)->Fill(mchMom, dy);
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDx_goodRanking_Qpn[quadrant][pn])->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDy_goodRanking_Qpn[quadrant][pn])->Fill(mchMom, dy);
      } else {
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDx_badRanking)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDy_badRanking)->Fill(mchMom, dy);
      }
      if (mftTrack.isCA()) {
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDx_ca)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDy_ca)->Fill(mchMom, dy);
      } else {
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDx_kalman)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->partMftTrackDy_kalman)->Fill(mchMom, dy);
      }

      // Particle-MCH residuals
      dx = mchTrackProp.getX() - particleProp.getX();
      dy = mchTrackProp.getY() - particleProp.getY();
      if (trueMatchIndex == 1) {
        std::get<std::shared_ptr<TH2>>(histos->partMchTrackDx_goodRanking)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->partMchTrackDy_goodRanking)->Fill(mchMom, dy);
      } else {
        std::get<std::shared_ptr<TH2>>(histos->partMchTrackDx_badRanking)->Fill(mchMom, dx);
        std::get<std::shared_ptr<TH2>>(histos->partMchTrackDy_badRanking)->Fill(mchMom, dy);
      }

      // momentum resolution
      float mchParticleX = mchTrack.mcParticle().vx();
      float mchParticleY = mchTrack.mcParticle().vy();
      float mchParticleZ = mchTrack.mcParticle().vz();
      float mchParticleP = mchTrack.mcParticle().p();
      auto mchTrackPropAtMcParticle = PropagateToVertexMCH(FwdToTrackPar(mchTrack, mchTrack), mchParticleX, mchParticleY, mchParticleZ, 0, 0);
      float dp = mchTrackPropAtMcParticle.getP() - mchTrack.mcParticle().p();
      std::get<std::shared_ptr<TH2>>(histos->mchDpVsP)->Fill(mchParticleP, dp);
      std::get<std::shared_ptr<TH2>>(histos->mchDpOverPVsP)->Fill(mchParticleP, dp / mchParticleP);

      if (std::abs(mftTrack.mcParticle().pdgCode()) == 13) {
        float mftParticleX = mftTrack.mcParticle().vx();
        float mftParticleY = mftTrack.mcParticle().vy();
        float mftParticleZ = mftTrack.mcParticle().vz();
        float mftParticleP = mftTrack.mcParticle().p();
        float mftDp = mftTrack.p() - mftTrack.mcParticle().p();
        float mftDQ = (GetParticleCharge(mftTrack.mcParticle()) - GetTrackCharge(mftTrack)) / 2;
        std::get<std::shared_ptr<TH2>>(histos->mftDpVsP)->Fill(mftParticleP, mftDp);
        std::get<std::shared_ptr<TH2>>(histos->mftDpOverPVsP)->Fill(mftParticleP, mftDp / mftParticleP);
        //std::get<std::shared_ptr<TH2>>(histos->mftDpOverSigmaPVsP)->Fill(mftParticleP, mftDp / mftParticleP);
        std::get<std::shared_ptr<TH2>>(histos->mftDQVsP)->Fill(mftParticleP, mftDQ);
        if (false && mftDQ < 0) {
          auto mftMotherParticles = GetMotherParticles(mftTrack);
          std::cout << std::format("  MFT track.signed1Pt(): {:0.3f}  Q={}", mftTrack.signed1Pt(), GetTrackCharge(mftTrack)) << std::endl;
          std::cout << std::format("  MFT particle: p={:0.3f} Q={}", mftTrack.mcParticle().p(), GetParticleCharge(mftTrack.mcParticle())) << std::endl;
          std::cout << "  MFT particles: ";
          for (const auto& mother : mftMotherParticles) {
            std::cout << std::format("[{}, {}] ", mother.first, mother.second);
          }
          std::cout << std::endl;
        }
      }
    }
  }

  template <class C, class TMUON, class TMFT>
  void FillDimuonPlotsMC(const CollisionInfo& collisionInfo,
                         C const& collisions,
                         TMUON const& muonTracks,
                         TMFT const& mftTracks,
                         //const MatchingCandidates& matchingCandidates,
                         const std::vector<std::pair<int64_t, int64_t>>& matchablePairs)
  {
    std::vector<MuonPair> muonPairs;
    std::vector<GlobalMuonPair> globalMuonPairs;

    GetMuonPairs(collisionInfo, muonPairs, globalMuonPairs);

    // check if a given MCH track belongs to a muon particle crossing also the MFT
    auto mchIsMuon = [&](TMUON::iterator const& mchTrack, TMFT const& mftTracks) -> bool
    {
      if (static_cast<int>(mchTrack.trackType()) < 2)
        return false;

      // skip tracks that do not have an associated MC particle
      if (!mchTrack.has_mcParticle()) return false;
      auto mchMcParticle = mchTrack.mcParticle();
      if (std::abs(mchMcParticle.pdgCode()) != 13) return false;

      // loop on MFT tracks to find the one belonging to the same MC muon
      for (const auto& mftTrack : mftTracks) {
        if (!mftTrack.has_mcParticle()) return false;

        auto mftMcParticle = mftTrack.mcParticle();
        if (mchMcParticle.globalIndex() == mftMcParticle.globalIndex()) return true;
      }

      return false;
    };

    for (auto& [muon1, muon2] : muonPairs) {
      auto collisionIndex = muon1.first;
      auto const& collision = collisions.rawIteratorAt(muon1.first);

      auto mchIndex1 = muon1.second;
      auto mchIndex2 = muon2.second;
      auto const& muonTrack1 = muonTracks.rawIteratorAt(mchIndex1);
      auto const& muonTrack2 = muonTracks.rawIteratorAt(mchIndex2);
      int sign1 = muonTrack1.sign();
      int sign2 = muonTrack2.sign();

      // only consider opposite-sign pairs
      if ((sign1 * sign2) >= 0) continue;

      bool goodMuonTracks = (IsGoodMuon(muonTrack1, collision) && IsGoodMuon(muonTrack2, collision));
      bool goodGlobalMuonTracks = (IsGoodGlobalMuon(muonTrack1, collision) && IsGoodGlobalMuon(muonTrack2, collision));

      bool isMuon1 = mchIsMuon(muonTrack1, mftTracks);
      bool isMuon2 = mchIsMuon(muonTrack2, mftTracks);

      if (goodMuonTracks) {
        double mass = GetMuMuInvariantMass(PropagateToVertexMCH(muonTrack1, collision),
                                           PropagateToVertexMCH(muonTrack2, collision));
        registryDimuon.get<TH1>(HIST("dimuon/invariantMass_MuonKine_MuonCuts"))->Fill(mass);
      }
    }

    for (auto& [muon1, muon2] : globalMuonPairs) {
      auto& candidates1 = muon1.second;
      auto& candidates2 = muon2.second;

      auto const& collision = collisions.rawIteratorAt(muon1.first);

      auto const& muonTrack1 = muonTracks.rawIteratorAt(candidates1[0].globalTrackId);
      auto const& muonTrack2 = muonTracks.rawIteratorAt(candidates2[0].globalTrackId);
      auto matchScore1 = candidates1[0].matchScore;
      auto matchScore2 = candidates2[0].matchScore;
      auto const& mchTrack1 = muonTrack1.template matchMCHTrack_as<TMUON>();
      auto const& mchTrack2 = muonTrack2.template matchMCHTrack_as<TMUON>();
      auto const& mftTrack1 = muonTrack1.template matchMFTTrack_as<TMFT>();
      auto const& mftTrack2 = muonTrack2.template matchMFTTrack_as<TMFT>();
      int sign1 = mchTrack1.sign();
      int sign2 = mchTrack2.sign();

      // only consider opposite-sign pairs
      if ((sign1 * sign2) >= 0) continue;

      std::cout << std::format("[TOTO] match types: {}, {}",
          static_cast<int>(candidates1[0].matchType),
          static_cast<int>(candidates2[0].matchType)) << std::endl;

      double p1 = mchTrack1.p();
      double p2 = mchTrack2.p();
      int matchType = -1;
      if (p1 >= p2) {
        matchType = candidates1[0].matchType * 10 + candidates2[0].matchType;
      } else {
        matchType = candidates2[0].matchType * 10 + candidates1[0].matchType;
      }

      //bool isMuon1 = mchIsMuon(mchTrack1, mftTracks);
      //bool isMuon2 = mchIsMuon(mchTrack2, mftTracks);
      // only consider MCH tracks that belong to a muon particle crossing also the MFT
      //if (!isMuon1 || !isMuon2) continue;

      //bool trueMatch1 = IsTrueGlobalMatching(muonTrack1, matchablePairs);
      //bool trueMatch2 = IsTrueGlobalMatching(muonTrack2, matchablePairs);

      //std::cout << std::format("Match scores: {:0.2f} {:0.2f}", matchScore1, matchScore2) << std::endl;

      bool goodGlobalMuonTracks = (IsGoodGlobalMuon(mchTrack1, collision) && IsGoodGlobalMuon(mchTrack2, collision));
      if (!goodGlobalMuonTracks) {
        continue;
      }

      bool goodGlobalMuonMatches = (IsGoodGlobalMatching(muonTrack1, matchScore1) && IsGoodGlobalMatching(muonTrack2, matchScore2));
      //std::cout << std::format("goodGlobalMuonTracks: {}", goodGlobalMuonTracks) << std::endl;
      //std::cout << std::format("goodGlobalMuonMatches: {}", goodGlobalMuonMatches) << std::endl;

      double massMCH = GetMuMuInvariantMass(PropagateToVertexMCH(mchTrack1, collision),
          PropagateToVertexMCH(mchTrack2, collision));
      double mass = GetMuMuInvariantMass(PropagateToVertexMCH(muonTrack1, collision),
          PropagateToVertexMCH(muonTrack2, collision));
      registryDimuon.get<TH1>(HIST("dimuon/invariantMass_MuonKine_GlobalMuonCuts"))->Fill(massMCH);
      registryDimuon.get<TH1>(HIST("dimuon/invariantMass_ScaledMftKine_GlobalMuonCuts"))->Fill(mass);
      registryDimuon.get<TH2>(HIST("dimuon/invariantMass_MuonKine_GlobalMuonCuts_vs_match_type"))->Fill(massMCH, matchType);
      registryDimuon.get<TH2>(HIST("dimuon/invariantMass_ScaledMftKine_GlobalMuonCuts_vs_match_type"))->Fill(mass, matchType);

      if (goodGlobalMuonMatches) {
        registryDimuon.get<TH1>(HIST("dimuon/invariantMass_MuonKine_GlobalMuonCuts_GoodMatches"))->Fill(massMCH);
        registryDimuon.get<TH1>(HIST("dimuon/invariantMass_ScaledMftKine_GlobalMuonCuts_GoodMatches"))->Fill(mass);
        registryDimuon.get<TH2>(HIST("dimuon/invariantMass_MuonKine_GlobalMuonCuts_GoodMatches_vs_match_type"))->Fill(massMCH, matchType);
        registryDimuon.get<TH2>(HIST("dimuon/invariantMass_ScaledMftKine_GlobalMuonCuts_GoodMatches_vs_match_type"))->Fill(mass, matchType);
      }
    }
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void PrintChi2Matching(C const& collisions,
      TMUON const& muonTracks,
      TMFT const& /*mftTracks*/,
      CMFT const& mftCovs,
      std::string funcName,
      float matchingPlaneZ,
      int extrapMethod,
      const std::vector<std::pair<int64_t, int64_t>>& matchablePairs,
      int64_t mchIndex,
      int64_t muonIndex,
      std::ostream& os = std::cout)
  {
    if (mMatchingFunctionMap.count(funcName) < 1)
      return;
    auto matchingFunc = mMatchingFunctionMap.at(funcName);

    auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

    auto const& muonTrack = muonTracks.rawIteratorAt(muonIndex);
    if (!muonTrack.has_collision())
      return;

    auto collision = collisions.rawIteratorAt(muonTrack.collisionId());

    // get MCH and MFT standalone tracks
    // auto mchTrack = muonTrack.template matchMCHTrack_as<TMUON>();
    auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();
    if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
      // os << std::format("Covariance matrix for MFT track #{} not found", mftTrack.globalIndex()) << std::endl;
      return;
    }
    auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

    // get tracks parameters in O2 format
    auto mftTrackProp = FwdToTrackPar(mftTrack, mftTrackCov);
    auto mchTrackProp = FwdToTrackPar(mchTrack, mchTrack);

    if (matchingPlaneZ < 0.) {
      mftTrackProp = PropagateToMatchingPlaneMFT(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZ, extrapMethod);
      mchTrackProp = PropagateToMatchingPlaneMCH(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZ, extrapMethod);
    }

    os << std::format("Matching MUON track #{} and MCH track #{} at z={} with method {} and function \"{}\"",
        muonIndex, mchIndex, matchingPlaneZ, extrapMethod, funcName) << std::endl;

    auto mchMotherParticles = GetMotherParticles(mchTrack);
    os << "  MCH particles: ";
    for (const auto& mother : mchMotherParticles) {
      os << std::format("[{}, {}] ", mother.first, mother.second);
    }
    os << std::endl;
    auto mftMotherParticles = GetMotherParticles(mftTrack);
    os << "  MFT particles: ";
    for (const auto& mother : mftMotherParticles) {
      os << std::format("[{}, {}] ", mother.first, mother.second);
    }
    os << std::endl;
    os << std::format("  collision:                  z={:0.3f} x={:0.3f} y={:0.3f}", collision.posZ(), collision.posX(), collision.posY()) << std::endl;
    if (mftTrack.has_mcParticle()) {
      auto mcParticle = mftTrack.mcParticle();
      os << std::format("  MFT particle vertex:        x={:0.3f} y={:0.3f} z={:0.3f}", mcParticle.vx(), mcParticle.vy(), mcParticle.vz()) << std::endl;
    }
    os << std::format("  MFT track quality:          nClus={} chi2={:0.3f} ca={}", mftTrack.nClusters(), mftTrack.chi2(), mftTrack.isCA()) << std::endl;
    os << std::format("  MFT track first point:      x={:0.3f} y={:0.3f} z={:0.3f}", mftTrack.x(), mftTrack.y(), mftTrack.z()) << std::endl;
    // positions
    os << std::format("  MFT track position:         x={:0.3f} +/- {:0.3f} y={:0.3f} +/- {:0.3f}", mftTrackProp.getX(), std::sqrt(mftTrackProp.getSigma2X()), mftTrackProp.getY(), std::sqrt(mftTrackProp.getSigma2Y())) << std::endl;
    os << std::format("  MCH track position:         x={:0.3f} +/- {:0.3f} y={:0.3f} +/- {:0.3f}", mchTrackProp.getX(), std::sqrt(mchTrackProp.getSigma2X()), mchTrackProp.getY(), std::sqrt(mchTrackProp.getSigma2Y())) << std::endl;
    double sigmaX = TMath::Sqrt(mftTrackProp.getSigma2X() + mchTrackProp.getSigma2X());
    double sigmaY = TMath::Sqrt(mftTrackProp.getSigma2Y() + mchTrackProp.getSigma2Y());
    os << std::format("  MCH-MFT difference:         dx={:0.3f} +/- {:0.3f} dy={:0.3f} +/- {:0.3f}",
        mchTrackProp.getX() - mftTrackProp.getX(), sigmaX, mchTrackProp.getY() - mftTrackProp.getY(), sigmaY) << std::endl << std::endl;
    // phi and tangent(lambda)
    os << std::format("  MFT track position:         phi={:0.3f} +/- {:0.3f} tanl={:0.3f} +/- {:0.3f}",
        mftTrackProp.getPhi(), std::sqrt(mftTrackProp.getSigma2Phi()), mftTrackProp.getTanl(), std::sqrt(mftTrackProp.getSigma2Tanl())) << std::endl;
    os << std::format("  MCH track position:         phi={:0.3f} +/- {:0.3f} tanl={:0.3f} +/- {:0.3f}",
        mchTrackProp.getPhi(), std::sqrt(mchTrackProp.getSigma2Phi()), mchTrackProp.getTanl(), std::sqrt(mchTrackProp.getSigma2Tanl())) << std::endl;
    double sigmaPhi = TMath::Sqrt(mftTrackProp.getSigma2Phi() + mchTrackProp.getSigma2Phi());
    double sigmaTanl = TMath::Sqrt(mftTrackProp.getSigma2Tanl() + mchTrackProp.getSigma2Tanl());
    os << std::format("  MCH-MFT difference:         dphi={:0.3f} +/- {:0.3f} dtanl={:0.3f} +/- {:0.3f}",
        mchTrackProp.getPhi() - mftTrackProp.getPhi(), sigmaPhi, mchTrackProp.getTanl() - mftTrackProp.getTanl(), sigmaTanl) << std::endl << std::endl;
    // slopes
    auto mftTrackPropMCH = mExtrap.FwdtoMCH(mftTrackProp);
    auto mchTrackPropMCH = mExtrap.FwdtoMCH(mchTrackProp);
    os << std::format("  MFT track slope:            sx={:0.3f} +/- {:0.3f} sy={:0.3f} +/- {:0.3f}",
        mftTrackPropMCH.getNonBendingSlope(), std::sqrt(mftTrackPropMCH.getCovariances()(1, 1)), mftTrackPropMCH.getBendingSlope(), std::sqrt(mftTrackPropMCH.getCovariances()(3, 3))) << std::endl;
    os << std::format("  MCH track slope:            sx={:0.3f} +/- {:0.3f} sy={:0.3f} +/- {:0.3f}",
        mchTrackPropMCH.getNonBendingSlope(), std::sqrt(mchTrackPropMCH.getCovariances()(1, 1)), mchTrackPropMCH.getBendingSlope(), std::sqrt(mchTrackPropMCH.getCovariances()(3, 3))) << std::endl;
    double sigmaSX = TMath::Sqrt(mftTrackPropMCH.getCovariances()(1, 1) + mchTrackPropMCH.getCovariances()(1, 1));
    double sigmaSY = TMath::Sqrt(mftTrackPropMCH.getCovariances()(3, 3) + mchTrackPropMCH.getCovariances()(3, 3));
    os << std::format("  MCH-MFT difference:         dsx={:0.3f} +/- {:0.3f} dsy={:0.3f} +/- {:0.3f}",
        mchTrackPropMCH.getNonBendingSlope() - mftTrackPropMCH.getNonBendingSlope(), sigmaSX,
        mchTrackPropMCH.getBendingSlope() - mftTrackPropMCH.getBendingSlope(), sigmaSY) << std::endl << std::endl;
    // momentum
    os << std::format("  MFT track momentum:         p={:0.3f} Q/pT={:0.3f} +/- {:0.3f}",
        mftTrackProp.getP(), mftTrackProp.getInvQPt(), std::sqrt(mftTrackProp.getSigma2InvQPt())) << std::endl;
    os << std::format("  MCH track momentum:         p={:0.3f} Q/pT={:0.3f} +/- {:0.3f}",
        mchTrackProp.getP(), mchTrackProp.getInvQPt(), std::sqrt(mchTrackProp.getSigma2InvQPt())) << std::endl;

    // run the chi2 matching function
    auto matchingChi2 = matchingFunc(mchTrackProp, mftTrackProp);
    //float matchingScore = chi2ToScore(matchingChi2);
    os << std::endl;
    os << std::format("  Matching chi2: {}, original chi2: {}", std::get<0>(matchingChi2) / std::get<1>(matchingChi2), muonTrack.chi2MatchMCHMFT() / 5.f) << std::endl;
    os << std::endl;
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void CompareChi2MatchingWithProd(C const& collisions,
                                   TMUON const& muonTracks,
                                   TMFT const& mftTracks,
                                   CMFT const& mftCovs,
                                   std::string funcName,
                                   float matchingPlaneZ,
                                   int extrapMethod,
                                   const std::vector<std::pair<int64_t, int64_t>>& matchablePairs,
                                   int64_t mchIndex,
                                   const std::vector<MatchingCandidate>& globalTracksVector)
  {
    //auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

    int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, matchablePairs);
    std::cout << std::endl
        << std::format("Index of true matching candidate: {}", trueMatchIndex) << std::endl;

    int candidateId = 0;
    for (const auto& candidate : globalTracksVector) {
      candidateId += 1;

      if (candidateId == trueMatchIndex)
        std::cout << "===========================================" << std::endl;
      std::cout << std::format("Matching candidate #{} with default method", candidateId) << std::endl;
      PrintChi2Matching(collisions, muonTracks, mftTracks, mftCovs, "matchALL", static_cast<float>(o2::mft::constants::mft::LayerZCoordinate()[9]), 0, matchablePairs, mchIndex, candidate.globalTrackId);

      std::cout << std::format("Matching candidate #{} with custom method", candidateId) << std::endl;
      PrintChi2Matching(collisions, muonTracks, mftTracks, mftCovs, funcName, matchingPlaneZ, extrapMethod, matchablePairs, mchIndex, candidate.globalTrackId);
      if (candidateId == trueMatchIndex)
        std::cout << "===========================================" << std::endl;
    }
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void RunChi2Matching(C const& collisions,
                       TMUON const& muonTracks,
                       TMFT const& mftTracks,
                       CMFT const& mftCovs,
                       std::string funcName,
                       float matchingPlaneZ,
                       int extrapMethod,
                       const std::vector<std::pair<int64_t, int64_t>>& matchablePairs,
                       const MatchingCandidates& matchingCandidates,
                       MatchingCandidates& newMatchingCandidates)
  {
    newMatchingCandidates.clear();

    if (funcName == "prod") {
      newMatchingCandidates = matchingCandidates;
      return;
    }

    if (mMatchingFunctionMap.count(funcName) < 1)
      return;
    auto matchingFunc = mMatchingFunctionMap.at(funcName);

    //std::cout << std::format("Matching method \"{}\"  z={}  alt={}", label, matchingPlaneZ, extrapMethod) << std::endl;

    // std::cout << std::format("Processing {} matches with chi2 function {}", matchingCandidates.size(), funcName) << std::endl;

    for (auto& [mchIndex, globalTracksVector] : matchingCandidates) {
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

      int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, matchablePairs);
      //std::cout << std::endl
      //          << std::format("Index of true matching candidate: {}", trueMatchIndex) << std::endl;

      int candidateId = 0;
      for (const auto& candidate : globalTracksVector) {
        candidateId += 1;
        auto const& muonTrack = muonTracks.rawIteratorAt(candidate.globalTrackId);
        if (!muonTrack.has_collision())
          continue;

        auto collision = collisions.rawIteratorAt(muonTrack.collisionId());

        // get MCH and MFT standalone tracks
        // auto mchTrack = muonTrack.template matchMCHTrack_as<TMUON>();
        auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();
        if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
          // std::cout << std::format("Covariance matrix for MFT track #{} not found", mftTrack.globalIndex()) << std::endl;
          continue;
        }
        auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

        // get tracks parameters in O2 format
        auto mftTrackProp = FwdToTrackPar(mftTrack, mftTrackCov);
        auto mchTrackProp = FwdToTrackPar(mchTrack, mchTrack);

        if (matchingPlaneZ < 0.) {
          //mftTrackProp = PropagateToZMFT(mftTrackProp, matchingPlaneZ);
          mftTrackProp = PropagateToMatchingPlaneMFT(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZ, extrapMethod);
          mchTrackProp = PropagateToMatchingPlaneMCH(mchTrack, mftTrack, mftTrackCov, collision, matchingPlaneZ, extrapMethod);
        }

        // run the chi2 matching function
        auto matchResult = matchingFunc(mchTrackProp, mftTrackProp);
        float matchChi2 = std::get<0>(matchResult) / std::get<1>(matchResult);
        float matchScore = chi2ToScore(std::get<0>(matchResult), std::get<1>(matchResult), 10.f * std::get<1>(matchResult));
        float matchChi2Prod = muonTrack.chi2MatchMCHMFT() / 5.f;
        float matchScoreProd = chi2ToScore(muonTrack.chi2MatchMCHMFT(), 5, 50.f);
        //std::cout << std::format("Matching chi2: {} / {}, original chi2: {}", matchChi2, matchChi2Alt, muonTrack.chi2MatchMCHMFT()) << std::endl;
        //if (candidateId == trueMatchIndex)
        //  std::cout << "===========================================" << std::endl;

        // check if a vector of global muon candidates is already available for the current MCH index
        // if not, initialize a new one and add the current global muon track
        auto matchingCandidateIterator = newMatchingCandidates.find(mchIndex);
        if (matchingCandidateIterator != newMatchingCandidates.end()) {
          //matchingCandidateIterator->second.push_back(std::make_pair(muonIndex, matchingScore));
          matchingCandidateIterator->second.emplace_back(MatchingCandidate{
            muonTrack.collisionId(),
            candidate.globalTrackId,
            mchIndex,
            mftTrack.globalIndex(),
            matchScore,
            matchChi2,
            -1,
            matchScoreProd,
            matchChi2Prod,
            -1,
            kMatchTypeUndefined
          });
        } else {
          //newMatchingCandidates[mchIndex].push_back(std::make_pair(muonIndex, matchingScore));
          newMatchingCandidates[mchIndex].emplace_back(MatchingCandidate{
            muonTrack.collisionId(),
            candidate.globalTrackId,
            mchIndex,
            mftTrack.globalIndex(),
            matchScore,
            matchChi2,
            -1,
            matchScoreProd,
            matchChi2Prod,
            -1,
            kMatchTypeUndefined
          });
        }
      }
    }

    // sort the vectors of matching candidates in ascending order based on the matching score value
    auto compareMatchingScore = [](const MatchingCandidate& track1, const MatchingCandidate& track2) -> bool {
      return (track1.matchScore > track2.matchScore);
    };

    for (auto& [mchIndex, globalTracksVector] : newMatchingCandidates) {
      std::sort(globalTracksVector.begin(), globalTracksVector.end(), compareMatchingScore);
    }

    if (false) {
    //if (label == "MatchALLMethod0a") {
    //if (label == "MatchXYPhiTanlMethod1b") {
      for (auto& [mchIndex, globalTracksVector] : newMatchingCandidates) {
        auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

        int trueMatchIndexProd = GetTrueMatchIndex(muonTracks, matchingCandidates.at(mchIndex), matchablePairs);
        if (trueMatchIndexProd != 1) continue;

        int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, matchablePairs);
        if (trueMatchIndex < 2) continue;

        int decayRanking = GetDecayRanking(mchTrack, mftTracks);
        if (decayRanking != 1) continue;

        CompareChi2MatchingWithProd(collisions, muonTracks, mftTracks, mftCovs,
            funcName, matchingPlaneZ, extrapMethod, matchablePairs, mchIndex, globalTracksVector);

        //DrawTracks(collisions, muonTracks, mftTracks, mftCovs,
        //    extrapMethod, matchablePairs, mchIndex, globalTracksVector);
      }
    }
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void RunChi2Matching(C const& collisions,
                       TMUON const& muonTracks,
                       TMFT const& mftTracks,
                       CMFT const& mftCovs,
                       std::string label,
                       const std::vector<std::pair<int64_t, int64_t>>& matchablePairs,
                       const MatchingCandidates& matchingCandidates,
                       MatchingCandidates& newMatchingCandidates)
  {
    newMatchingCandidates.clear();

    auto funcIter = matchingChi2Functions.find(label);
    if (funcIter == matchingChi2Functions.end())
      return;

    auto funcName = funcIter->second;

    if (funcName == "prod") {
      newMatchingCandidates = matchingCandidates;
      return;
    }

    if (mMatchingFunctionMap.count(funcName) < 1)
      return;
    //auto matchingFunc = mMatchingFunctionMap.at(funcName);

    // extrapolation parameters
    auto matchingPlaneZ = matchingPlanesZ[label];
    auto extrapMethod = matchingExtrapMethod[label];
    //std::cout << std::format("Matching method \"{}\"  z={}  alt={}", label, matchingPlaneZ, extrapMethod) << std::endl;

    // std::cout << std::format("Processing {} matches with chi2 function {}", matchingCandidates.size(), funcName) << std::endl;

    RunChi2Matching(collisions, muonTracks, mftTracks, mftCovs, funcName, matchingPlaneZ, extrapMethod, matchablePairs, matchingCandidates, newMatchingCandidates);
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void RunMLMatching(C const& collisions,
                     TMUON const& muonTracks,
                     TMFT const& /*mftTracks*/,
                     CMFT const& mftCovs,
                     std::string label,
                     const MatchingCandidates& matchingCandidates,
                     MatchingCandidates& newMatchingCandidates)
  {
    newMatchingCandidates.clear();
    auto mlIter = matchingMlResponses.find(label);
    if (mlIter == matchingMlResponses.end())
      return;

    auto& mlResponse = mlIter->second;
    for (auto& [mchIndex, globalTracksVector] : matchingCandidates) {
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
      for (const auto& candidate : globalTracksVector) {
        auto const& muonTrack = muonTracks.rawIteratorAt(candidate.globalTrackId);
        if (!muonTrack.has_collision())
          continue;

        auto collision = collisions.rawIteratorAt(muonTrack.collisionId());

        // get MFT standalone track
        auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();
        if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
          // std::cout << std::format("Covariance matrix for MFT track #{} not found", mftTrack.globalIndex()) << std::endl;
          continue;
        }
        // std::cout << fmt::format("Getting covariance matrix for MFT track #{} -> {}", mftTrack.globalIndex(), mftTrackCovs[mftTrack.globalIndex()]) << std::endl;
        auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);
        // std::cout << fmt::format("Covariance matrix for MFT track #{} retrieved", mftTrack.globalIndex()) << std::endl;

        // get tracks parameters in O2 format
        auto mftTrackProp = FwdToTrackPar(mftTrack, mftTrackCov);
        auto mchTrackProp = FwdToTrackPar(mchTrack, mchTrack);

        // extrapolate to the matching plane
        auto matchingPlaneZ = matchingPlanesZ[label];
        if (matchingPlaneZ < 0.) {
          mftTrackProp = PropagateToZMFT(mftTrackProp, matchingPlaneZ);
          mchTrackProp = PropagateToZMCH(mchTrackProp, matchingPlaneZ);
        }

        // run the ML model
        std::vector<float> output;
        //std::vector<float> inputML = mlResponse.getInputFeaturesGlob(muonTrack, mchTrackProp, mftTrackProp, collision);
        std::vector<float> inputML = mlResponse.getInputFeatures(muonTrack, mftTrack, mchTrack, mftTrackProp, mchTrackProp, collision);
        mlResponse.isSelectedMl(inputML, 0, output);
        float matchScore = output[0];
        float matchChi2Prod = muonTrack.chi2MatchMCHMFT() / 5.f;
        float matchScoreProd = chi2ToScore(muonTrack.chi2MatchMCHMFT(), 5, 50.f);
        // std::cout << std::format("Matching score: {}, Chi2: {}", matchingScore, muonTrack.chi2MatchMCHMFT()) << std::endl;

        // check if a vector of global muon candidates is already available for the current MCH index
        // if not, initialize a new one and add the current global muon track
        auto matchingCandidateIterator = newMatchingCandidates.find(mchIndex);
        if (matchingCandidateIterator != newMatchingCandidates.end()) {
          //matchingCandidateIterator->second.push_back(std::make_pair(muonIndex, matchingScore));
          matchingCandidateIterator->second.emplace_back(MatchingCandidate{
            muonTrack.collisionId(),
            candidate.globalTrackId,
            mchIndex,
            mftTrack.globalIndex(),
            matchScore,
            -1,
            -1,
            matchScoreProd,
            matchChi2Prod,
            -1,
            kMatchTypeUndefined
          });
        } else {
          //newMatchingCandidates[mchIndex].push_back(std::make_pair(muonIndex, matchingScore));
          newMatchingCandidates[mchIndex].emplace_back(MatchingCandidate{
            muonTrack.collisionId(),
            candidate.globalTrackId,
            mchIndex,
            mftTrack.globalIndex(),
            matchScore,
            -1,
            -1,
            matchScoreProd,
            matchChi2Prod,
            -1,
            kMatchTypeUndefined
          });
        }
      }
    }

    // sort the vectors of matching candidates in ascending order based on the matching score value
    auto compareMatchingScore = [](const MatchingCandidate& track1, const MatchingCandidate& track2) -> bool {
      return (track1.matchScore > track2.matchScore);
    };

    for (auto& [mchIndex, globalTracksVector] : newMatchingCandidates) {
      std::sort(globalTracksVector.begin(), globalTracksVector.end(), compareMatchingScore);
    }
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void DrawTracks(C const& collisions,
                                   TMUON const& muonTracks,
                                   TMFT const& mftTracks,
                                   CMFT const& mftCovs,
                                   int extrapMethod,
                                   const std::vector<std::pair<int64_t, int64_t>>& matchablePairs,
                                   int64_t mchIndex,
                                   const std::vector<MatchingCandidate>& globalTracksVector)
  {
    static int index = 1;

    if (index >= 100) return;

    auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

    int trueMatchIndexProd = GetTrueMatchIndex(muonTracks, globalTracksVector, matchablePairs);
    std::cout << std::endl
        << std::format("Index of true matching candidate: {}", trueMatchIndexProd) << std::endl;

    if (trueMatchIndexProd < 2) return;

    auto const& muonTrackBestMatch = muonTracks.rawIteratorAt(globalTracksVector[0].globalTrackId);
    auto const& muonTrackTrueMatch = muonTracks.rawIteratorAt(globalTracksVector[trueMatchIndexProd - 1].globalTrackId);

    if (!muonTrackBestMatch.has_collision())
      return;
    if (!muonTrackTrueMatch.has_collision())
      return;

    auto collision = collisions.rawIteratorAt(muonTrackBestMatch.collisionId());

    // get MCH and MFT standalone tracks
    auto const& mftTrackBestMatch = muonTrackBestMatch.template matchMFTTrack_as<TMFT>();
    if (mftTrackCovs.count(mftTrackBestMatch.globalIndex()) < 1) {
      return;
    }
    auto const& mftTrackCovBestMatch = mftCovs.rawIteratorAt(mftTrackCovs[mftTrackBestMatch.globalIndex()]);
    auto const& mftTrackTrueMatch = muonTrackTrueMatch.template matchMFTTrack_as<TMFT>();
    if (mftTrackCovs.count(mftTrackTrueMatch.globalIndex()) < 1) {
      return;
    }
    auto const& mftTrackCovTrueMatch = mftCovs.rawIteratorAt(mftTrackCovs[mftTrackTrueMatch.globalIndex()]);

    std::vector<float> zPlanes{-77.5, -466.0};
    std::vector<float> extrapolatonMethod{0, 0};
    //std::vector<std::string> matchingFunction{"matchXYPhiTanl", "matchXYPhiTanl"};
    //std::vector<float> extrapolatonMethod{0, 0, 1};
    std::vector<std::string> matchingFunction{"matchALL", "matchALL"};
    std::vector<float> mchTrackX;
    std::vector<float> mchTrackY;
    std::vector<float> mftTrackBestMatchX;
    std::vector<float> mftTrackBestMatchY;
    std::vector<float> mftTrackTrueMatchX;
    std::vector<float> mftTrackTrueMatchY;
    std::vector<float> matchingChi2LeadingProd;
    std::vector<float> matchingChi2Leading;
    std::vector<float> matchingChi2True;
    std::vector<int> trueMatchIndexes;

    std::ofstream tracksDump(std::format("tracks-{:02d}.txt", index));
    for (size_t zi = 0; zi < zPlanes.size(); zi++) {

      auto z = zPlanes[zi];

      MatchingCandidates matchingCandidates;
      matchingCandidates[mchIndex] = globalTracksVector;
      MatchingCandidates newMatchingCandidates;
      RunChi2Matching(collisions, muonTracks, mftTracks, mftCovs, matchingFunction[zi], z, extrapolatonMethod[zi], matchablePairs, matchingCandidates, newMatchingCandidates);

      size_t bestMatchProdIndex = 0;
      size_t trueMatchIndex = 0;

      for (size_t index = 0; index < newMatchingCandidates.at(mchIndex).size(); index++) {
        if (newMatchingCandidates.at(mchIndex)[index].globalTrackId == globalTracksVector[0].globalTrackId) {
          std::cout << std::format("Best match:     index={} score={:0.3f} chi2={:0.3f}", index + 1, newMatchingCandidates.at(mchIndex)[index].matchScore, newMatchingCandidates.at(mchIndex)[index].matchChi2) << std::endl;
          bestMatchProdIndex = index + 1;
        } else if (newMatchingCandidates.at(mchIndex)[index].globalTrackId == globalTracksVector[trueMatchIndexProd - 1].globalTrackId) {
          std::cout << std::format("True candidate: index={} score={:0.3f} chi2={:0.3f}", index + 1, newMatchingCandidates.at(mchIndex)[index].matchScore, newMatchingCandidates.at(mchIndex)[index].matchChi2) << std::endl;
          trueMatchIndex = index + 1;
        }
      }
      if (bestMatchProdIndex < 1 || trueMatchIndex < 1)
        continue;

      //leadingMatchIndexes.push_back(bestMatchProdIndex);
      trueMatchIndexes.push_back(trueMatchIndex);
      matchingChi2LeadingProd.push_back(newMatchingCandidates.at(mchIndex)[bestMatchProdIndex - 1].matchChi2);
      matchingChi2Leading.push_back(newMatchingCandidates.at(mchIndex)[0].matchChi2);
      matchingChi2True.push_back(newMatchingCandidates.at(mchIndex)[trueMatchIndex - 1].matchChi2);

    // get tracks parameters in O2 format
      auto mchTrackProp = FwdToTrackPar(mchTrack, mchTrack);
      auto mftTrackBestMatchProp = FwdToTrackPar(mftTrackBestMatch, mftTrackCovBestMatch);
      auto mftTrackTrueMatchProp = FwdToTrackPar(mftTrackTrueMatch, mftTrackCovTrueMatch);

      mchTrackProp = PropagateToMatchingPlaneMCH(mchTrack, mftTrackBestMatch, mftTrackCovBestMatch, collision, z, extrapMethod);
      mftTrackBestMatchProp = PropagateToMatchingPlaneMFT(mchTrack, mftTrackBestMatch, mftTrackCovBestMatch, collision, z, extrapMethod);
      mftTrackTrueMatchProp = PropagateToMatchingPlaneMFT(mchTrack, mftTrackTrueMatch, mftTrackCovTrueMatch, collision, z, extrapMethod);

      mchTrackX.push_back(mchTrackProp.getX());
      mchTrackY.push_back(mchTrackProp.getY());
      mftTrackBestMatchX.push_back(mftTrackBestMatchProp.getX());
      mftTrackBestMatchY.push_back(mftTrackBestMatchProp.getY());
      mftTrackTrueMatchX.push_back(mftTrackTrueMatchProp.getX());
      mftTrackTrueMatchY.push_back(mftTrackTrueMatchProp.getY());

      std::cout << std::format("Matching MUON track #{} and MCH track #{} at z={} with method {}",
          globalTracksVector[0].globalTrackId, mchIndex, z, extrapolatonMethod[zi]) << std::endl;
      PrintChi2Matching(collisions, muonTracks, mftTracks, mftCovs, matchingFunction[zi], z, extrapolatonMethod[zi], matchablePairs, mchIndex, globalTracksVector[0].globalTrackId, tracksDump);
      std::cout << std::format("Matching MUON track #{} and MCH track at z={} with method {}",
          globalTracksVector[trueMatchIndexProd - 1].globalTrackId, mchIndex, z, extrapolatonMethod[zi]) << std::endl;
      PrintChi2Matching(collisions, muonTracks, mftTracks, mftCovs, matchingFunction[zi], z, extrapolatonMethod[zi], matchablePairs, mchIndex, globalTracksVector[trueMatchIndexProd - 1].globalTrackId, tracksDump);
    }

    //leadingMatchIndexes[0] = 0;
    //trueMatchIndexes[0] = trueMatchIndex;
    //matchingChi2Leading[0] = scoreToChi2(globalTracksVector[0].matchScore);
    //matchingChi2True[0] = scoreToChi2(globalTracksVector[trueMatchIndex - 1].matchScore);

    std::ofstream tracksFile("tracks.C");
    tracksFile << "TCanvas* canvas;\n" << std::endl <<
    "float zmin = -10;" << std::endl <<
    "\nfloat zmax = 600;" << std::endl <<
    "float absMin = 90;" << std::endl <<
    "float absMax = 505;\n" << std::endl <<
    "void DrawAbsorber()" << std::endl <<
    "{" << std::endl <<
    "  Int_t n=5;" << std::endl <<
    "  Float_t x[] = {absMin, absMin, absMax, absMax, absMin};" << std::endl <<
    "  Float_t y[] = {-100.f * absMin / absMax, 100.f * absMin / absMax, 100, -100, -100.f * absMin / absMax};\n" << std::endl <<
    "  TPolyLine *polyLine = new TPolyLine(n, x, y, \"F\");" << std::endl <<
    "  polyLine->SetLineColor(4);" << std::endl <<
    "  polyLine->SetFillColor(17);" << std::endl <<
    "  polyLine->SetFillStyle(1001);" << std::endl <<
    "  polyLine->Draw(\"f\");" << std::endl <<
    "}\n" << std::endl <<
    "void tracks()" << std::endl <<
    "{" << std::endl <<
    "  canvas = new TCanvas(\"c\", \"c\", 1200, 600);" << std::endl <<
    "  canvas->Range(zmin, -150, zmax, 120);\n" << std::endl <<

    std::format("  Float_t mchTrackZV[] = {{{}, {}, {}}};",
        -1.0 * zPlanes[0], -1.0 * zPlanes[1], -1.0 * mchTrack.z()) << std::endl <<
    std::format("  Float_t mchTrackXV[] = {{{}, {}, {}}};",
        mchTrackX[0], mchTrackX[1], mchTrack.x()) << std::endl <<
    std::format("  Float_t mchTrackYV[] = {{{}, {}, {}}};",
        mchTrackY[0], mchTrackY[1], mchTrack.y()) << std::endl <<

    std::format("  Float_t mftTrackBestMatchZV[] = {{{}, {}, {}}};",
        -1.0 * mftTrackBestMatch.z(), -1.0 * zPlanes[0], -1.0 * zPlanes[1]) << std::endl <<
    std::format("  Float_t mftTrackBestMatchXV[] = {{{}, {}, {}}};",
        mftTrackBestMatch.x(), mftTrackBestMatchX[0], mftTrackBestMatchX[1]) << std::endl <<
    std::format("  Float_t mftTrackBestMatchYV[] = {{{}, {}, {}}};",
        mftTrackBestMatch.y(), mftTrackBestMatchY[0], mftTrackBestMatchY[1]) << std::endl <<

    std::format("  Float_t mftTrackTrueMatchZV[] = {{{}, {}, {}}};",
        -1.0 * mftTrackTrueMatch.z(), -1.0 * zPlanes[0], -1.0 * zPlanes[1]) << std::endl <<
    std::format("  Float_t mftTrackTrueMatchXV[] = {{{}, {}, {}}};",
        mftTrackTrueMatch.x(), mftTrackTrueMatchX[0], mftTrackTrueMatchX[1]) << std::endl <<
    std::format("  Float_t mftTrackTrueMatchYV[] = {{{}, {}, {}}};",
        mftTrackTrueMatch.y(), mftTrackTrueMatchY[0], mftTrackTrueMatchY[1]) << std::endl <<

    "  {" << std::endl <<
    "    canvas->Clear();" << std::endl <<
    "    DrawAbsorber();\n" << std::endl <<
    "    TPolyLine* l1 = new TPolyLine(3, mchTrackZV, mchTrackXV);" << std::endl <<
    "    l1->SetLineColor(kBlue);" << std::endl <<
    "    l1->SetLineWidth(1);" << std::endl <<
    "    l1->Draw();\n" << std::endl <<
    "    TPolyLine* l2 = new TPolyLine(3, mftTrackBestMatchZV, mftTrackBestMatchXV);" << std::endl <<
    "    l2->SetLineColor(kRed);" << std::endl <<
    "    l2->SetLineWidth(1);" << std::endl <<
    "    l2->Draw();\n" << std::endl <<
    "    TPolyLine* l3 = new TPolyLine(3, mftTrackTrueMatchZV, mftTrackTrueMatchXV);" << std::endl <<
    "    l3->SetLineColor(kGreen);" << std::endl <<
    "    l3->SetLineWidth(1);" << std::endl <<
    "    l3->Draw();\n" << std::endl <<
    "    TLine* matchingPlane1 = new TLine(77.5, -100, 77.5, 100);" << std::endl <<
    "    matchingPlane1->SetLineColor(kRed);" << std::endl <<
    "    matchingPlane1->SetLineStyle(kDashed);" << std::endl <<
    "    matchingPlane1->Draw();\n" << std::endl <<
    "    TLine* matchingPlane2 = new TLine(466, -100, 466, 100);" << std::endl <<
    "    matchingPlane2->SetLineColor(kRed);" << std::endl <<
    "    matchingPlane2->SetLineStyle(kDashed);" << std::endl <<
    "    matchingPlane2->Draw();\n" << std::endl <<
    std::format("    TPaveText* text = new TPaveText({} - 50, -140, {} + 200, -100);", -1.0 * zPlanes[0], -1.0 * zPlanes[0]) << std::endl <<
    std::format("    text->AddText(\"chi2 (prod best): {:0.2f}\");", matchingChi2LeadingProd[0]) << std::endl <<
    std::format("    text->AddText(\"chi2 (best): {:0.2f}\");", matchingChi2Leading[0]) << std::endl <<
    std::format("    text->AddText(\"chi2 (true):       {:0.2f}\");", matchingChi2True[0]) << std::endl <<
    std::format("    text->AddText(\"true ranking:  #{}\");", trueMatchIndexes[0]) << std::endl <<
    "    text->SetFillStyle(4000); text->SetBorderSize(0); text->SetTextColor(kRed); text->SetTextAlign(13);" << std::endl <<
    "    text->Draw();\n" << std::endl <<
    std::format("    text = new TPaveText({} - 50, -140, {} + 200, -100);", -1.0 * zPlanes[1], -1.0 * zPlanes[1]) << std::endl <<
    std::format("    text->AddText(\"chi2 (prod best): {:0.2f}\");", matchingChi2LeadingProd[1]) << std::endl <<
    std::format("    text->AddText(\"chi2 (best): {:0.2f}\");", matchingChi2Leading[1]) << std::endl <<
    std::format("    text->AddText(\"chi2 (true):       {:0.2f}\");", matchingChi2True[1]) << std::endl <<
    std::format("    text->AddText(\"true ranking:  #{}\");", trueMatchIndexes[1]) << std::endl <<
    "    text->SetFillStyle(4000); text->SetBorderSize(0); text->SetTextColor(kRed); text->SetTextAlign(13);" << std::endl <<
    "    text->Draw();\n" << std::endl <<
    std::format("    canvas->SaveAs(\"tracks-{:02d}.pdf(\");", index) << std::endl <<
    "  }" << std::endl <<
    "  {" << std::endl <<
    "    canvas->Clear();" << std::endl <<
    "    DrawAbsorber();\n" << std::endl <<
    "    TPolyLine* l1 = new TPolyLine(3, mchTrackZV, mchTrackYV);" << std::endl <<
    "    l1->SetLineColor(kBlue);" << std::endl <<
    "    l1->SetLineWidth(1);" << std::endl <<
    "    l1->Draw();\n" << std::endl <<
    "    TPolyLine* l2 = new TPolyLine(3, mftTrackBestMatchZV, mftTrackBestMatchYV);" << std::endl <<
    "    l2->SetLineColor(kRed);" << std::endl <<
    "    l2->SetLineWidth(1);" << std::endl <<
    "    l2->Draw();\n" << std::endl <<
    "    TPolyLine* l3 = new TPolyLine(3, mftTrackTrueMatchZV, mftTrackTrueMatchYV);" << std::endl <<
    "    l3->SetLineColor(kGreen);" << std::endl <<
    "    l2->SetLineWidth(1);" << std::endl <<
    "    l3->Draw();\n" << std::endl <<
    "    TLine* matchingPlane1 = new TLine(77.5, -100, 77.5, 100);" << std::endl <<
    "    matchingPlane1->SetLineColor(kRed);" << std::endl <<
    "    matchingPlane1->SetLineStyle(kDashed);" << std::endl <<
    "    matchingPlane1->Draw();\n" << std::endl <<
    "    TLine* matchingPlane2 = new TLine(466, -100, 466, 100);" << std::endl <<
    "    matchingPlane2->SetLineColor(kRed);" << std::endl <<
    "    matchingPlane2->SetLineStyle(kDashed);" << std::endl <<
    "    matchingPlane2->Draw();\n" << std::endl <<
    std::format("    TPaveText* text = new TPaveText({} - 50, -140, {} + 200, -100);", -1.0 * zPlanes[0], -1.0 * zPlanes[0]) << std::endl <<
    std::format("    text->AddText(\"leading chi2: {:0.2f}\");", 1.0f) << std::endl <<
    std::format("    text->AddText(\"true chi2:       {:0.2f}\");", 2.0f) << std::endl <<
    std::format("    text->AddText(\"true ranking:  #{}\");", trueMatchIndexes[0]) << std::endl <<
    "    text->SetFillStyle(4000); text->SetBorderSize(0); text->SetTextColor(kRed); text->SetTextAlign(13);" << std::endl <<
    "    text->Draw();\n" << std::endl <<
    std::format("    text = new TPaveText({} - 50, -140, {} + 200, -100);", -1.0 * zPlanes[1], -1.0 * zPlanes[1]) << std::endl <<
    std::format("    text->AddText(\"leading chi2: {:0.2f}\");", 1.0f) << std::endl <<
    std::format("    text->AddText(\"true chi2:       {:0.2f}\");", 2.0f) << std::endl <<
    std::format("    text->AddText(\"true ranking:  #{}\");", trueMatchIndexes[1]) << std::endl <<
    "    text->SetFillStyle(4000); text->SetBorderSize(0); text->SetTextColor(kRed); text->SetTextAlign(13);" << std::endl <<
    "    text->Draw();\n" << std::endl <<
    std::format("    canvas->SaveAs(\"tracks-{:02d}.pdf)\");", index) << std::endl <<
    "  }" << std::endl <<
    "}" << std::endl;

    system("root -l -b -q tracks.C");

    index += 1;
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void ProcessCollisionMC(const CollisionInfo& collisionInfo,
                          C const& collisions,
                          TMUON const& muonTracks,
                          TMFT const& mftTracks,
                          CMFT const& mftCovs,
                          aod::McParticles const& mcParticles)
  {
    //std::cout << std::endl << std::format("ProcessMatching() collision #{}", collisionInfo.index) << std::endl;
    auto collision = collisions.rawIteratorAt(collisionInfo.index);

    registry.get<TH1>(HIST("tracksMultiplicityMFT"))->Fill(collisionInfo.mftTracks.size());
    registry.get<TH1>(HIST("tracksMultiplicityMCH"))->Fill(collisionInfo.mchTracks.size());

    // Leading / sub-leading matching chi2 difference
    for (auto [mchIndex, globalTracksVector] : collisionInfo.matchingCandidates) {
      if (globalTracksVector.size() < 2)
        continue;
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

      // MCH track quality flag
      if (!IsGoodGlobalMuon(mchTrack, collision)) continue;

      float dchi2 = globalTracksVector[1].matchChi2 - globalTracksVector[0].matchChi2;
      registry.get<TH1>(HIST("matchCandidatesDeltaChi2"))->Fill(dchi2);
    }

    //std::vector<std::pair<int64_t, int64_t>> matchablePairs;
    //FillMatchablePairs(collisionInfo, muonTracks, mftTracks, matchablePairs);
    for (auto [mchIndex, mftIndex] : collisionInfo.matchablePairs) {
      auto const& muonTrack = muonTracks.rawIteratorAt(mchIndex);
      auto muonMcParticle = muonTrack.mcParticle();
      int64_t muonMcTrackIndex = muonMcParticle.globalIndex();
      double mchMom = muonTrack.p();

      //std::cout << std::format("TOTO1 MCH track #{} type={} p={:0.3} - MFT track #{} - collision #{} - has_collision={}",
      //     mchIndex, muonTrack.trackType(), mchMom, mftIndex, muonTrack.collisionId(), muonTrack.has_collision()) << std::endl;

      auto const& mftTrack = mftTracks.rawIteratorAt(mftIndex);
      auto mftMcParticle = mftTrack.mcParticle();
      int64_t mftMcTrackIndex = mftMcParticle.globalIndex();

      if (muonMcTrackIndex == mftMcTrackIndex) {
        registry.get<TH1>(HIST("matching/MC/pairableType"))->Fill(0);
      } else {
        registry.get<TH1>(HIST("matching/MC/pairableType"))->Fill(1);
      }

      if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
        // std::cout << std::format("Covariance matrix for MFT track #{} not found", mftTrack.globalIndex()) << std::endl;
        continue;
      }
      auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

      // propagate tracks at the last MFT plane
      auto mftTrackProp = FwdToTrackPar(mftTrack, mftTrackCov);
      auto mchTrackProp = FwdToTrackPar(muonTrack, muonTrack);
      auto mchTrackPropAlt = FwdToTrackPar(muonTrack, muonTrack);

      auto z = o2::mft::constants::mft::LayerZCoordinate()[9];
      //std::cout << std::format("Extrapolating tracks to z={}", z) << std::endl;
      mftTrackProp = PropagateToZMFT(mftTrackProp, z);
      mchTrackProp = PropagateToZMCH(mchTrackProp, z);
      auto mchTrackAtVertex = VarManager::PropagateMuon(muonTrack, collision, VarManager::kToVertex);
      mchTrackPropAlt = PropagateToZMCH(mchTrackAtVertex, z);

      float dx = mchTrackProp.getX() - mftTrackProp.getX();
      float dy = mchTrackProp.getY() - mftTrackProp.getY();
      registryAlignment.get<TH2>(HIST("resolution/MC/trackDxAtMFTVsP"))->Fill(mchMom, dx);
      registryAlignment.get<TH2>(HIST("resolution/MC/trackDyAtMFTVsP"))->Fill(mchMom, dy);

      float dxAlt = mchTrackPropAlt.getX() - mftTrackProp.getX();
      float dyAlt = mchTrackPropAlt.getY() - mftTrackProp.getY();
      registryAlignment.get<TH2>(HIST("resolution/MC/trackDxAtMFTVsP_alt"))->Fill(mchMom, dxAlt);
      registryAlignment.get<TH2>(HIST("resolution/MC/trackDyAtMFTVsP_alt"))->Fill(mchMom, dyAlt);

      //std::cout << std::format("MCH track position: x={} y={}", mchTrackProp.getX(), mchTrackProp.getY()) << std::endl;
      //std::cout << std::format("MCH track pos. alt: x={} y={}", mchTrackPropAlt.getX(), mchTrackPropAlt.getY()) << std::endl;
      //std::cout << std::format("MFT track position: x={} y={}", mftTrackProp.getX(), mftTrackProp.getY()) << std::endl;
    }

    // plot MFT-MCH tracks difference at matching plane for wrong pairs
    // outer loop on MCH tracks associated to the current collision
    for (auto const& mchTrackIndex : collisionInfo.mchTracks) {
      auto matchablePair = GetMatchablePairForMCH(mchTrackIndex, collisionInfo.matchablePairs);
      if (!matchablePair.has_value())
        continue;

      auto const& mchTrack = muonTracks.rawIteratorAt(mchTrackIndex);
      double mchMom = mchTrack.p();

      // std::cout << std::format("TOTO2 MCH track #{} type={} p={:0.3} - MFT track #{} - collision #{}",
      //     mchTrackIndex, mchTrack.trackType(), mchMom, matchablePair.value().second, mchTrack.collisionId()) << std::endl;

      // extrapolate the MCH track to the last MFT plane
      auto z = o2::mft::constants::mft::LayerZCoordinate()[9];
      auto mchTrackProp = PropagateToZMCH(FwdToTrackPar(mchTrack, mchTrack), z);
      auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToVertex);
      auto mchTrackPropAlt = PropagateToZMCH(mchTrackAtVertex, z);

      // inner loop on MFT tracks associated to the current collision
      for (auto const& mftTrackIndex : collisionInfo.mftTracks) {
        // skip true matches
        if (mftTrackIndex == matchablePair.value().second)
          continue;
        auto const& mftTrack = mftTracks.rawIteratorAt(mftTrackIndex);

        // get the corresponding covariance matrix
        if (mftTrackCovs.count(mftTrackIndex) < 1) {
          continue;
        }
        auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrackIndex]);

        auto mftTrackProp = PropagateToZMFT(FwdToTrackPar(mftTrack, mftTrackCov), z);

        float dx = mchTrackProp.getX() - mftTrackProp.getX();
        float dy = mchTrackProp.getY() - mftTrackProp.getY();
        registryAlignment.get<TH2>(HIST("resolution/MC/trackDxAtMFTVsP_fake"))->Fill(mchMom, dx);
        registryAlignment.get<TH2>(HIST("resolution/MC/trackDyAtMFTVsP_fake"))->Fill(mchMom, dy);

        float dxAlt = mchTrackPropAlt.getX() - mftTrackProp.getX();
        float dyAlt = mchTrackPropAlt.getY() - mftTrackProp.getY();
        registryAlignment.get<TH2>(HIST("resolution/MC/trackDxAtMFTVsP_alt_fake"))->Fill(mchMom, dxAlt);
        registryAlignment.get<TH2>(HIST("resolution/MC/trackDyAtMFTVsP_alt_fake"))->Fill(mchMom, dyAlt);
      }
    }

    // Chi2-based matching analysis
    //std::cout << std::endl << std::format("Filling matching plots for label Prod") << std::endl;
    FillMatchingPlotsMC(collision, collisionInfo, muonTracks, mftTracks, collisionInfo.matchingCandidates, collisionInfo.matchingCandidates, collisionInfo.matchablePairs, fMatchingChi2ScoreMftMchLow, fChi2MatchingPlotter.get(), false);
    for (auto& [label, func] : matchingChi2Functions) {
      MatchingCandidates matchingCandidates;
      RunChi2Matching(collisions, muonTracks, mftTracks, mftCovs, label, collisionInfo.matchablePairs, collisionInfo.matchingCandidates, matchingCandidates);

      auto* plotter = fMatchingPlotters.at(label).get();
      double matchingScoreCut = matchingScoreCuts.at(label);
      auto* resolutionHistos = fTrackResolutionHistos.at(label).get();

      std::cout << std::format("Calling FillMatchingPlotsMC() for label \"{}\" and score cut {}", label, matchingScoreCut) << std::endl;
      //std::cout << std::endl << std::format("Filling matching plots for label {}", label) << std::endl;
      FillMatchingPlotsMC(collision, collisionInfo, muonTracks, mftTracks, matchingCandidates, collisionInfo.matchingCandidates, collisionInfo.matchablePairs, matchingScoreCut, plotter, false);
      FillResolutionPlotsMC(collision, muonTracks, mftTracks, mftCovs, label, matchingCandidates, collisionInfo.matchablePairs, resolutionHistos);
    }

    // ML-based matching analysis
    for (auto& [label, mlResponse] : matchingMlResponses) {
      MatchingCandidates matchingCandidates;
      RunMLMatching(collisions, muonTracks, mftTracks, mftCovs, label, collisionInfo.matchingCandidates, matchingCandidates);

      auto* plotter = fMatchingPlotters.at(label).get();
      double matchingScoreCut = matchingScoreCuts.at(label);
      auto* resolutionHistos = fTrackResolutionHistos.at(label).get();

      FillMatchingPlotsMC(collision, collisionInfo, muonTracks, mftTracks, matchingCandidates, collisionInfo.matchingCandidates, collisionInfo.matchablePairs, matchingScoreCut, plotter);
      FillResolutionPlotsMC(collision, muonTracks, mftTracks, mftCovs, label, matchingCandidates, collisionInfo.matchablePairs, resolutionHistos);
    }

    // match ranking based on global track
    {
      MatchingCandidates matchingCandidates = collisionInfo.matchingCandidates;

      // sort the vectors of matching candidates in ascending order based on the matching score value
      auto compareGlobalTrackChi2 = [&](const MatchingCandidate& track1, const MatchingCandidate& track2) -> bool {
        const auto& muonTrack1 = muonTracks.rawIteratorAt(track1.globalTrackId);
        const auto& muonTrack2 = muonTracks.rawIteratorAt(track2.globalTrackId);
        float chi2_1 = GetTrackChi2OverNDF(muonTrack1, muonTracks, mftTracks);
        float chi2_2 = GetTrackChi2OverNDF(muonTrack2, muonTracks, mftTracks);
        return (chi2_1 < chi2_2);
      };

      //std::cout << std::format("matchingCandidates.size(): {}", matchingCandidates.size()) << std::endl;

      for (auto& [mchIndex, globalTracksVector] : matchingCandidates) {
        //std::cout << std::format("  globalTracksVector.size(): {}", globalTracksVector.size()) << std::endl;
        std::sort(globalTracksVector.begin(), globalTracksVector.end(), compareGlobalTrackChi2);
      }

      FillMatchingPlotsMC(collision, collisionInfo, muonTracks, mftTracks, matchingCandidates, collisionInfo.matchingCandidates, collisionInfo.matchablePairs, fMatchingChi2ScoreMftMchLow, fTrackChi2MatchingPlotter.get());
      //FillResolutionPlotsMC(collision, muonTracks, mftTracks, mftCovs, label, matchingCandidates, collisionInfo.matchablePairs, resolutionHistos);
    }

    // Matching chi2 correlation at two matching planes
    for (auto [mchIndex, globalTracksVector] : collisionInfo.matchingCandidates) {
      if (globalTracksVector.size() < 1)
        continue;

      // get MCH and MFT standalone tracks
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

      int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, collisionInfo.matchablePairs);

      int matchIndex = 1;
      for (const auto& candidate : globalTracksVector) {

        auto const& muonTrack = muonTracks.rawIteratorAt(candidate.globalTrackId);
        auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();
        if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
          continue;
        }
        auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

        // get tracks parameters in O2 format
        auto mftTrackProp1 = PropagateToMatchingPlaneMFT(mchTrack, mftTrack, mftTrackCov, collision, -77.5, 0);
        auto mchTrackProp1 = PropagateToMatchingPlaneMCH(mchTrack, mftTrack, mftTrackCov, collision, -77.5, 0);

        auto mftTrackProp2 = PropagateToMatchingPlaneMFT(mchTrack, mftTrack, mftTrackCov, collision, -466.0, 0);
        auto mchTrackProp2 = PropagateToMatchingPlaneMCH(mchTrack, mftTrack, mftTrackCov, collision, -466.0, 0);

        // run the chi2 matching function
        auto matchingFunc = mMatchingFunctionMap.at("matchXYPhiTanl");
        //auto matchingFunc = mMatchingFunctionMap.at("matchXYSxSy");
        if (matchIndex == trueMatchIndex) {
          if (IsMuon(mchTrack, mftTrack)) {
            auto matchResult_1 = matchingFunc(mchTrackProp1, mftTrackProp1);
            auto matchChi2_1 = std::get<0>(matchResult_1) / std::get<1>(matchResult_1);
            auto matchResult_2 = matchingFunc(mchTrackProp2, mftTrackProp2);
            auto matchChi2_2 = std::get<0>(matchResult_2) / std::get<1>(matchResult_2);
            registry.get<TH2>(HIST("matching/MC/matchingChi2CorrelationMuon"))->Fill(matchChi2_1, matchChi2_2);
          }
        } else {
          auto matchResult_1 = matchingFunc(mchTrackProp1, mftTrackProp1);
          auto matchChi2_1 = std::get<0>(matchResult_1) / std::get<1>(matchResult_1);
          auto matchResult_2 = matchingFunc(mchTrackProp2, mftTrackProp2);
          auto matchChi2_2 = std::get<0>(matchResult_2) / std::get<1>(matchResult_2);
          registry.get<TH2>(HIST("matching/MC/matchingChi2CorrelationFake"))->Fill(matchChi2_1, matchChi2_2);
        }
      }
    }

    // Muons tagging
    for (auto [mchIndex, mftIndex] : collisionInfo.matchablePairs) {
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
      if (!mchTrack.has_collision())
        continue;
      auto collision = collisions.rawIteratorAt(mchTrack.collisionId());

      auto const& mftTrack = mftTracks.rawIteratorAt(mftIndex);
      if (mftTrackCovs.count(mftTrack.globalIndex()) < 1) {
        continue;
      }
      auto const& mftTrackCov = mftCovs.rawIteratorAt(mftTrackCovs[mftTrack.globalIndex()]);

      auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToVertex);

      // extrapolate to the matching plane
      auto z = o2::mft::constants::mft::LayerZCoordinate()[9];
      auto mchTrackProp = PropagateToZMCH(mchTrackAtVertex, z);
      auto mftTrackProp = PropagateToZMFT(FwdToTrackPar(mftTrack, mftTrackCov), z);

      registry.get<TH2>(HIST("matching/MC/pairedMCHTracksAtMFT"))->Fill(mchTrackProp.getX(), mchTrackProp.getY());
      registry.get<TH2>(HIST("matching/MC/pairedMFTTracksAtMFT"))->Fill(mftTrackProp.getX(), mftTrackProp.getY());
    }

    std::vector<int64_t> selectedMuons;
    GetSelectedMuons(collisionInfo, collisions, muonTracks, selectedMuons);
    for (auto mchIndex : selectedMuons) {
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
      if (!mchTrack.has_collision())
        continue;
      auto collision = collisions.rawIteratorAt(mchTrack.collisionId());

      auto mchTrackAtVertex = VarManager::PropagateMuon(mchTrack, collision, VarManager::kToVertex);

      // extrapolate to the matching plane
      auto mchTrackPropMFTFront = PropagateToZMCH(mchTrackAtVertex, o2::mft::constants::mft::LayerZCoordinate()[0]);
      auto mchTrackPropMFTBack = PropagateToZMCH(mchTrackAtVertex, o2::mft::constants::mft::LayerZCoordinate()[9]);

      registry.get<TH2>(HIST("matching/MC/selectedMCHTracksAtMFTFront"))->Fill(mchTrackPropMFTFront.getX(), mchTrackPropMFTFront.getY());
      registry.get<TH2>(HIST("matching/MC/selectedMCHTracksAtMFTBack"))->Fill(mchTrackPropMFTBack.getX(), mchTrackPropMFTBack.getY());

      bool isPairedMCH = IsMatchableMCH(static_cast<int64_t>(mchIndex), collisionInfo.matchablePairs);
      if (isPairedMCH) {
        registry.get<TH2>(HIST("matching/MC/selectedMCHTracksAtMFTFrontTrue"))->Fill(mchTrackPropMFTFront.getX(), mchTrackPropMFTFront.getY());
        registry.get<TH2>(HIST("matching/MC/selectedMCHTracksAtMFTBackTrue"))->Fill(mchTrackPropMFTBack.getX(), mchTrackPropMFTBack.getY());
      } else {
        registry.get<TH2>(HIST("matching/MC/selectedMCHTracksAtMFTFrontFake"))->Fill(mchTrackPropMFTFront.getX(), mchTrackPropMFTFront.getY());
        registry.get<TH2>(HIST("matching/MC/selectedMCHTracksAtMFTBackFake"))->Fill(mchTrackPropMFTBack.getX(), mchTrackPropMFTBack.getY());
      }
    }

    MatchingCandidates selectedMatchingCandidates;
    for (auto [mchIndex, globalTracksVector] : collisionInfo.matchingCandidates) {
      if (std::find(selectedMuons.begin(), selectedMuons.end(), mchIndex) != selectedMuons.end()) {
        selectedMatchingCandidates[mchIndex] = globalTracksVector;
        continue;
        int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, collisionInfo.matchablePairs);
        std::cout << std::format("Selected MCH track: true match index = {}", trueMatchIndex) << std::endl;
        if (trueMatchIndex == 0) {
          auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
          auto mchMotherParticles = GetMotherParticles(mchTrack);
          std::cout << "  MC particles for selected MCH track: " << std::endl;
          for (const auto& mother : mchMotherParticles) {
            const auto& mcParticle = mcParticles.rawIteratorAt(mother.second);
            std::cout << std::format("{}, {}, p={:0.3}, V x={:0.2} y={:0.2} z={:0.2}",
                mother.first, mother.second, mcParticle.p(), mcParticle.vx(), mcParticle.vy(), mcParticle.vz()) << std::endl;
          }
          //std::cout << std::endl;
          /*
          int candidateId = 0;
          for (auto [mftIndex, score] : globalTracksVector) {
            candidateId += 1;
            auto const& mftTrack = mftTracks.rawIteratorAt(mftIndex);
            auto mftMotherParticles = GetMotherParticles(mftTrack);
            std::cout << std::format("  MC particles for matched MFT track #{}: ", candidateId);
            for (const auto& mother : mftMotherParticles) {
              const auto& mcParticle = mcParticles.rawIteratorAt(mother.second);
              std::cout << std::format("[{}, {}, p={}] ", mother.first, mother.second, mcParticle.p());
            }
            std::cout << std::endl;
          }
          */
          // search for an MFT track that is associated to one of the MCH mother particles
          for (const auto& mftTrack : mftTracks) {
            // skip tracks that do not have an associated MC particle
            if (!mftTrack.has_mcParticle())
              continue;
            // get the index associated to the MC particle
            auto mftMcParticle = mftTrack.mcParticle();
            int64_t mftMcTrackIndex = mftMcParticle.globalIndex();
            for (const auto& mother : mchMotherParticles) {
              if (mother.second == mftMcTrackIndex) {
                std::cout << std::format("  Parent MFT particle: [{}, {}, p={}] ", mother.first, mother.second, mftMcParticle.p()) << std::endl;
              }
            }
          }
        }
      }
    }
    FillMatchingPlotsMC(collision, collisionInfo, muonTracks, mftTracks, selectedMatchingCandidates, collisionInfo.matchingCandidates, collisionInfo.matchablePairs, fMatchingChi2ScoreMftMchLow, fSelectedMuonsMatchingPlotter.get());

    std::vector<int64_t> taggedMuons;
    GetTaggedMuons(collisionInfo, muonTracks, selectedMuons, taggedMuons);

    MatchingCandidates taggedMatchingCandidates;
    for (auto [mchIndex, globalTracksVector] : collisionInfo.matchingCandidates) {
      if (std::find(taggedMuons.begin(), taggedMuons.end(), mchIndex) != taggedMuons.end()) {
        taggedMatchingCandidates[mchIndex] = globalTracksVector;
      }
    }
    FillMatchingPlotsMC(collision, collisionInfo, muonTracks, mftTracks, taggedMatchingCandidates, collisionInfo.matchingCandidates, collisionInfo.matchablePairs, fMatchingChi2ScoreMftMchLow, fTaggedMuonsMatchingPlotter.get());

    // Di-muon analysis
    FillDimuonPlotsMC(collisionInfo, collisions, muonTracks, mftTracks, collisionInfo.matchablePairs);

    // Diagnostics
    if (false) {
      for (auto [mchIndex, globalTracksVector] : collisionInfo.matchingCandidates) {
        int trueMatchIndex = GetTrueMatchIndex(muonTracks, globalTracksVector, collisionInfo.matchablePairs);
        // only inspect badly ranked matches
        if (trueMatchIndex < 2) continue;

        auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
        int decayRanking = GetDecayRanking(mchTrack, mftTracks);
        // only inspect direct matches (no intermediate decay between MFT and MCH)
        if (decayRanking != 1) continue;

        DrawTracks(collisions, muonTracks, mftTracks, mftCovs,
            0, collisionInfo.matchablePairs, mchIndex, globalTracksVector);
      }
    }
  }

  template <class C, class TMUON, class TMFT, class CMFT>
  void ProcessCollision(const CollisionInfo& collisionInfo,
                          C const& collisions,
                          TMUON const& muonTracks,
                          TMFT const& mftTracks,
                          CMFT const& mftCovs)
  {
    //std::cout << std::endl << std::format("ProcessMatching() collision #{}", collisionInfo.index) << std::endl;
    auto collision = collisions.rawIteratorAt(collisionInfo.index);

    registry.get<TH1>(HIST("tracksMultiplicityMFT"))->Fill(collisionInfo.mftTracks.size());
    registry.get<TH1>(HIST("tracksMultiplicityMCH"))->Fill(collisionInfo.mchTracks.size());

    // Leading / sub-leading matching chi2 difference
    for (auto [mchIndex, globalTracksVector] : collisionInfo.matchingCandidates) {
      if (globalTracksVector.size() < 2)
        continue;
      auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);

      // MCH track quality flag
      if (!IsGoodGlobalMuon(mchTrack, collision)) continue;

      float dchi2 = globalTracksVector[1].matchChi2 - globalTracksVector[0].matchChi2;
      registry.get<TH1>(HIST("matchCandidatesDeltaChi2"))->Fill(dchi2);
    }
  }

  void processQAMC(MyEvents const& collisions,
                   aod::BCsWithTimestamps const& bcs,
                   MyMuonsMC const& muonTracks,
                   MyMFTsMC const& mftTracks,
                   MyMFTCovariances const& mftCovs,
                   aod::McParticles const& mcParticles)
  {
    auto bc = bcs.begin();
    initCCDB(bc);

    for (auto& muon : muonTracks) {
      registry.get<TH1>(HIST("nTracksPerType"))->Fill(static_cast<int>(muon.trackType()));
    }

    FillCollisions(collisions, bcs, muonTracks, mftTracks, fCollisionInfos);

    mftTrackCovs.clear();
    for (auto& mftTrackCov : mftCovs) {
      mftTrackCovs[mftTrackCov.matchMFTTrackId()] = mftTrackCov.globalIndex();
    }

    for (auto const& [collisionIndex, collisionInfo] : fCollisionInfos) {
      ProcessCollisionMC(collisionInfo, collisions, muonTracks, mftTracks, mftCovs, mcParticles);
    }
  }

  PROCESS_SWITCH(qaMatching, processQAMC, "process qa MC", true);
/*
  void processQA(MyEvents const& collisions,
                   aod::BCsWithTimestamps const& bcs,
                   MyMuons const& muonTracks,
                   MyMFTs const& mftTracks,
                   MyMFTCovariances const& mftCovs)
  {
    auto bc = bcs.begin();
    initCCDB(bc);

    for (auto& muon : muonTracks) {
      registry.get<TH1>(HIST("nTracksPerType"))->Fill(static_cast<int>(muon.trackType()));
    }

    FillCollisions(collisions, bcs, muonTracks, mftTracks, fCollisionInfos);

    mftTrackCovs.clear();
    for (auto& mftTrackCov : mftCovs) {
      mftTrackCovs[mftTrackCov.matchMFTTrackId()] = mftTrackCov.globalIndex();
    }

    for (auto const& [collisionIndex, collisionInfo] : fCollisionInfos) {
      ProcessCollision(collisionInfo, collisions, muonTracks, mftTracks, mftCovs);
    }
  }

  PROCESS_SWITCH(qaMatching, processQA, "process qa", false);
*/
  void processAlignment(MyEvents const& collisions,
                        aod::BCsWithTimestamps const& bcs,
                        MyMuonsMC const& muonTracks,
                        MyMFTsMC const& mftTracks,
                        aod::McParticles const& mcParticles)
  {
    auto bc = bcs.begin();
    initCCDB(bc);

    for (auto& muon : muonTracks) {
      registry.get<TH1>(HIST("nTracksPerType"))->Fill(static_cast<int>(muon.trackType()));
    }

    FillCollisions(collisions, bcs, muonTracks, mftTracks, fCollisionInfos);

    for (auto const& [collisionIndex, collisionInfo] : fCollisionInfos) {
      auto collision = collisions.rawIteratorAt(collisionInfo.index);

      // plot MFT-MCH tracks difference at matching plane for wrong pairs
      // outer loop on MCH tracks associated to the current collision
      for (auto const& muonTrackIndex1 : collisionInfo.mchTracks) {
        auto const& muonTrack1 = muonTracks.rawIteratorAt(muonTrackIndex1);
        if (static_cast<int>(muonTrack1.trackType()) != 3)
          continue;

        auto muonTrack1Par = FwdToTrackPar(muonTrack1, muonTrack1);
        auto muonTrack1ParShifted = muonTrack1Par;
        TransformMCH(muonTrack1ParShifted);

        if (muonTrack1.p() > 20) {
          int quadrant = GetQuadrant(muonTrack1);

          auto muonTrack1AtDCA = mExtrap.FwdtoMCH(muonTrack1Par);
          o2::mch::TrackExtrap::extrapToVertexWithoutBranson(muonTrack1AtDCA, collision.posZ());
          double dcax = muonTrack1AtDCA.getNonBendingCoor() - collision.posX();
          double dcay = muonTrack1AtDCA.getBendingCoor() - collision.posY();

          std::get<std::shared_ptr<TH1>>(dcaHistos[quadrant]["DCA_x"])->Fill(dcax);
          std::get<std::shared_ptr<TH1>>(dcaHistos[quadrant]["DCA_y"])->Fill(dcay);

          auto muonTrack1AtDCAShifted = mExtrap.FwdtoMCH(muonTrack1ParShifted);
          o2::mch::TrackExtrap::extrapToVertexWithoutBranson(muonTrack1AtDCAShifted, collision.posZ());
          double dcaxShifted = muonTrack1AtDCAShifted.getNonBendingCoor() - collision.posX();
          double dcayShifted = muonTrack1AtDCAShifted.getBendingCoor() - collision.posY();

          std::get<std::shared_ptr<TH1>>(dcaHistos[quadrant]["DCA_x_shifted"])->Fill(dcaxShifted);
          std::get<std::shared_ptr<TH1>>(dcaHistos[quadrant]["DCA_y_shifted"])->Fill(dcayShifted);
        }

        auto muonTrack1AtVertex = mExtrap.FwdtoMCH(muonTrack1Par);
        o2::mch::TrackExtrap::extrapToVertex(muonTrack1AtVertex,
                                             collision.posX(), collision.posY(), collision.posZ(),
                                             collision.covXX(), collision.covYY());

        auto muonTrack1AtVertexShifted = mExtrap.FwdtoMCH(muonTrack1ParShifted);
        o2::mch::TrackExtrap::extrapToVertex(muonTrack1AtVertexShifted,
                                             collision.posX(), collision.posY(), collision.posZ(),
                                             collision.covXX(), collision.covYY());

        for (auto const& muonTrackIndex2 : collisionInfo.mchTracks) {
          if (muonTrackIndex1 >= muonTrackIndex2)
            continue;

          auto const& muonTrack2 = muonTracks.rawIteratorAt(muonTrackIndex2);
          if (static_cast<int>(muonTrack2.trackType()) != 3)
            continue;

          auto muonTrack2Par = FwdToTrackPar(muonTrack2, muonTrack2);
          auto muonTrack2ParShifted = muonTrack2Par;
          TransformMCH(muonTrack2ParShifted);

          auto muonTrack2AtVertex = mExtrap.FwdtoMCH(muonTrack2Par);
          o2::mch::TrackExtrap::extrapToVertex(muonTrack2AtVertex,
                                               collision.posX(), collision.posY(), collision.posZ(),
                                               collision.covXX(), collision.covYY());

          auto muonTrack2AtVertexShifted = mExtrap.FwdtoMCH(muonTrack2ParShifted);
          o2::mch::TrackExtrap::extrapToVertex(muonTrack2AtVertexShifted,
                                               collision.posX(), collision.posY(), collision.posZ(),
                                               collision.covXX(), collision.covYY());

          int sign1 = muonTrack1.sign();
          int sign2 = muonTrack2.sign();

          // only consider opposite-sign pairs
          if ((sign1 * sign2) >= 0)
            continue;

          const auto& muonPos = sign1 > 0 ? muonTrack1AtVertex : muonTrack2AtVertex;
          const auto& muonNeg = sign1 < 0 ? muonTrack1AtVertex : muonTrack2AtVertex;

          double mass = GetMuMuInvariantMass(muonPos, muonNeg);

          const auto& muonPosShifted = sign1 > 0 ? muonTrack1AtVertexShifted : muonTrack2AtVertexShifted;
          const auto& muonNegShifted = sign1 < 0 ? muonTrack1AtVertexShifted : muonTrack2AtVertexShifted;

          double massShifted = GetMuMuInvariantMass(muonPosShifted, muonNegShifted);

          registryAlignment.get<TH1>(HIST("resolution/MC/invariantMass"))->Fill(mass);
          registryAlignment.get<TH1>(HIST("resolution/MC/invariantMass_shifted"))->Fill(massShifted);
        }
      }
    }
  }

  PROCESS_SWITCH(qaMatching, processAlignment, "process alignment", false);

  void processMftTracks(MyEvents const& collisions,
                        aod::BCsWithTimestamps const& bcs,
                        MyMFTs const& mftTracks)
  {
    auto bc = bcs.begin();
    initCCDB(bc);

    std::ofstream outFile("points.txt");

    for (const auto& collision : collisions) {
      auto bc = bcs.rawIteratorAt(collision.bcId());
      auto bcId = bc.globalBC();
      auto bcTimestamp = bc.timestamp();

      outFile << std::format("BC #{} timestamp {} collision #{}", bcId, bcTimestamp, collision.globalIndex()) << std::endl;

      for (const auto& mftTrack : mftTracks) {
        if (!mftTrack.has_collision())
          continue;
        int64_t collisionId = mftTrack.collisionId();
        if (collisionId != collision.globalIndex())
          continue;

        outFile << std::format("  c1->cd(); ellpise = new TEllipse({}, {}, 0.05, 0.05); ellipse->SetFillColor(kBlue); ellipse->SetFillStyle(1001); ellipse->Draw()",
            mftTrack.x(), mftTrack.y()) << std::endl;

        auto mftProp = PropagateToZMFT(FwdToTrackPar(mftTrack), -77.5f);

        outFile << std::format("  c2->cd(); ellpise = new TEllipse({}, {}, 0.05, 0.05); ellipse->SetFillColor(kBlue); ellipse->SetFillStyle(1001); ellipse->Draw()",
            mftProp.getX(), mftProp.getY()) << std::endl;
      }
    }
  }

  PROCESS_SWITCH(qaMatching, processMftTracks, "process MFT tracks", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<qaMatching>(cfgc)};
};
