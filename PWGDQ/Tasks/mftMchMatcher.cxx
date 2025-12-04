// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copdyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file mftMchMatcher.cxx
/// \brief MFT-MCH matching tool for data preparation

#include "Common/DataModel/EventSelection.h"

#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "GlobalTracking/MatchGlobalFwd.h"
#include "MFTTracking/Constants.h"
#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include <TGeoGlobalMagField.h>

#include <TTree.h>

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
using MyMuonsWithCov = soa::Join<aod::FwdTracks, aod::FwdTracksCov>;
using MyMuonsMC = soa::Join<aod::FwdTracks, aod::FwdTracksCov, aod::McFwdTrackLabels>;
using MyMFTs = aod::MFTTracks;
using MyMFTCovariances = aod::MFTTracksCov;
using MyMFTsMC = soa::Join<aod::MFTTracks, aod::McMFTTrackLabels>;

using MyMuon = MyMuonsWithCov::iterator;
using MyMuonMC = MyMuonsMC::iterator;
using MyMFT = MyMFTs::iterator;
using MyMFTCovariance = MyMFTCovariances::iterator;


using namespace std;

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix55Std = ROOT::Math::SMatrix<double, 5>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

// Track parameter structure at matching plane
struct TrackAtPlane {
  float x, y, phi, tanl, invQPt;
  float cXX, cXY, cYY;
  float cPhiX, cPhiY, cPhiPhi;
  float cTglX, cTglY, cTglPhi, cTglTgl;
  float c1PtX, c1PtY, c1PtPhi, c1PtTgl, c1Pt21Pt2;
};

// MFT Track structure
struct MFTTrack {
  int collisionId;
  float z, x, y, phi, tgl, signed1Pt;
  float cXX, cXY, cYY, cPhiX, cPhiY, cPhiPhi;
  float cTglX, cTglY, cTglPhi, cTglTgl;
  float c1PtX, c1PtY, c1PtPhi, c1PtTgl, c1Pt21Pt2;
};

// MCH Track structure
struct MCHTrack {
  int collisionId;
  float z, x, y, phi, tgl, signed1Pt;
  float cXX, cXY, cYY, cPhiX, cPhiY, cPhiPhi;
  float cTglX, cTglY, cTglPhi, cTglTgl;
  float c1PtX, c1PtY, c1PtPhi, c1PtTgl, c1Pt21Pt2;
  
  // Original matching data
  float chi2MatchOriginal;
  int indexMFTOriginal;
};

namespace o2::aod
{
namespace fwdmatchcandidates
{
// MCH track parameters at matching plane
DECLARE_SOA_COLUMN(XMch, xMch, float);           //! X position of the extrapolated MCH track
DECLARE_SOA_COLUMN(YMch, yMch, float);           //! Y position of the extrapolated MCH track
DECLARE_SOA_COLUMN(PhiMch, phiMch, float);       //! phi angle of the extrapolated MCH track
DECLARE_SOA_COLUMN(TanlMch, tanlMch, float);     //! tan(labda) of the extrapolated MCH track
DECLARE_SOA_COLUMN(InvQPtMch, invQPtMch, float); //! Q/pT of the extrapolated MCH track
// MFT track parameters at matching plane
DECLARE_SOA_COLUMN(XMft, xMft, float);           //! X position of the extrapolated MFT track
DECLARE_SOA_COLUMN(YMft, yMft, float);           //! Y position of the extrapolated MFT track
DECLARE_SOA_COLUMN(PhiMft, phiMft, float);       //! phi angle of the extrapolated MFT track
DECLARE_SOA_COLUMN(TanlMft, tanlMft, float);     //! tan(labda) of the extrapolated MFT track
DECLARE_SOA_COLUMN(InvQPtMft, invQPtMft, float); //! Q/pT of the extrapolated MFT track
// Match properties
DECLARE_SOA_COLUMN(Label, label, int);           //! =1 for leading matches, =0 otherwise
DECLARE_SOA_COLUMN(TrueMatch, trueMatch, int);   //! =1 for true matches, =0 otherwise
}

DECLARE_SOA_TABLE(FwdMatchMLCandidates, "AOD", "FWDMLCAND",
                  fwdmatchcandidates::XMch,
                  fwdmatchcandidates::YMch,
                  fwdmatchcandidates::PhiMch,
                  fwdmatchcandidates::TanlMch,
                  fwdmatchcandidates::InvQPtMch,
                  fwdmatchcandidates::XMft,
                  fwdmatchcandidates::YMft,
                  fwdmatchcandidates::PhiMft,
                  fwdmatchcandidates::TanlMft,
                  fwdmatchcandidates::InvQPtMft,
                  fwdmatchcandidates::Label,
                  fwdmatchcandidates::TrueMatch);
}

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

struct mftMchMatcher {
  Produces<o2::aod::FwdMatchMLCandidates> fwdMatchMLCandidates;
  struct FwdMatchMLCandidate
  {
    float xMch;
    float yMch;
    float phiMch;
    float tanlMch;
    float invQPtMch;
    float xMft;
    float yMft;
    float phiMft;
    float tanlMft;
    float invQPtMft;
    int label;
    int trueMatch;
  };
  ////   Variables for selecting muon tracks
  Configurable<float> fPMchLow{"cfgPMchLow", 0.0f, ""};
  Configurable<float> fPtMchLow{"cfgPtMchLow", 0.7f, ""};
  //Configurable<float> fEtaMchLow{"cfgEtaMchLow", -4.0f, ""};
  //Configurable<float> fEtaMchUp{"cfgEtaMchUp", -2.5f, ""};
  Configurable<float> fRabsLow{"cfgRabsLow", 17.6f, ""};
  Configurable<float> fRabsUp{"cfgRabsUp", 89.5f, ""};
  Configurable<float> fSigmaPdcaUp{"cfgPdcaUp", 6.f, ""};
  Configurable<float> fTrackChi2MchUp{"cfgTrackChi2MchUp", 5.f, ""};
  Configurable<float> fMatchingChi2MchMidUp{"cfgMatchingChi2MchMidUp", 999.f, ""};

  ////   Variables for selecting mft tracks
  Configurable<float> fEtaMftLow{"cfgEtaMftlow", -3.6f, ""};
  Configurable<float> fEtaMftUp{"cfgEtaMftup", -2.5f, ""};

  ////   Variables for matching configuration
  Configurable<float> fMatchingPlaneZ{"cfgMatchingPlaneZ", -77.5f, ""};
  Configurable<int> fMaxCandidates{"cfgMaxCandidates", 0, ""};

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

  double mBzAtMftCenter{0};
  o2::globaltracking::MatchGlobalFwd mExtrap;

  int mRunNumber{0}; // needed to detect if the run changed and trigger update of magnetic field
  Service<o2::ccdb::BasicCCDBManager> ccdbManager;
  o2::ccdb::CcdbApi fCCDBApi;

  std::unordered_map<int64_t, int32_t> mftCovIndexes;

  // vector of all MFT-MCH(-MID) matching candidates associated to the same MCH(-MID) track,
  // to be sorted in descending order with respect to the matching score
  // the map key is the MCH(-MID) track global index
  // the elements are pairs og global muon track indexes and associated matching scores
  using MatchingCandidates = std::unordered_map<int64_t, std::vector<std::pair<int64_t, double>>>;

  // Output variables
  float out_X_MCH, out_Y_MCH, out_Phi_MCH, out_TanL_MCH, out_InvQPt_MCH;
  float out_X_MFT, out_Y_MFT, out_Phi_MFT, out_TanL_MFT, out_InvQPt_MFT;
  float out_Chi2Match, out_Chi2MatchOriginal;
  int out_Label, out_TrueMatch;

  //TFile* mOutFile{nullptr};
  //TTree* mOutTree{nullptr};

  HistogramRegistry registry{"registry", {}};

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

  TrackAtPlane PropagateToZMCH(const o2::dataformats::GlobalFwdTrack& muon, const double z)
  {
    TrackAtPlane result;

    auto mchTrack = mExtrap.FwdtoMCH(muon);

    float absFront = -90.f;
    float absBack = -505.f;

    if (muon.getZ() < absBack && z > absFront) {
      // extrapolation through the absorber in the upstream direction
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrack, z);
    } if (muon.getZ() < absBack && z <= absFront && z > absBack) {
      // extrapolation inside the absorber in the upstream direction
      // first extrapolate through the whole absorber, correcting for energy loss
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrack, absFront + 0.1f);
      // then extrapolate back to the given z
      o2::mch::TrackExtrap::extrapToZCov(mchTrack, z);
    } else {
      // all other cases
      o2::mch::TrackExtrap::extrapToZCov(mchTrack, z);
    }

    auto propTrack = mExtrap.MCHtoFwd(mchTrack);

    result.x = propTrack.getX();
    result.y = propTrack.getY();
    result.phi = propTrack.getPhi();
    result.tanl = propTrack.getTanl();
    result.invQPt = propTrack.getInverseMomentum();

    const auto& cov = propTrack.getCovariances();
    result.cXX = cov(0, 0);
    result.cXY = cov(0, 1);
    result.cYY = cov(1, 1);
    result.cPhiX = cov(0, 2);
    result.cPhiY = cov(1, 2);
    result.cPhiPhi = cov(2, 2);
    result.cTglX = cov(0, 3);
    result.cTglY = cov(1, 3);
    result.cTglPhi = cov(2, 3);
    result.cTglTgl = cov(3, 3);
    result.c1PtX = cov(0, 4);
    result.c1PtY = cov(1, 4);
    result.c1PtPhi = cov(2, 4);
    result.c1PtTgl = cov(3, 4);
    result.c1Pt21Pt2 = cov(4, 4);

    return result;
  }

  TrackAtPlane PropagateToZMFT(const o2::dataformats::GlobalFwdTrack& mftTrack, const double z)
  {
    TrackAtPlane result;

    o2::dataformats::GlobalFwdTrack propTrack{mftTrack};
    propTrack.propagateToZ(z, mBzAtMftCenter);

    result.x = propTrack.getX();
    result.y = propTrack.getY();
    result.phi = propTrack.getPhi();
    result.tanl = propTrack.getTanl();
    result.invQPt = propTrack.getInverseMomentum();

    const auto& cov = propTrack.getCovariances();
    result.cXX = cov(0, 0);
    result.cXY = cov(0, 1);
    result.cYY = cov(1, 1);
    result.cPhiX = cov(0, 2);
    result.cPhiY = cov(1, 2);
    result.cPhiPhi = cov(2, 2);
    result.cTglX = cov(0, 3);
    result.cTglY = cov(1, 3);
    result.cTglPhi = cov(2, 3);
    result.cTglTgl = cov(3, 3);
    result.c1PtX = cov(0, 4);
    result.c1PtY = cov(1, 4);
    result.c1PtPhi = cov(2, 4);
    result.c1PtTgl = cov(3, 4);
    result.c1Pt21Pt2 = cov(4, 4);

    return result;
  }

  template <class TMUON, class TMFT>
  void GetMatchablePairs(TMUON const& muonTracks,
                         TMFT const& mftTracks,
                         std::vector<std::pair<int64_t, int64_t>>& matchablePairs)
  {
    matchablePairs.clear();
    for (const auto& muonTrack : muonTracks) {
      // only consider MCH standalone or MCH-MID matches
      if (static_cast<int>(muonTrack.trackType()) <= 2)
        continue;

      // skip tracks that do not have an associated MC particle
      if (!muonTrack.has_mcParticle())
        continue;
      // get the index associated to the MC particle
      auto muonMcParticle = muonTrack.mcParticle();
      int64_t muonMcTrackIndex = muonMcParticle.globalIndex();

      for (const auto& mftTrack : mftTracks) {
        // skip tracks that do not have an associated MC particle
        if (!mftTrack.has_mcParticle())
          continue;
        // get the index associated to the MC particle
        auto mftMcParticle = mftTrack.mcParticle();
        int64_t mftMcTrackIndex = mftMcParticle.globalIndex();

        if (muonMcTrackIndex == mftMcTrackIndex) {
          matchablePairs.emplace_back(std::make_pair(static_cast<int64_t>(muonTrack.globalIndex()),
                                                     static_cast<int64_t>(mftTrack.globalIndex())));
        }
      }
    }
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

  template <class TMUON>
  int GetTrueMatchIndex(TMUON const& muonTracks,
                        const std::vector<std::pair<int64_t, double>>& matchCandidatesVector,
                        const std::vector<std::pair<int64_t, int64_t>>& matchablePairs)
  {
    // find the index of the matching candidate that corresponds to the true match
    // index=1 corresponds to the leading candidate
    // index=0 means no candidate was found that corresponds to the true match
    int trueMatchIndex = 0;
    for (size_t i = 0; i < matchCandidatesVector.size(); i++) {
      auto const& muonTrack = muonTracks.rawIteratorAt(matchCandidatesVector[i].first);

      if (IsTrueGlobalMatching(muonTrack, matchablePairs)) {
        trueMatchIndex = i + 1;
        break;
      }
    }
    return trueMatchIndex;
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
    auto mchTrackAtVertex = FwdtoMCH(FwdToTrackPar(mchTrack, mchTrack));
    o2::mch::TrackExtrap::extrapToVertex(mchTrackAtVertex, collision.posX(), collision.posY(), collision.posZ(), collision.covXX(), collision.covYY());

    // double pUncorr = mchTrack.p();
    double p = mchTrackAtVertex.p();

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
  bool IsGoodGlobalMuon(const T& muonTrack, const C& collision)
  {
    return IsGoodMuon(muonTrack, collision, fTrackChi2MchUp, fPMchLow, fPtMchLow, {fEtaMftLow, fEtaMftUp}, {fRabsLow, fRabsUp}, fSigmaPdcaUp);
  }

  float ComputeMatchChi2(const TrackAtPlane& mchTrack, const TrackAtPlane& mftTrack)
  {
    // MFT parameters
    SMatrix5 m_k(mftTrack.x, mftTrack.y, mftTrack.phi, mftTrack.tanl, mftTrack.invQPt);

    // MFT covariance
    SMatrix55 V_k;
    V_k(0, 0) = mftTrack.cXX;
    //V_k(0, 1) = mftTrack.cXY;
    V_k(1, 1) = mftTrack.cYY;
    //V_k(0, 2) = mftTrack.cPhiX;
    //V_k(1, 2) = mftTrack.cPhiY;
    V_k(2, 2) = mftTrack.cPhiPhi;
    //V_k(0, 3) = mftTrack.cTglX;
    //V_k(1, 3) = mftTrack.cTglY;
    //V_k(2, 3) = mftTrack.cTglPhi;
    V_k(3, 3) = mftTrack.cTglTgl;
    //V_k(0, 4) = mftTrack.c1PtX;
    //V_k(1, 4) = mftTrack.c1PtY;
    //V_k(2, 4) = mftTrack.c1PtPhi;
    //V_k(3, 4) = mftTrack.c1PtTgl;
    V_k(4, 4) = mftTrack.c1Pt21Pt2;

    // MCH parameters
    SMatrix5 GlobalMuonTrackParameters(mchTrack.x, mchTrack.y, mchTrack.phi,
                                        mchTrack.tanl, mchTrack.invQPt);

    // MCH covariance
    SMatrix55 GlobalMuonTrackCovariances;
    GlobalMuonTrackCovariances(0, 0) = mchTrack.cXX;
    GlobalMuonTrackCovariances(0, 1) = mchTrack.cXY;
    GlobalMuonTrackCovariances(1, 1) = mchTrack.cYY;
    GlobalMuonTrackCovariances(0, 2) = mchTrack.cPhiX;
    GlobalMuonTrackCovariances(1, 2) = mchTrack.cPhiY;
    GlobalMuonTrackCovariances(2, 2) = mchTrack.cPhiPhi;
    GlobalMuonTrackCovariances(0, 3) = mchTrack.cTglX;
    GlobalMuonTrackCovariances(1, 3) = mchTrack.cTglY;
    GlobalMuonTrackCovariances(2, 3) = mchTrack.cTglPhi;
    GlobalMuonTrackCovariances(3, 3) = mchTrack.cTglTgl;
    GlobalMuonTrackCovariances(0, 4) = mchTrack.c1PtX;
    GlobalMuonTrackCovariances(1, 4) = mchTrack.c1PtY;
    GlobalMuonTrackCovariances(2, 4) = mchTrack.c1PtPhi;
    GlobalMuonTrackCovariances(3, 4) = mchTrack.c1PtTgl;
    GlobalMuonTrackCovariances(4, 4) = mchTrack.c1Pt21Pt2;

    // Sum of covariances
    SMatrix55Std sumCov = V_k + GlobalMuonTrackCovariances;

    // Invert
    SMatrix55Std invResCov = sumCov;
    bool inversionOk = invResCov.Invert();
    if (!inversionOk) {
      return 999999.0f; // Very large value for failed inversion
    }

    // Residuals
    SMatrix5 r_k_kminus1 = m_k - GlobalMuonTrackParameters;

    // Chi2 = r^T * C^-1 * r
    auto chi2 = ROOT::Math::Similarity(r_k_kminus1, invResCov);

    // Check for valid result
    if (!isfinite(chi2) || chi2 < 0.0) {
      return 999999.0f;
    }

    return static_cast<float>(chi2);
  }

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
    o2::mch::TrackExtrap::setField();
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
/*
    // Create output file and tree
    mOutFile = new TFile(fOutputFileName->c_str(), "RECREATE");
    mOutTree = new TTree("MFTMCHMatches", "MFT-MCH Matching Results");
    mOutTree->SetDirectory(mOutFile);

    mOutTree->Branch("X_MCH", &out_X_MCH);
    mOutTree->Branch("Y_MCH", &out_Y_MCH);
    mOutTree->Branch("phi_MCH", &out_Phi_MCH);
    mOutTree->Branch("tanL_MCH", &out_TanL_MCH);
    mOutTree->Branch("invqpt_MCH", &out_InvQPt_MCH);
    mOutTree->Branch("X_MFT", &out_X_MFT);
    mOutTree->Branch("Y_MFT", &out_Y_MFT);
    mOutTree->Branch("phi_MFT", &out_Phi_MFT);
    mOutTree->Branch("tanL_MFT", &out_TanL_MFT);
    mOutTree->Branch("invqpt_MFT", &out_InvQPt_MFT);
    mOutTree->Branch("chi2Match", &out_Chi2Match);
    mOutTree->Branch("chi2MatchOriginal", &out_Chi2MatchOriginal);
    mOutTree->Branch("label", &out_Label);
    mOutTree->Branch("trueMatch", &out_TrueMatch);
*/
    AxisSpec chi2Axis = {1000, 0, 1000, "chi^{2}"};
    AxisSpec chi2ProdAxis = {1000, 0, 1000, "chi^{2} (prod)"};
    registry.add("matchingChi2", "Matching #chi^{2}", {HistType::kTH1F, {chi2Axis}});
    registry.add("matchingChi2Prod", "Matching #chi^{2} (production)", {HistType::kTH1F, {chi2ProdAxis}});
    registry.add("matchingChi2Corr", "Matching #chi^{2} (current vs. production)", {HistType::kTH2F, {chi2ProdAxis, chi2Axis}});
  }

  template <class TMUON, class C>
  void FillMatchingCandidates(TMUON const& muonTracks,
                              C const& collisions,
                              MatchingCandidates& matchingCandidates)
  {
    for (auto muonTrack : muonTracks) {
      // only consider global MFT-MCH-MID matches
      if (static_cast<int>(muonTrack.trackType()) >= 2) {
        continue;
      }

      if (!muonTrack.has_collision()) {
        continue;
      }
      const auto& collision = collisions.rawIteratorAt(muonTrack.collisionId());

      int64_t muonTrackIndex = muonTrack.globalIndex();
      double matchingChi2 = muonTrack.chi2MatchMCHMFT();
      auto const& mchTrack = muonTrack.template matchMCHTrack_as<TMUON>();
      int64_t mchTrackIndex = mchTrack.globalIndex();

      // only consider good MCH tracks in the MFT acceptance
      if (!IsGoodGlobalMuon(mchTrack, collision)) {
        continue;
      }

      // check if a vector of global muon candidates is already available for the current MCH index
      // if not, initialize a new one and add the current global muon track
      // bool globalMuonTrackFound = false;
      auto matchingCandidateIterator = matchingCandidates.find(mchTrackIndex);
      if (matchingCandidateIterator != matchingCandidates.end()) {
        matchingCandidateIterator->second.push_back(std::make_pair(muonTrackIndex, matchingChi2));
        // globalMuonTrackFound = true;
      } else {
        matchingCandidates[mchTrackIndex].push_back(std::make_pair(muonTrackIndex, matchingChi2));
      }
    }

    // sort the vectors of matching candidates in ascending order based on the matching chi2 value
    auto compareMatchingChi2 = [](std::pair<int64_t, double> track1, std::pair<int64_t, double> track2) -> bool {
      return (track1.second < track2.second);
    };

    for (auto& [mchIndex, globalTracksVector] : matchingCandidates) {
      std::sort(globalTracksVector.begin(), globalTracksVector.end(), compareMatchingChi2);
    }
}

  template <class TMUON, class TMFT, class CMFT>
  void FillCandidates(TMUON const& muonTracks,
                TMFT const& /*mftTracks*/,
                CMFT const& mftCovs,
                int64_t mchIndex,
                const std::vector<std::pair<int64_t, double>>& candidates,
                const std::vector<std::pair<int64_t, int64_t>>& matchablePairs,
                std::vector<FwdMatchMLCandidate>& mlCandidates)
  {
    auto const& mchTrack = muonTracks.rawIteratorAt(mchIndex);
    int trueMatchIndex = GetTrueMatchIndex(muonTracks, candidates, matchablePairs);

    int matchIndex = 1;
    for (const auto& [muonIndex, score] : candidates) {
      const auto& muonTrack = muonTracks.rawIteratorAt(muonIndex);

      // get MFT standalone track
      auto const& mftTrack = muonTrack.template matchMFTTrack_as<TMFT>();
      if (mftCovIndexes.count(mftTrack.globalIndex()) < 1) {
        return;
      }
      auto const& mftTrackCov = mftCovs.rawIteratorAt(mftCovIndexes[mftTrack.globalIndex()]);

      auto mchAtPlane = PropagateToZMCH(FwdToTrackPar(mchTrack, mchTrack), fMatchingPlaneZ);
      auto mftAtPlane = PropagateToZMFT(FwdToTrackPar(mftTrack, mftTrackCov), fMatchingPlaneZ);

      double chi2 = ComputeMatchChi2(mchAtPlane, mftAtPlane);
      double chi2Prod = muonTrack.chi2MatchMCHMFT();

      FwdMatchMLCandidate candidate(
          static_cast<float>(mchAtPlane.x),
          static_cast<float>(mchAtPlane.y),
          static_cast<float>(mchAtPlane.phi),
          static_cast<float>(mchAtPlane.tanl),
          static_cast<float>(mchAtPlane.invQPt),
          static_cast<float>(mftAtPlane.x),
          static_cast<float>(mftAtPlane.y),
          static_cast<float>(mftAtPlane.phi),
          static_cast<float>(mftAtPlane.tanl),
          static_cast<float>(mftAtPlane.invQPt),
          static_cast<int>((matchIndex == 1) ? 1 : 0),
          static_cast<int>((matchIndex == trueMatchIndex) ? 1 : 0)
      );
      mlCandidates.emplace_back(candidate);

      registry.get<TH1>(HIST("matchingChi2"))->Fill(chi2);
      registry.get<TH1>(HIST("matchingChi2Prod"))->Fill(chi2Prod);
      registry.get<TH2>(HIST("matchingChi2Corr"))->Fill(chi2Prod, chi2);

      matchIndex += 1;
    }
  }

  void processMC(MyEvents const& collisions,
                 aod::BCsWithTimestamps const& bcs,
                 MyMuonsMC const& muonTracks,
                 MyMFTsMC const& mftTracks,
                 MyMFTCovariances const& mftCovs,
                 aod::McParticles const& /*mcParticles*/)
  {
    auto bc = bcs.begin();
    initCCDB(bc);

    mftCovIndexes.clear();
    for (auto& mftTrackCov : mftCovs) {
      mftCovIndexes[mftTrackCov.matchMFTTrackId()] = mftTrackCov.globalIndex();
    }

    // step 1: collect the matching candidates for each MCH-MID track
    MatchingCandidates matchingCandidates;
    FillMatchingCandidates(muonTracks, collisions, matchingCandidates);

    std::vector<std::pair<int64_t, int64_t>> matchablePairs;
    GetMatchablePairs(muonTracks, mftTracks, matchablePairs);

    // step 2: loop over candidates and fill vector of candidates
    std::vector<FwdMatchMLCandidate> mlCandidates;
    for (auto& [mchIndex, globalTracks] : matchingCandidates) {
      FillCandidates(muonTracks, mftTracks, mftCovs, mchIndex, globalTracks, matchablePairs, mlCandidates);
    }

    fwdMatchMLCandidates.reserve(mlCandidates.size());
    for (const auto& candidate : mlCandidates) {
      //std::cout << "Storing ML candidate" << std::endl;
      fwdMatchMLCandidates(
          candidate.xMch,
      candidate.yMch,
      candidate.phiMch,
      candidate.tanlMch,
      candidate.invQPtMch,
      candidate.xMft,
      candidate.yMft,
      candidate.phiMft,
      candidate.tanlMft,
      candidate.invQPtMft,
      candidate.label,
      candidate.trueMatch
      );
    }

    //mOutFile->cd();
    //mOutTree->Write();
  }

  PROCESS_SWITCH(mftMchMatcher, processMC, "process_MC", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<mftMchMatcher>(cfgc)};
};

