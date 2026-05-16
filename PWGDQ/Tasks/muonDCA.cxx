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
#include "PWGDQ/Core/VarManager.h"
#include "PWGDQ/DataModel/ReducedInfoTables.h"

#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "GlobalTracking/MatchGlobalFwd.h"
#include "Common/DataModel/EventSelection.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod;

using ColEvSels = soa::Join<aod::Collisions, aod::EvSels, aod::Mults>;
using BCsRun3 = soa::Join<aod::BCs, aod::Timestamps, aod::BcSels, aod::Run3MatchedToBCSparse>;

// constexpr static uint32_t gkMuonDCAFillMapWithCov = VarManager::ObjTypes::ReducedMuon | VarManager::ObjTypes::ReducedMuonExtra | VarManager::ObjTypes::ReducedMuonCov | VarManager::ObjTypes::MuonDCA;

static o2::globaltracking::MatchGlobalFwd mExtrap;
template <typename T>
bool isSelected(const T& muon);

struct muonExtrap {
  Configurable<std::string> geoPath{"geoPath", "GLO/Config/GeometryAligned", "Path of the geometry file"};
  Configurable<std::string> grpmagPath{"grpmagPath", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
  Configurable<std::string> fConfigCcdbUrl{"ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};

  Service<o2::ccdb::BasicCCDBManager> fCCDB;
  o2::parameters::GRPMagField* grpmag = nullptr; // for run 3, we access GRPMagField from GLO/Config/GRPMagField
  int fCurrentRun;                               // needed to detect if the run changed and trigger update of magnetic field

  o2::aod::rctsel::RCTFlagsChecker rctChecker{"CBT_muon_glo", false, true, true};

  HistogramRegistry registry{
    "registry",
    {}};

  void init(o2::framework::InitContext&)
  {
    // Load geometry
    fCCDB->setURL(fConfigCcdbUrl);
    fCCDB->setCaching(true);
    fCCDB->setLocalObjectValidityChecking();

    if (!o2::base::GeometryManager::isGeometryLoaded()) {
      LOGF(info, "Load geometry from CCDB");
      fCCDB->get<TGeoManager>(geoPath);
    }

    AxisSpec collsAxis = {3, 0.0, 3.0, "number of collisions"};
    AxisSpec pdcaAxis = {5000, 0.0, 5000.0, "p #times DCA"};
    AxisSpec dcaAxis = {200, 0.0, 200.0, "DCA"};
    AxisSpec dcaxAxis = {200, -100.0, 100.0, "DCA_x"};
    AxisSpec dcayAxis = {200, -100.0, 100.0, "DCA_y"};
    AxisSpec rabsAxis = {100, 0., 100.0, "R_{abs}"};
    AxisSpec xAxis = {200, -100., 100.0, "x (cm)"};
    AxisSpec yAxis = {200, -100., 100.0, "y (cm)"};
    AxisSpec zAxis = {200, -100., 100.0, "z (cm)"};

    HistogramConfigSpec collsSpec({HistType::kTH1F, {collsAxis}});
    HistogramConfigSpec pdcaSpec({HistType::kTH1F, {pdcaAxis}});
    HistogramConfigSpec dcaSpec({HistType::kTH1F, {dcaAxis}});
    HistogramConfigSpec dcaxSpec({HistType::kTH1F, {dcaxAxis}});
    HistogramConfigSpec dcaySpec({HistType::kTH1F, {dcayAxis}});
    HistogramConfigSpec rabsSpec({HistType::kTH1F, {rabsAxis}});
    HistogramConfigSpec xSpec({HistType::kTH1F, {xAxis}});
    HistogramConfigSpec ySpec({HistType::kTH1F, {yAxis}});
    HistogramConfigSpec zSpec({HistType::kTH1F, {zAxis}});

    registry.add("colls", "Collisions", collsSpec);
    registry.add("pdca", "pDCA", pdcaSpec);
    registry.add("dca", "DCA", dcaSpec);
    registry.add("dcax", "DCA_x", dcaxSpec);
    registry.add("dcay", "DCA_y", dcaySpec);
    registry.add("rabs", "R_{abs}", rabsSpec);
    registry.add("xAtVtx", "x at vertex", xSpec);
    registry.add("xAtDCA", "x at DCA", xSpec);
    registry.add("xAtRabs", "x at end abs", xSpec);
    registry.add("yAtVtx", "y at vertex", ySpec);
    registry.add("yAtDCA", "y at DCA", ySpec);
    registry.add("yAtRabs", "y at end abs", ySpec);
    registry.add("zAtVtx", "z at vertex", zSpec);
    registry.add("zAtDCA", "z at DCA", zSpec);
    registry.add("zAtRabs", "z at end abs", zSpec);
  }

  void checkCollisions(
      ColEvSels const& cols,
      BCsRun3 const& bcs)
  {
    return;
    //LOGF(info, "checkCollisions() called");
    for (auto& col : cols) {
      registry.get<TH1>(HIST("colls"))->Fill(0);
      if (rctChecker(col)) {
        registry.get<TH1>(HIST("colls"))->Fill(1);
      } else {
        auto bc = col.foundBC_as<BCsRun3>();
        int64_t ts = bc.timestamp();
        int runNumber = bc.runNumber();
        //LOGF(info, "Bad collision found in run " + std::to_string(runNumber) + " at " + std::to_string(ts));
        registry.get<TH1>(HIST("colls"))->Fill(2);
      }
    }
  }

  PROCESS_SWITCH(muonExtrap, checkCollisions, "check collisions", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<muonExtrap>(cfgc)};
};
