#pragma once

#include <string>

#include "Vec.h"
#include "json.hpp"

namespace cubble
{
struct SimulationProperties
{
    // Parameters from json
    double phiTarget = 0.0;
    double kappa = 0.0;
    int numBubbles = 0;
    int minNumBubbles = 0;
    dvec boxRelativeDimensions = dvec(0, 0, 0);
    double muZero = 0.0;
    double sigmaZero = 0.0;
    double avgRad = 0.0;
    double stdDevRad = 0.0;
    double errorTolerance = 0.0;
    double timeStep = 0.0;
    int rngSeed = 0;
    int numBubblesPerCell = 0;
    int numStepsToRelax = 0;
    double maxDeltaEnergy = 0.0;
    double kParameter = 0.0;
    std::string snapshotFilename = "";
    std::string dataFilename = "";

    // Derived parameters
    double minRad = 0.0;
    double fZeroPerMuZero = 0.0;

#define TO_JSON(j, p, param) j[#param] = p.param
    friend void to_json(nlohmann::json &j, const SimulationProperties &p)
    {
        TO_JSON(j, p, phiTarget);
        TO_JSON(j, p, kappa);
        TO_JSON(j, p, numBubbles);
        TO_JSON(j, p, minNumBubbles);
        TO_JSON(j, p, boxRelativeDimensions);
        TO_JSON(j, p, muZero);
        TO_JSON(j, p, sigmaZero);
        TO_JSON(j, p, avgRad);
        TO_JSON(j, p, stdDevRad);
        TO_JSON(j, p, errorTolerance);
        TO_JSON(j, p, timeStep);
        TO_JSON(j, p, rngSeed);
        TO_JSON(j, p, numBubblesPerCell);
        TO_JSON(j, p, numStepsToRelax);
        TO_JSON(j, p, maxDeltaEnergy);
        TO_JSON(j, p, kParameter);
        TO_JSON(j, p, snapshotFilename);
        TO_JSON(j, p, dataFilename);
    }
#undef TO_JSON

#define FROM_JSON(j, p, param)                               \
    do                                                       \
    {                                                        \
        p.param = j[#param];                                 \
        std::cout << #paran << ": " << p.param << std::endl; \
    } while (0)
    friend void from_json(const nlohmann::json &j, SimulationProperties &p)
    {
        std::cout << "Simulation parameters:\n"
                  << std::endl;
        FROM_JSON(j, p, phiTarget);
        FROM_JSON(j, p, kappa);
        FROM_JSON(j, p, numBubbles);
        FROM_JSON(j, p, minNumBubbles);
        FROM_JSON(j, p, boxRelativeDimensions);
        FROM_JSON(j, p, muZero);
        FROM_JSON(j, p, sigmaZero);
        FROM_JSON(j, p, avgRad);
        FROM_JSON(j, p, stdDevRad);
        FROM_JSON(j, p, errorTolerance);
        FROM_JSON(j, p, timeStep);
        FROM_JSON(j, p, rngSeed);
        FROM_JSON(j, p, numBubblesPerCell);
        FROM_JSON(j, p, numStepsToRelax);
        FROM_JSON(j, p, maxDeltaEnergy);
        FROM_JSON(j, p, kParameter);
        FROM_JSON(j, p, snapshotFilename);
        FROM_JSON(j, p, dataFilename);

        assert(p.muZero > 0);
        assert(p.boxRelativeDimensions.x > 0);
        assert(p.boxRelativeDimensions.y > 0);
        assert(p.boxRelativeDimensions.z > 0);

        p.minRad = p.avgRad * 0.1;
        p.fZeroPerMuZero = p.sigmaZero * p.avgRad / p.muZero;
    }
#undef FROM_JSON
};

} // namespace cubble