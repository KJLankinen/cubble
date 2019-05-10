// -*- C++ -*-

#pragma once

#include <memory>
#include <vector>

#include "Simulator.cuh"
#include "Env.h"

namespace cubble
{
class CubbleApp
{
public:
	CubbleApp(const std::string &inputFile,
						const std::string &saveFile);
	~CubbleApp();
	void run();

private:
	void setupSimulation();
	void stabilizeSimulation();
	void runSimulation();
	void saveSnapshotToFile();
	double getScaledTime() const { return simulator->getSimulationTime() * env->getKParameter() / (env->getAvgRad() * env->getAvgRad()); };

	static int numSnapshots;
	std::unique_ptr<Simulator> simulator;
	std::shared_ptr<Env> env;
	std::vector<double> hostData;
};
} // namespace cubble
