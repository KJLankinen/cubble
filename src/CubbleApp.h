// -*- C++ -*-

#pragma once

#include <memory>

#include "Simulator.h"
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

	static int numSnapshots;
	std::unique_ptr<Simulator> simulator;
	std::shared_ptr<Env> env;
};
} // namespace cubble
