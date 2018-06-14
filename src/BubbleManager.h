// -*- C++ -*-

#pragma once

#include "Bubble.h"
#include "Env.h"
#include "CudaContainer.h"
#include "Cell.h"

#include <vector>
#include <assert.h>
#include <memory>
#include <cuda_runtime.h>

namespace cubble
{
    class BubbleManager
    {
    public:
	BubbleManager(std::shared_ptr<Env> e);
	~BubbleManager();

	double getVolumeOfBubbles() const;
	
	void setBubbles(const std::vector<Bubble> &b);
	void setBubblesFromDevice(CudaContainer<Bubble> &b);
	void getBubbles(std::vector<Bubble> &b) const;
	void getBubbles(CudaContainer<Bubble> &b) const;
	size_t getBubblesSize() const { return bubbles.size(); }

	void setIndicesFromDevice(CudaContainer<int> &i);
	void getIndices(std::vector<int> &i) const;
	void getIndices(CudaContainer<int> &i) const;
	int getIndicesSize() const { return indices.size(); }
	
	void setCellsFromDevice(CudaContainer<Cell> &c);
	void getCells(std::vector<Cell> &c) const;
	void getCells(CudaContainer<Cell> &c) const;
	int getCellsSize() const { return cells.size(); }
	
	Bubble getBubble(size_t i) const;
	
    private:
	
	std::shared_ptr<Env> env;
	
	std::vector<Bubble> bubbles;
	std::vector<int> indices;
	std::vector<Cell> cells;
    };
}
