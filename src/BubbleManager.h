// -*- C++ -*-

#pragma once

#include "Bubble.h"
#include "Env.h"
#include "CudaContainer.h"

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
	
	void setBubbles(const std::vector<Bubble> &b);
	void setBubblesFromDevice(CudaContainer<Bubble> &b);
	void getBubbles(std::vector<Bubble> &b) const;

	void setOffsets(const std::vector<int> &o);
	void setOffsetsFromDevice(CudaContainer<int> &o);
	void getOffsets(std::vector<int> &o) const;
	
	Bubble getBubble(size_t i) const;
	
    private:
	
	std::shared_ptr<Env> env;
	
	std::vector<Bubble> bubbles;
	std::vector<int> offsets;
    };
}
