#include "BubbleManager.h"

using namespace cubble;

BubbleManager::BubbleManager(std::shared_ptr<Env> e)
{
    env = e;
}

BubbleManager::~BubbleManager()
{}

void BubbleManager::setBubbles(const std::vector<Bubble> &b)
{
    bubbles = b;
}

void BubbleManager::setBubblesFromDevice(CudaContainer<Bubble> &b)
{
    b.copyDeviceDataToVec(bubbles);
}

void BubbleManager::getBubbles(std::vector<Bubble> &b) const
{
    b = bubbles;
}

void BubbleManager::getBubbles(CudaContainer<Bubble> &b) const
{
    b.copyVecToHost(bubbles);
}

void BubbleManager::setIndicesFromDevice(CudaContainer<int> &i)
{
    i.copyDeviceDataToVec(indices);
}

void BubbleManager::getIndices(std::vector<int> &i) const
{
    i = indices;
}

void BubbleManager::getIndices(CudaContainer<int> &i) const
{
    i.copyVecToHost(indices);
}

void BubbleManager::setCellsFromDevice(CudaContainer<Cell> &c)
{
    c.copyDeviceDataToVec(cells);
}

void BubbleManager::getCells(std::vector<Cell> &c) const
{
    c = cells;
}

void BubbleManager::getCells(CudaContainer<Cell> &c) const
{
    c.copyVecToHost(cells);
}

Bubble BubbleManager::getBubble(size_t i) const
{
    assert(i < bubbles.size());
    return bubbles[i];
}

void BubbleManager::getIndicesFromNeighborCells(std::vector<int> &indVec,
						int index,
						bool firstCall) const
{
    assert(index < (int)cells.size());
    indVec.clear();

    Cell cell = cells[index];

    for (int i = cell.offset; i < cell.offset + cell.size; ++i)
	indVec.push_back(indices[i]);

    if (firstCall)
    {
	for (int i = 0; i < numNeighbors; ++i)
	{
	    std::vector<int> temp;
	    int j = getNthNeighborOfIndex(index, i);
	    getIndicesFromNeighborCells(temp, j, false);
	    indVec.insert(indVec.end(), temp.begin(), temp.end());
	}
    }
}

int BubbleManager::getNthNeighborOfIndex(int index, int n) const
{
    assert(n < numNeighbors);
    
    int numCellsPerDim = 0;

    auto indexToIndexVec = [](int index, int numCellsPerDim) -> ivec
	{
	    ivec indexVec;
	    indexVec.x = index % numCellsPerDim;
	    indexVec.y = (index % (numCellsPerDim * numCellsPerDim)) / numCellsPerDim;
	    indexVec.z = index / (numCellsPerDim * numCellsPerDim);

	    return indexVec;
	};

    auto indexVecToIndex = [](ivec indexVec, int numCellsPerDim) -> int
	{
	    indexVec = (indexVec + numCellsPerDim) % numCellsPerDim;
	    int index = indexVec.z * numCellsPerDim * numCellsPerDim
	    + indexVec.y * numCellsPerDim
	    + indexVec.x;

	    return index;
	};
    
#if NUM_DIM == 3
    numCellsPerDim = std::cbrt(cells.size());
    assert(numCellsPerDim * numCellsPerDim * numCellsPerDim == cells.size());
    
    ivec idxVec = indexToIndexVec(index, numCellsPerDim);

    switch (n)
    {
    case 0:
        idxVec += ivec(-1, 1, 0);
	break;
    case 1:
        idxVec += ivec(-1, 0, 0);
	break;
    case 2:
        idxVec += ivec(-1, -1, 0);
	break;
    case 3:
        idxVec += ivec(0, -1, 0);
	break;
    case 4:
        idxVec += ivec(-1, 1, -1);
	break;
    case 5:
        idxVec += ivec(0, 1, -1);
	break;
    case 6:
        idxVec += ivec(1, 1, -1);
	break;
    case 7:
        idxVec += ivec(-1, 0, -1);
	break;
    case 8:
        idxVec += ivec(0, 0, -1);
	break;
    case 9:
        idxVec += ivec(1, 0, -1);
	break;
    case 10:
        idxVec += ivec(-1, -1, -1);
	break;
    case 11:
        idxVec += ivec(0, -1, -1);
	break;
    case 12:
        idxVec += ivec(1, -1, -1);
	break;
    default:
	assert(false && "Index out of bounds!");
	break;
    }
#else
    numCellsPerDim = std::sqrt(cells.size());
    assert(numCellsPerDim * numCellsPerDim == cells.size());
    
    ivec idxVec = indexToIndexVec(index, numCellsPerDim);
    
    switch (n)
    {
    case 0:
        idxVec += ivec(-1, 1, 0);
	break;
    case 1:
        idxVec += ivec(-1, 0, 0);
	break;
    case 2:
        idxVec += ivec(-1, -1, 0);
	break;
    case 3:
        idxVec += ivec(0, -1, 0);
	break;
    default:
	assert(false && "Index out of bounds!");
	break;
    }
#endif

    return indexVecToIndex(idxVec, numCellsPerDim);
}
