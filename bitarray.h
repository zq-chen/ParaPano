#pragma once

#include <stdint.h>

class BitArray {
public:
	BitArray();
	virtual ~BitArray() {}

	void set(int pos, bool cond);

	int64_t value[4];

	int num_cells;
};
