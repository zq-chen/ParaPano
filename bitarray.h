#pragma once

#include <stdint.h>

class BitArray {
public:
	Bitarray();
	virtual ~Bitarray() {}

	void set(int pos, bool cond);

	int64_t value[4];

	int num_cells;
};
