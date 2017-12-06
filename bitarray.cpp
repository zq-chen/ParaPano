#include "bitarray.h"

BitArray::BitArray() {
	num_cells = 4;
	for (int i = 0; i < num_cells; i++) {
		value[i] = 0;
	}
}

/* valid pos: 0 <= pos <= 255 */
void BitArray::set(int pos, bool cond) {
	if (cond) {
		int id = pos / num_cells;
		int real_pos = pos % 64;
		value[id] |= (1 << real_pos);
	}
}