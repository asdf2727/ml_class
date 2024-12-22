#pragma once

#include <iostream>

template <typename T>
void read(std::istream &in, T *val, const size_t count = 1) {
	in.read(reinterpret_cast<char *>(val), sizeof(T) * count);
}
template <typename T>
T read(std::istream &in, const size_t size = sizeof(T)) {
	T val;
	read(in, &val, size);
	return val;
}

template <typename T>
void write(std::ostream &out, const T *val, const size_t count = 1) {
	out.write(reinterpret_cast<const char *>(val), sizeof(T) * count);
}