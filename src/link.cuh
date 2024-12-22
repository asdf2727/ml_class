#pragma once

// TODO
template <typename T>
struct link {
private:
	T *data;

public:
	explicit link (T &val) : data(&val) {}
	link &operator= (T &val) {
		data = &val;
		return *this;
	}

	link() : data(nullptr) {}

	link (const link &other) = default;
	link &operator= (const link &other) = default;
	~link () = default;

	const T *operator-> () const { return data; }
	T *operator-> () noexcept { return data; }
	const T &operator* () const & { return *data; }
	T &operator* () & { return *data; }
	const T &&operator* () const && { return std::move(*data); }
	T &&operator* () && { return std::move(*data); }
};
