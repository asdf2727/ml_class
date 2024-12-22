#pragma once

#include <functional>

template <typename T>
class lazy {
	bool valid = false;
	T *data = nullptr;
	std::function <void  (T *&)> build;

	T *get () {
		if (!valid) validate();
		return data;
	}

public:
	explicit lazy (std::function <void  (T *&)> build) : build(std::move(build)) {}

	lazy () : data(nullptr), build(nullptr) {}

	lazy (const lazy &other) = delete;
	lazy &operator= (const lazy &other) = delete;
	lazy (lazy &&other) noexcept :
		valid(other.valid),
		data(other.data),
		build(std::move(other.build)) {
		other.valid = false;
		other.data = nullptr;
	}
	lazy &operator= (lazy &&other) noexcept {
		valid = other.valid;
		data = other.data;
		other.valid = false;
		other.data = nullptr;
		return *this;
	}
	~lazy () { delete data; }

	void validate();
	void invalidate () { valid = false; }
	[[nodiscard]] bool isValid () const { return valid; }

	operator const T* () const { return data; }
	operator T* () noexcept { return get(); }
	const T *operator-> () const { return data; }
	T *operator-> () noexcept { return get(); }
	const T &operator* () const & { return *data; }
	T &operator* () & { return *get(); }
	const T &&operator* () const && { return std::move(*data); }
	T &&operator* () && { return std::move(*get()); }
};

template <typename T>
void lazy <T>::validate() {
	delete data;
	data = nullptr;
	build(data);
	valid = true;
}
