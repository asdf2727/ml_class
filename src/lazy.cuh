#pragma once

template <typename T, class C>
struct lazy {
private:
	T *data = nullptr;
	void (C::*build)(const T *&data) const;

public:
	explicit lazy(void (C::*build)(const T *&data) const) : build(build) {}

	~lazy() {
		delete data;
	}

	friend ::operator T *(lazy *rhs) {
		return rhs.data;
	}

	T &get() {
		if (data == nullptr) {
			build(data);
		}
		return *data;
	}
	operator T &() {
		return get();
	}
	const T &get() const {
		if (data == nullptr) {
			build(data);
		}
		return *data;
	}
	operator const T &() const {
		return get();
	}

	void invalidate() {
		delete data;
		data = nullptr;
	}
};