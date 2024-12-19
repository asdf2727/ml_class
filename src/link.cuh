#pragma once

template <typename T>
struct link {
private:
	T *data;

public:
	link (T &val) { data = *val; }
	T &operator=(T &val) { return *data = val; }

	friend ::operator T *(link *rhs) {
		return rhs.data;
	}

	T &get() {
		return *data;
	}
	operator T &() {
		return get();
	}
	const T &get() const {
		return *data;
	}
	operator const T &() const {
		return get();
	}
};