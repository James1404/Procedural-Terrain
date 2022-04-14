#pragma once
#include "common.h"

namespace math
{
	template<typename T>
	struct vector<4, T>
	{
		T x, y, z, w;

		// constructors
		inline constexpr vector() = default;
		inline ~vector() = default;

		template<typename X, typename Y, typename Z, typename W>
		inline constexpr vector(X x, Y y, Z z, W w)
			: x(static_cast<T>(x)),
			  y(static_cast<T>(y)),
			  z(static_cast<T>(z)),
			  w(static_cast<T>(w))
		{}

		template<typename V>
		inline constexpr vector(V v)
			: x(static_cast<T>(v)),
			  y(static_cast<T>(v)),
			  z(static_cast<T>(v)),
			  w(static_cast<T>(v))
		{}

		inline constexpr vector(const vector<4,T>& other)
			: vector(other.x, other.y, other.z, other.w)
		{}

		inline constexpr vector<4,T>& operator=(const vector<4,T>& other)
		{
			if(this == &other)
			{
				return *this;
			}
			
			x = other.x;
			y = other.y;
			z = other.z;
			w = other.w;

			return *this;
		}

		// unaray operators
		inline vector<4,T> operator+=(const vector<4,T>& other)
		{
			x += other.x;
			y += other.y;
			z += other.z;
			w += other.w;

			return *this;
		}

		inline vector<4,T> operator-=(const vector<4,T>& other)
		{
			x -= other.x;
			y -= other.y;
			z -= other.z;
			w -= other.w;

			return *this;
		}

		inline vector<4,T> operator*=(const vector<4,T>& other)
		{
			x *= other.x;
			y *= other.y;
			z *= other.z;
			w *= other.w;

			return *this;
		}

		inline vector<4,T> operator/=(const vector<4,T>& other)
		{
			x /= other.x;
			y /= other.y;
			z /= other.z;
			w /= other.w;

			return *this;
		}

		template<typename S>
		inline vector<4,T> operator+=(const S& other)
		{
			x += other;
			y += other;
			z += other;
			w += other;

			return *this;
		}

		template<typename S>
		inline vector<4,T> operator-=(const S& other)
		{
			x -= other;
			y -= other;
			z -= other;
			w -= other;

			return *this;
		}

		template<typename S>
		inline vector<4,T> operator*=(const S& other)
		{
			x *= other;
			y *= other;
			z *= other;
			w *= other;

			return *this;
		}

		template<typename S>
		inline vector<4,T> operator/=(const S& other)
		{
			x /= other;
			y /= other;
			z /= other;
			w /= other;

			return *this;
		}
	};

	// binary operators
	template<typename T>
	inline bool operator==(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
		return lhs.x == rhs.x &&
			   lhs.y == rhs.y &&
			   lhs.z == rhs.z &&
			   lhs.w == rhs.w;
	}

	template<typename T>
	inline bool operator!=(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
		return lhs.x != rhs.x &&
			   lhs.y != rhs.y &&
			   lhs.z != rhs.z &&
			   lhs.w != rhs.w;
	}

	template<typename T>
	inline vector<4,T> operator+(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
		return vector<4,T>(lhs.x + rhs.x,
						   lhs.y + rhs.y,
						   lhs.z + rhs.z,
						   lhs.w + rhs.w);
	}

	template<typename T>
	inline vector<4,T> operator-(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
		return vector<4,T>(lhs.x - rhs.x,
						   lhs.y - rhs.y,
						   lhs.z - rhs.z,
						   lhs.w - rhs.w);
	}

	template<typename T>
	inline vector<4,T> operator*(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
		return vector<4,T>(lhs.x * rhs.x,
						   lhs.y * rhs.y,
						   lhs.z * rhs.z,
						   lhs.w * rhs.w);
	}

	template<typename T>
	inline vector<4,T> operator/(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
		return vector<4,T>(lhs.x / rhs.x,
						   lhs.y / rhs.y,
						   lhs.z / rhs.z,
						   lhs.w / rhs.w);
	}

	// Right hand scalar
	template<typename T, typename S>
	inline vector<4,T> operator+(const vector<4,T>& lhs, const S& rhs)
	{
		return vector<4,T>(lhs.x + rhs,
						   lhs.y + rhs,
						   lhs.z + rhs,
						   lhs.w + rhs);
	}

	template<typename T, typename S>
	inline vector<4,T> operator-(const vector<4,T>& lhs, const S& rhs)
	{
		return vector<4,T>(lhs.x - rhs,
						   lhs.y - rhs,
						   lhs.z - rhs,
						   lhs.w - rhs);
	}

	template<typename T, typename S>
	inline vector<4,T> operator*(const vector<4,T>& lhs, const S& rhs)
	{
		return vector<4,T>(lhs.x * rhs,
						   lhs.y * rhs,
						   lhs.z * rhs,
						   lhs.w * rhs);
	}

	template<typename T, typename S>
	inline vector<4,T> operator/(const vector<4,T>& lhs, const S& rhs)
	{
		return vector<4,T>(lhs.x / rhs,
						   lhs.y / rhs,
						   lhs.z / rhs,
						   lhs.w / rhs);
	}

	// Left hand scalar
	template<typename T, typename S>
	inline vector<4,T> operator+(const S& lhs, const vector<4,T>& rhs)
	{
		return vector<4,T>(lhs + rhs.x,
						   lhs + rhs.y,
						   lhs + rhs.z,
						   lhs + rhs.w);
	}

	template<typename T, typename S>
	inline vector<4,T> operator-(const S& lhs, const vector<4,T>& rhs)
	{
		return vector<4,T>(lhs - rhs.x,
						   lhs - rhs.y,
						   lhs - rhs.z,
						   lhs - rhs.w);
	}

	template<typename T, typename S>
	inline vector<4,T> operator*(const S& lhs, const vector<4,T>& rhs)
	{
		return vector<4,T>(lhs * rhs.x,
						   lhs * rhs.y,
						   lhs * rhs.z,
						   lhs * rhs.w);
	}

	template<typename T, typename S>
	inline vector<4,T> operator/(const S& lhs, const vector<4,T>& rhs)
	{
		return vector<4,T>(lhs / rhs.x,
						   lhs / rhs.y,
						   lhs / rhs.z,
						   lhs / rhs.w);
	}

	// Functions
	template<typename T>
	inline float distance(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec4 v1(lhs.x, lhs.y, lhs.z, lhs.w);
		glm::vec4 v2(rhs.x, rhs.y, rhs.z, rhs.w);
		return glm::distance(v1,v2);
#else
		vector<4,T> v = lhs - rhs;
		return basic::sqrt((v.x*v.x) + (v.y*v.y) + (v.z*v.z) + (v.w*v.w));
#endif
	}

	template<typename T>
	inline float length(const vector<4,T>& v)
	{
#ifdef USE_GLM
		glm::vec4 vec(v.x, v.y, v.z, v.w);
		return glm::length(vec);
#else
		return basic::sqrt((v.x*v.x) + (v.y*v.y) + (v.z*v.z) + (v.w*v.w));
#endif
	}

	template<typename T>
	inline float dot(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec4 v1(lhs.x, lhs.y, lhs.z, lhs.w);
		glm::vec4 v2(rhs.x, rhs.y, rhs.z, rhs.w);
		return glm::dot(v1,v2);
#else
		vector<4,T> l = length(lhs);
		vector<4,T> r = length(rhs);

		// TODO: this defo doesnt work...???
		return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z) + (lhs.w * rhs.w);
#endif
	}

	template<typename T>
	inline vector<4,T> cross(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec4 v1(lhs.x, lhs.y, lhs.z, lhs.w);
		glm::vec4 v2(rhs.x, rhs.y, rhs.z, rhs.w);
		glm::vec4 value = glm::cross(v1,v2);
		return vector<4,T>(value.x, value.y, value.z, value.w);
#endif
	}

	template<typename T>
	inline vector<4,T> normalize(const vector<4,T>& v)
	{
#ifdef USE_GLM
		glm::vec4 vec(v.x, v.y, v.z, v.w);
		glm::vec4 value = glm::normalize(vec);
		return vector<4,T>(value.x, value.y, value.z, value.w);
#endif
	}

	template<typename T>
	inline vector<4,T> reflect(const vector<4,T>& lhs, const vector<4,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec4 v1(lhs.x, lhs.y, lhs.z, lhs.w);
		glm::vec4 v2(rhs.x, rhs.y, rhs.z, rhs.w);
		glm::vec4 value = glm::reflect(v1, v2);
		return vector<4,T>(value.x, value.y, value.z, value.w);
#endif
	}
}

// TODO: Remove vector3_to_glm_vec3
template<typename T>
inline glm::vec4 vector4_to_glm_vec4(math::vector<4,T>& v)
{
#ifdef USE_GLM
	return glm::vec4(v.x,v.y,v.z,v.w);
#else
	return v;
#endif
}
