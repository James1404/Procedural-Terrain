#pragma once
#include "common.h"

namespace math
{
	template<typename T>
	struct vector<2,T>
	{
		T x, y;

		inline constexpr vector() = default;
		inline ~vector() = default;

		template<typename X, typename Y>
		inline constexpr vector(X x, Y y)
			: x(static_cast<T>(x)),
			  y(static_cast<T>(y))
		{}

		template<typename V>
		inline constexpr vector(V v)
			: x(static_cast<T>(v)),
			  y(static_cast<T>(v))
		{}

		inline constexpr vector(const vector<2,T>& other)
			: vector(other.x, other.y)
		{}

		inline constexpr vector<2,T>& operator=(const vector<2,T>& other)
		{
			if(this == &other)
			{
				return *this;
			}
			
			x = other.x;
			y = other.y;

			return *this;
		}

		// unaray operators
		inline vector<2,T> operator+=(const vector<2,T>& other)
		{
			x += other.x;
			y += other.y;

			return *this;
		}

		inline vector<2,T> operator-=(const vector<2,T>& other)
		{
			x -= other.x;
			y -= other.y;

			return *this;
		}

		inline vector<2,T> operator*=(const vector<2,T>& other)
		{
			x *= other.x;
			y *= other.y;

			return *this;
		}

		inline vector<2,T> operator/=(const vector<2,T>& other)
		{
			x /= other.x;
			y /= other.y;

			return *this;
		}

		template<typename S>
		inline vector<2,T> operator+=(const S& other)
		{
			x += other;
			y += other;

			return *this;
		}

		template<typename S>
		inline vector<2,T> operator-=(const S& other)
		{
			x -= other;
			y -= other;

			return *this;
		}

		template<typename S>
		inline vector<2,T> operator*=(const S& other)
		{
			x *= other;
			y *= other;

			return *this;
		}

		template<typename S>
		inline vector<2,T> operator/=(const S& other)
		{
			x /= other;
			y /= other;

			return *this;
		}
	};

	// binary operators
	template<typename T>
	inline bool operator==(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
		return lhs.x == rhs.x &&
			   lhs.y == rhs.y;
	}

	template<typename T>
	inline bool operator!=(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
		return lhs.x != rhs.x &&
			   lhs.y != rhs.y;
	}

	template<typename T>
	inline vector<2,T> operator+(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
		return vector<2,T>(lhs.x + rhs.x,
						   lhs.y + rhs.y);
	}

	template<typename T>
	inline vector<2,T> operator-(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
		return vector<2,T>(lhs.x - rhs.x,
						   lhs.y - rhs.y);
	}

	template<typename T>
	inline vector<2,T> operator*(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
		return vector<2,T>(lhs.x * rhs.x,
						   lhs.y * rhs.y);
	}

	template<typename T>
	inline vector<2,T> operator/(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
		return vector<2,T>(lhs.x / rhs.x,
						   lhs.y / rhs.y);
	}

	// Right hand scalar
	template<typename T, typename S>
	inline vector<2,T> operator+(const vector<2,T>& lhs, const S& rhs)
	{
		return vector<2,T>(lhs.x + rhs,
						   lhs.y + rhs);
	}

	template<typename T, typename S>
	inline vector<2,T> operator-(const vector<2,T>& lhs, const S& rhs)
	{
		return vector<2,T>(lhs.x - rhs,
						   lhs.y - rhs);
	}

	template<typename T, typename S>
	inline vector<2,T> operator*(const vector<2,T>& lhs, const S& rhs)
	{
		return vector<2,T>(lhs.x * rhs,
						   lhs.y * rhs);
	}

	template<typename T, typename S>
	inline vector<2,T> operator/(const vector<2,T>& lhs, const S& rhs)
	{
		return vector<2,T>(lhs.x / rhs,
						   lhs.y / rhs);
	}

	// Left hand scalar
	template<typename T, typename S>
	inline vector<2,T> operator+(const S& lhs, const vector<2,T>& rhs)
	{
		return vector<2,T>(lhs + rhs.x,
						   lhs + rhs.y);
	}

	template<typename T, typename S>
	inline vector<2,T> operator-(const S& lhs, const vector<2,T>& rhs)
	{
		return vector<2,T>(lhs - rhs.x,
						   lhs - rhs.y);
	}

	template<typename T, typename S>
	inline vector<2,T> operator*(const S& lhs, const vector<2,T>& rhs)
	{
		return vector<2,T>(lhs * rhs.x,
						   lhs * rhs.y);
	}

	template<typename T, typename S>
	inline vector<2,T> operator/(const S& lhs, const vector<2,T>& rhs)
	{
		return vector<2,T>(lhs / rhs.x,
						   lhs / rhs.y);
	}

	// Functions
	template<typename T>
	inline float distance(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec2 v1(lhs.x, lhs.y);
		glm::vec2 v2(rhs.x, rhs.y);
		return glm::distance(v1,v2);
#else
		vector<2,T> v = lhs - rhs;
		return basic::sqrt(((v.x*v.x) + (v.y*v.y));
#endif
	}

	template<typename T>
	inline float length(const vector<2,T>& v)
	{
#ifdef USE_GLM
		glm::vec2 vec(v.x, v.y);
		return glm::length(vec);
#else
		return basic::sqrt((v.x*v.x) + (v.y*v.y));
#endif
	}

	template<typename T>
	inline float dot(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec2 v1(lhs.x, lhs.y);
		glm::vec2 v2(rhs.x, rhs.y);
		return glm::dot(v1,v2);
#else
		vector<2,T> l = length(lhs);
		vector<2,T> r = length(rhs);

		// TODO: this probably doesnt work...???
		return (lhs.x * rhs.x) + (lhs.y * rhs.y);
#endif
	}

	template<typename T>
	inline vector<2,T> cross(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec2 v1(lhs.x, lhs.y);
		glm::vec2 v2(rhs.x, rhs.y);
		glm::vec2 value = glm::cross(v1,v2);
		return vector<2,T>(value.x, value.y);
#endif
	}

	template<typename T>
	inline vector<2,T> normalize(const vector<2,T>& v)
	{
#ifdef USE_GLM
		glm::vec2 vec(v.x, v.y);
		glm::vec2 value = glm::normalize(vec);
		return vector<2,T>(value.x, value.y);
#endif
	}

	template<typename T>
	inline vector<2,T> reflect(const vector<2,T>& lhs, const vector<2,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec2 v1(lhs.x, lhs.y);
		glm::vec2 v2(rhs.x, rhs.y);
		glm::vec2 value = glm::reflect(v1, v2);
		return vector<2,T>(value.x, value.y);
#endif
	}
}

// TODO: Remove vector3_to_glm_vec3
template<typename T>
inline glm::vec2 vector2_to_glm_vec2(math::vector<2,T>& v)
{
#ifdef USE_GLM
	return glm::vec2(v.x,v.y);
#else
	return v;
#endif
}
