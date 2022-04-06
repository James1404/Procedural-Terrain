#pragma once
#include "common.h"

namespace math
{
	template<typename T>
	struct vector<3, T>
	{
		T x, y, z;

		// constructors
		constexpr vector()
			: x(0), y(0), z(0)
		{}

		template<typename X, typename Y, typename Z>
		constexpr vector(X x, Y y, Z z)
			: x(static_cast<T>(x)),
			  y(static_cast<T>(y)),
			  z(static_cast<T>(z))
		{}

		template<typename V>
		constexpr vector(V v)
			: x(static_cast<T>(v)),
			  y(static_cast<T>(v)),
			  z(static_cast<T>(v))
		{}

		constexpr vector(const vector<3,T>& other)
			: vector(other.x, other.y, other.z)
		{}

		constexpr vector<3,T>& operator=(const vector<3,T>& other)
		{
			if(this == &other)
			{
				return *this;
			}
			
			x = other.x;
			y = other.y;
			z = other.z;

			return *this;
		}

		~vector() = default;

		// unaray operators
		vector<3,T> operator+=(const vector<3,T>& other)
		{
			x += other.x;
			y += other.y;
			z += other.z;

			return *this;
		}

		vector<3,T> operator-=(const vector<3,T>& other)
		{
			x -= other.x;
			y -= other.y;
			z -= other.z;

			return *this;
		}

		vector<3,T> operator*=(const vector<3,T>& other)
		{
			x *= other.x;
			y *= other.y;
			z *= other.z;

			return *this;
		}

		vector<3,T> operator/=(const vector<3,T>& other)
		{
			x /= other.x;
			y /= other.y;
			z /= other.z;

			return *this;
		}

		template<typename S>
		vector<3,T> operator+=(const S& other)
		{
			x += other;
			y += other;
			z += other;

			return *this;
		}

		template<typename S>
		vector<3,T> operator-=(const S& other)
		{
			x -= other;
			y -= other;
			z -= other;

			return *this;
		}

		template<typename S>
		vector<3,T> operator*=(const S& other)
		{
			x *= other;
			y *= other;
			z *= other;

			return *this;
		}

		template<typename S>
		vector<3,T> operator/=(const S& other)
		{
			x /= other;
			y /= other;
			z /= other;

			return *this;
		}
	};

	// binary operators
	template<typename T>
	bool operator==(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
		return lhs.x == rhs.x &&
			   lhs.y == rhs.y &&
			   lhs.z == rhs.z;
	}

	template<typename T>
	bool operator!=(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
		return lhs.x != rhs.x &&
			   lhs.y != rhs.y &&
			   lhs.z != rhs.z;
	}

	template<typename T>
	vector<3,T> operator+(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
		return vector<3,T>(lhs.x + rhs.x,
						  lhs.y + rhs.y,
						  lhs.z + rhs.z);
	}

	template<typename T>
	vector<3,T> operator-(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
		return vector<3,T>(lhs.x - rhs.x,
						  lhs.y - rhs.y,
						  lhs.z - rhs.z);
	}

	template<typename T>
	vector<3,T> operator*(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
		return vector<3,T>(lhs.x * rhs.x,
						  lhs.y * rhs.y,
						  lhs.z * rhs.z);
	}

	template<typename T>
	vector<3,T> operator/(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
		return vector<3,T>(lhs.x / rhs.x,
						  lhs.y / rhs.y,
						  lhs.z / rhs.z);
	}

	// Right hand scalar
	template<typename T, typename S>
	vector<3,T> operator+(const vector<3,T>& lhs, const S& rhs)
	{
		return vector<3,T>(lhs.x + rhs,
						   lhs.y + rhs,
						   lhs.z + rhs);
	}

	template<typename T, typename S>
	vector<3,T> operator-(const vector<3,T>& lhs, const S& rhs)
	{
		return vector<3,T>(lhs.x - rhs,
						   lhs.y - rhs,
						   lhs.z - rhs);
	}

	template<typename T, typename S>
	vector<3,T> operator*(const vector<3,T>& lhs, const S& rhs)
	{
		return vector<3,T>(lhs.x * rhs,
						   lhs.y * rhs,
						   lhs.z * rhs);
	}

	template<typename T, typename S>
	vector<3,T> operator/(const vector<3,T>& lhs, const S& rhs)
	{
		return vector<3,T>(lhs.x / rhs,
						   lhs.y / rhs,
						   lhs.z / rhs);
	}

	// Left hand scalar
	template<typename T, typename S>
	vector<3,T> operator+(const S& lhs, const vector<3,T>& rhs)
	{
		return vector<3,T>(lhs + rhs.x,
						   lhs + rhs.y,
						   lhs + rhs.z);
	}

	template<typename T, typename S>
	vector<3,T> operator-(const S& lhs, const vector<3,T>& rhs)
	{
		return vector<3,T>(lhs - rhs.x,
						   lhs - rhs.y,
						   lhs - rhs.z);
	}

	template<typename T, typename S>
	vector<3,T> operator*(const S& lhs, const vector<3,T>& rhs)
	{
		return vector<3,T>(lhs * rhs.x,
						   lhs * rhs.y,
						   lhs * rhs.z);
	}

	template<typename T, typename S>
	vector<3,T> operator/(const S& lhs, const vector<3,T>& rhs)
	{
		return vector<3,T>(lhs / rhs.x,
						   lhs / rhs.y,
						   lhs / rhs.z);
	}

	// Functions
	template<typename T>
	float distance(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec3 v1(lhs.x, lhs.y, lhs.z);
		glm::vec3 v2(rhs.x, rhs.y, rhs.z);
		return glm::distance(v1,v2);
#else
		vector<3,T> v = lhs - rhs;
		return basic::sqrt(((v.x*v.x) + (v.y*v.y) + (v.z*v.z)));
#endif
	}

	template<typename T>
	float length(const vector<3,T>& v)
	{
#ifdef USE_GLM
		glm::vec3 vec(v.x, v.y, v.z);
		return glm::length(vec);
#else
		return basic::sqrt((v.x*v.x) + (v.y*v.y) + (v.z*v.z));
#endif
	}

	template<typename T>
	float dot(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec3 v1(lhs.x, lhs.y, lhs.z);
		glm::vec3 v2(rhs.x, rhs.y, rhs.z);
		return glm::dot(v1,v2);
#else
		vector<3,T> l = length(lhs);
		vector<3,T> r = length(rhs);

		// TODO: this defo doesnt work...???
		return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
#endif
	}

	template<typename T>
	vector<3,T> cross(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec3 v1(lhs.x, lhs.y, lhs.z);
		glm::vec3 v2(rhs.x, rhs.y, rhs.z);
		glm::vec3 value = glm::cross(v1,v2);
		return vector<3,T>(value.x, value.y, value.z);
#endif
	}

	template<typename T>
	vector<3,T> normalize(const vector<3,T>& v)
	{
#ifdef USE_GLM
		glm::vec3 vec(v.x, v.y, v.z);
		glm::vec3 value = glm::normalize(vec);
		return vector<3,T>(value.x, value.y, value.z);
#endif
	}

	template<typename T>
	vector<3,T> reflect(const vector<3,T>& lhs, const vector<3,T>& rhs)
	{
#ifdef USE_GLM
		glm::vec3 v1(lhs.x, lhs.y, lhs.z);
		glm::vec3 v2(rhs.x, rhs.y, rhs.z);
		glm::vec3 value = glm::reflect(v1, v2);
		return vector<3,T>(value.x, value.y, value.z);
#endif
	}
}

// TODO: Remove vector3_to_glm_vec3
template<typename T>
glm::vec3 vector3_to_glm_vec3(math::vector<3,T>& v)
{
#ifdef USE_GLM
	return glm::vec3(v.x,v.y,v.z);
#else
	return v;
#endif
}
