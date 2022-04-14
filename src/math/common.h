#pragma once
#include <glm.hpp>
#include <mat4x4.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

#include <cmath>

#define USE_GLM

// TODO: Completely remove glm

namespace math
{
	namespace basic
	{
		template<typename T>
		inline T sqrt(T t)
		{
			// TODO: This is temporary will be replaced 
			// with proper implementation

			return std::sqrt(t);
		}

		template<typename T>
		inline T cos(T t)
		{
			return std::cos(t);
		}

		template<typename T>
		inline T sin(T t)
		{

			return std::sin(t);
		}

		template<typename T>
		inline T tan(T t)
		{
			return std::tan(t);
		}
	}

	template<size_t L, typename T> struct vector;
	template<size_t LA, size_t LB, typename T> struct matrix;
}
