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
		T sqrt(T t)
		{
			// TODO: This is temporary will be replaced 
			// with proper implementation

			return std::sqrt(t);
		}

		template<typename T>
		T cos(T t)
		{
			return std::cos(t);
		}

		template<typename T>
		T sin(T t)
		{

			return std::sin(t);
		}

		template<typename T>
		T tan(T t)
		{
			return std::tan(t);
		}
	}

	template<size_t L, typename T> struct vector;
}
