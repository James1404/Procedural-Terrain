#pragma once
#include "common.h"
#include "vector3.h"
#include "vector2.h"

using vector3f = math::vector<3,float>;
using vector3d = math::vector<3,double>;
using vector3i = math::vector<3,int>;
using vector3u = math::vector<3,unsigned int>;

//using vec2f = glm::vec2;
//using vec2i = glm::ivec2;
//using vec2d = glm::dvec2;
//using vec2u = glm::uvec2;
using vector2f = math::vector<2,float>;
using vector2i = math::vector<2,double>;
using vector2d = math::vector<2,int>;
using vector2u = math::vector<2,unsigned int>;

using matrix4 = glm::mat4;
using matrix3 = glm::mat3;
using matrix2 = glm::mat2;
