#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>

#pragma comment(lib, "user32")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include "math/math.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// TODO: Replace FastNoiseLite with custom solution.
#include "FastNoiseLite.h"

#include <chrono>
#include <random>
#include <functional>
#include <cassert>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <memory>
#include <iostream>

template <typename ...Args>
inline void log(Args&& ..._args)
{
	(std::cout << ... << _args);
	std::cout << std::endl;
}

#define log_info(...) log("", __VA_ARGS__);
#define log_warning(...) log("'", __FILE__, "' Warning at Line(", __LINE__, "): ", __VA_ARGS__);
#define log_error(...) log("'", __FILE__, "' Error at Line(", __LINE__, "): ", __VA_ARGS__);

#define USE_RAW_INPUT

constexpr int DEFAULT_SCREEN_WIDTH = 1280; 
constexpr int DEFAULT_SCREEN_HEIGHT = 720; 

int screen_width = DEFAULT_SCREEN_WIDTH;
int screen_height = DEFAULT_SCREEN_HEIGHT;

IDXGISwapChain* swapchain;
ID3D11Device* dev;
ID3D11DeviceContext* devcon;
ID3D11RenderTargetView* backbuffer;
ID3D11DepthStencilView* depthbuffer;

constexpr float NEAR_CLIP_PLANE = 0.1f;
constexpr float FAR_CLIP_PLANE = 1000.0f;

constexpr vector3f WORLD_UP = vector3f(0, 1, 0);

constexpr float CAMERA_MOVE_SPEED_NORMAL = 3.0f;
constexpr float CAMERA_MOVE_SPEED_SLOW = 0.5f;
constexpr float CAMERA_MOVE_SPEED_FAST = 30.0f;

constexpr float CAMERA_ROTATION_SPEED = 0.05f;

matrix4 projection_matrix, view_matrix;

inline vector3f sqrt_magnitude(vector3f vec)
{
	float length = sqrt((vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z));
	if (length != 0)
	{
		vec /= length;
	}

	return vec;
}

inline vector3f move_towards(const vector3f pos, const vector3f target, const float step)
{
	const vector3f delta = target - pos;
	const float len2 = math::dot(delta, delta);

	if (len2 < step * step)
	{
		return target;
	}

	const vector3f direction = delta / glm::sqrt(len2);

	return pos + step * direction;
}

struct transform_t
{
	vector3f position, rotation, scale;

	transform_t()
		: position(vector3f(0)), rotation(vector3f(0)), scale(vector3f(1)) {}

	transform_t(vector3f position, vector3f rotation, vector3f scale)
		: position(position), rotation(rotation), scale(scale) {}

	matrix4 get_matrix()
	{
		matrix4 model = matrix4(1.0f);
		model = glm::translate(model, vector3_to_glm_vec3(position));

		model = glm::rotate(model, glm::radians(rotation.x), glm::vec3(1, 0, 0));
		model = glm::rotate(model, glm::radians(rotation.y), glm::vec3(0, 1, 0));
		model = glm::rotate(model, glm::radians(rotation.z), glm::vec3(0, 0, 1));

		model = glm::scale(model, vector3_to_glm_vec3(scale));

		return model;
	}
};

struct shader_t 
{
	ID3D11VertexShader* pVs;
	ID3D11PixelShader* pPs;
	ID3D11InputLayout* pLayout;
	ID3D11Buffer* pConstantBuffer;

	~shader_t()
	{
		pConstantBuffer->Release();

		pLayout->Release();
		pVs->Release();
		pPs->Release();
	}

	void load_from_file(std::string path)
	{
		ID3DBlob* vs;
		ID3DBlob* ps;
		ID3DBlob* errorBlob;

		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG;

		std::wstring stemp = std::wstring(path.begin(), path.end());
		LPCWSTR sw = stemp.c_str();

		{
			HRESULT hr = D3DCompileFromFile(sw, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "VSMain", "vs_4_0", flags, 0, &vs, &errorBlob);
			assert(SUCCEEDED(hr) && "Vertex Shader failed to compile");
		}

		{
			HRESULT hr = D3DCompileFromFile(sw, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "PSMain", "ps_4_0", flags, 0, &ps, &errorBlob);
			assert(SUCCEEDED(hr) && "Pixel Shader failed to compile");
		}

		assert(errorBlob == NULL);
		assert(vs != NULL);
		assert(ps != NULL);


		dev->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), NULL, &pVs);
		dev->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(), NULL, &pPs);

		devcon->VSSetShader(pVs, 0, 0);
		devcon->PSSetShader(pPs, 0, 0);

		D3D11_INPUT_ELEMENT_DESC ied[] =
		{
			{"POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,							   D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"NORMAL",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"TEXCOORD",  0, DXGI_FORMAT_R32G32_FLOAT,    0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
		};

		dev->CreateInputLayout(ied, 3, vs->GetBufferPointer(), vs->GetBufferSize(), &pLayout);

		devcon->IASetInputLayout(pLayout);

		{
			struct VS_CONSTANT_BUFFER
			{
				matrix4 view_and_projection;
				matrix4 transform;
			};

			D3D11_BUFFER_DESC constant_buffer_desc = {};
			constant_buffer_desc.ByteWidth = sizeof(VS_CONSTANT_BUFFER);
			constant_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			constant_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constant_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			constant_buffer_desc.MiscFlags = 0;
			constant_buffer_desc.StructureByteStride = 0;

			HRESULT hr = dev->CreateBuffer(&constant_buffer_desc, NULL, &pConstantBuffer);

			assert(SUCCEEDED(hr));

			devcon->VSSetConstantBuffers(0, 1, &pConstantBuffer);
		}

		assert(pLayout != NULL);
	}
};

struct texture_t
{
	ID3D11ShaderResourceView* pTexture;
	ID3D11SamplerState* pSampleState;

	~texture_t()
	{
		pTexture->Release();
		pSampleState->Release();
	}

	void set(int slot)
	{
		devcon->PSSetShaderResources(slot, 1, &pTexture);
		devcon->PSSetSamplers(slot, 1, &pSampleState);
	}

	void load_from_file(std::string path)
	{
		int width, height, nrcomponents, desired_channels = 4;
		unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrcomponents, desired_channels);

		assert(data && "Texture failed to load");
		
		if (data)
		{
			DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
			if (nrcomponents == 1)
			{
				format = DXGI_FORMAT_R8_UNORM;
			}
			else if (nrcomponents == 3)
			{
				// if RGB just use RGBA no RGB format;
				format = DXGI_FORMAT_R8G8B8A8_UNORM;
			}
			else if (nrcomponents == 3)
			{
				format = DXGI_FORMAT_R8G8B8A8_UNORM;
			}

			assert(format != DXGI_FORMAT_UNKNOWN);

			ID3D11Texture2D* texture = {};

			{
				// MAKE TEXTURE
				D3D11_TEXTURE2D_DESC texture_desc = {};
				texture_desc.Width = width;
				texture_desc.Height = height;
				texture_desc.MipLevels = 1;
				texture_desc.ArraySize = 1;
				texture_desc.Format = format;
				texture_desc.SampleDesc.Count = 1;
				texture_desc.Usage = D3D11_USAGE_IMMUTABLE;
				texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
				//texture_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				
				D3D11_SUBRESOURCE_DATA init_data = {};
				init_data.pSysMem = data;
				init_data.SysMemPitch = width * 4;
				init_data.SysMemSlicePitch = 0;
				
				HRESULT hr = dev->CreateTexture2D(&texture_desc, &init_data, &texture);

				assert(SUCCEEDED(hr) && "Failed Texture Creation");
			}

			{
				HRESULT hr = dev->CreateShaderResourceView(texture, NULL, &pTexture);

				assert(SUCCEEDED(hr) && "Failed to creater shader resource for texture");
			}

			stbi_image_free(data);

			{
				D3D11_SAMPLER_DESC samplerDesc = {};
				samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
				samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
				samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
				samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
				samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
				samplerDesc.MinLOD = 0;
				samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

				HRESULT hr = dev->CreateSamplerState(&samplerDesc, &pSampleState);

				assert(SUCCEEDED(hr) && "Failed to create sampler");
			}
		}
	}
};

struct vertex_t
{
	vector3f position;
	vector3f normal;
	vector2f tex_coords;
};

struct model_t
{
	std::vector<vertex_t> vertices;
	std::vector<unsigned int> indices;

	ID3D11Buffer* pVBuffer;
	ID3D11Buffer* pIBuffer;

	~model_t()
	{
		vertices.clear();
		indices.clear();

		pVBuffer->Release();
		pIBuffer->Release();
	}

	void process_mesh(aiMesh* mesh)
	{
		for (unsigned int i = 0; i < mesh->mNumVertices; i++)
		{
			vertex_t vertex;

			vector3f vector;
			vector.x = mesh->mVertices[i].x;
			vector.y = mesh->mVertices[i].y;
			vector.z = mesh->mVertices[i].z;
			vertex.position = vector;

			vector.x = mesh->mNormals[i].x;
			vector.y = mesh->mNormals[i].y;
			vector.z = mesh->mNormals[i].z;
			vertex.normal = vector;

			if (mesh->mTextureCoords[0])
			{
				vector2f vec;
				vec.x = mesh->mTextureCoords[0][i].x;
				vec.y = mesh->mTextureCoords[0][i].y;
				vertex.tex_coords = vec;
			}
			else
			{
				vertex.tex_coords = vector2f(0);
			}

			vertices.push_back(vertex);
		}

		int indices_count = (int)indices.size();
		for (unsigned int i = 0; i < mesh->mNumFaces; i++)
		{
			aiFace face = mesh->mFaces[i];
			for (unsigned int j = 0; j < face.mNumIndices; j++)
			{
				indices.push_back(face.mIndices[j] + indices_count);
			}
		}
	}

	void process_node(aiNode* node, const aiScene* scene)
	{
		for (unsigned int i = 0; i < node->mNumMeshes; i++)
		{
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			process_mesh(mesh);
		}

		for (unsigned int i = 0; i < node->mNumChildren; i++)
		{
			process_node(node->mChildren[i], scene);
		}
	}

	void draw()
	{
		UINT stride = sizeof(vertex_t);
		UINT offset = 0;
		devcon->IASetIndexBuffer(pIBuffer, DXGI_FORMAT_R32_UINT, 0);
		devcon->IASetVertexBuffers(0, 1, &pVBuffer, &stride, &offset);

		devcon->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		devcon->DrawIndexed((UINT)indices.size(), 0, 0);
	}

	void setup_buffers()
	{
		// vertex buffer
		D3D11_BUFFER_DESC vertex_buffer_desc = {};
		vertex_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
		vertex_buffer_desc.ByteWidth = sizeof(vertex_t) * (UINT)vertices.size();
		vertex_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		vertex_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

		D3D11_SUBRESOURCE_DATA vertex_data = {};
		vertex_data.pSysMem = vertices.data();

		HRESULT hr = dev->CreateBuffer(&vertex_buffer_desc, &vertex_data, &pVBuffer);

		assert(SUCCEEDED(hr));
		assert(pVBuffer != NULL);

		// index buffer
		D3D11_BUFFER_DESC index_buffer_desc = {};
		index_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
		index_buffer_desc.ByteWidth = sizeof(unsigned int) * (UINT)indices.size();
		index_buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
		index_buffer_desc.CPUAccessFlags = 0;

		D3D11_SUBRESOURCE_DATA index_data = {};
		index_data.pSysMem = indices.data();

		hr = dev->CreateBuffer(&index_buffer_desc, &index_data, &pIBuffer);

		assert(SUCCEEDED(hr));
		assert(pIBuffer != NULL);
	}

	void load_from_file(std::string path)
	{
		Assimp::Importer import;
		const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_PreTransformVertices);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			log_error("ERROR::ASSIMP::", import.GetErrorString());
			return;
		}

		process_node(scene->mRootNode, scene);
		setup_buffers();
	}
};

struct asset_manager_t
{
	std::map<std::string, std::shared_ptr<shader_t>> shaders;
	std::map<std::string, std::shared_ptr<texture_t>> textures;
	std::map<std::string, std::shared_ptr<model_t>> models;

	void shutdown()
	{
		shaders.clear();
		textures.clear();
		models.clear();
	}

	std::shared_ptr<shader_t> load_shader_from_file(std::string path)
	{
		std::shared_ptr<shader_t> result;

		auto iter = shaders.find(path);
		if(iter != shaders.end())
		{
			result = iter->second;
		}
		else
		{
			result = std::make_shared<shader_t>();
			result->load_from_file(path);
			shaders.emplace(path, result); 
			log_info("Loaded Shader ", path);
		}

		return result;
	}

	std::shared_ptr<texture_t> load_texture_from_file(std::string path)
	{
		std::shared_ptr<texture_t> result;

		auto iter = textures.find(path);
		if(iter != textures.end())
		{
			result = iter->second;
		}
		else 
		{
			result = std::make_shared<texture_t>();
			result->load_from_file(path);
			textures.emplace(path, result); 
			log_info("Loaded Texture ", path);
		}

		return result;
	}

	std::shared_ptr<model_t> load_model_from_file(std::string path)
	{
		std::shared_ptr<model_t> result;

		auto iter = models.find(path);
		if(iter != models.end())
		{
			result = iter->second;
		}
		else
		{
			result = std::make_shared<model_t>();
			result->load_from_file(path);
			models.emplace(path, result); 
			log_info("Loaded Model ", path);
		}

		return result;
	}
};

asset_manager_t asset_manager;

struct camera_t
{
	vector3f position, rotation;

	vector3f front, right, up;

	matrix4 get_view_matrix()
	{
		front.x = cos(glm::radians(rotation.y)) * cos(glm::radians(rotation.x));
		front.y = sin(glm::radians(rotation.x));
		front.z = sin(glm::radians(rotation.y)) * cos(glm::radians(rotation.x));

		front = math::normalize(front);
		right = math::normalize(math::cross(front, WORLD_UP));
		up = math::normalize(math::cross(right, front));

		return glm::lookAt(vector3_to_glm_vec3(position), vector3_to_glm_vec3(position) + vector3_to_glm_vec3(front), vector3_to_glm_vec3(up));
	}

	camera_t()
		: position(0), rotation(0, -90, 0),
		  front(0, 0, -1), right(0), up(WORLD_UP)
	{}

	~camera_t() {}
};

struct drawable_model_t
{
	std::shared_ptr<model_t> model;
	std::shared_ptr<texture_t> texture;
	std::shared_ptr<shader_t> shader;

	transform_t transform;

	drawable_model_t(asset_manager_t& asset_manager, std::string model_path, std::string texture_path, std::string shader_path)
		: model(asset_manager.load_model_from_file(model_path)),
		  texture(asset_manager.load_texture_from_file(texture_path)),
		  shader(asset_manager.load_shader_from_file(shader_path)),
		  transform()
	{}

	~drawable_model_t()
	{}

	void draw()
	{
		{
			struct VS_CONSTANT_BUFFER
			{
				matrix4 view_and_projection;
				matrix4 transform;
			};

			VS_CONSTANT_BUFFER const_buffer = {};
			const_buffer.view_and_projection = projection_matrix * view_matrix;
			const_buffer.transform = transform.get_matrix();

			D3D11_MAPPED_SUBRESOURCE ms;
			devcon->Map(shader->pConstantBuffer, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms);
			memcpy(ms.pData, &const_buffer, sizeof(VS_CONSTANT_BUFFER));
			devcon->Unmap(shader->pConstantBuffer, NULL);

			devcon->VSSetConstantBuffers(0, 1, &shader->pConstantBuffer);
		}
		
		// bind shaders and textures
		devcon->VSSetShader(shader->pVs, 0, 0);
		devcon->PSSetShader(shader->pPs, 0, 0);
		devcon->IASetInputLayout(shader->pLayout);

		devcon->PSSetShaderResources(0, 1, &texture->pTexture);
		devcon->PSSetSamplers(0, 1, &texture->pSampleState);

		model->draw();
	}
};

enum key_code
{
	key_code_lbutton = 0x01, 
	key_code_rbutton = 0x02,
	key_code_cancel = 0x03,
	key_code_mbutton = 0x04, 
	key_code_xbutton1 = 0x05,
	key_code_xbutton2 = 0x06,
	// 0x07 Undefined
	key_code_back = 0x08,
	key_code_tab = 0x09,
	// 0x0A-0B Reserved
	key_code_clear = 0x0C,
	key_code_return = 0x0D,
	//0x0E-0F Undefined
	key_code_shift = 0x10, 
	key_code_control = 0x11,
	key_code_menu = 0x12,
	key_code_pause = 0x13,
	key_code_capital = 0x14,
	key_code_kana = 0x15,
	key_code_hanguel = 0x15,
	key_code_hangul = 0x15,
	key_code_ime_on = 0x16,
	key_code_junja = 0x17,
	key_code_final = 0x18,
	key_code_hanja = 0x19,
	key_code_kanji = 0x19,
	key_code_ime_off = 0x1A,
	key_code_escape = 0x1B,
	key_code_convert = 0x1C,
	key_code_nonconvert = 0x1D,
	key_code_accept = 0x1E,
	key_code_modechange = 0x1F,
	key_code_space = 0x20,
	key_code_prior = 0x21,
	key_code_next = 0x22,
	key_code_end = 0x23,
	key_code_home = 0x24,
	key_code_left = 0x25,
	key_code_up = 0x26,
	key_code_right = 0x27,
	key_code_down = 0x28,
	key_code_select = 0x29,
	key_code_print = 0x2A,
	key_code_execute = 0x2B,
	key_code_snapshot = 0x2C,
	key_code_insert = 0x2D,
	key_code_delete = 0x2E,
	key_code_help = 0x2F,
	key_code_0 = 0x30, 
	key_code_1 = 0x31,
	key_code_2 = 0x32,
	key_code_3 = 0x33,
	key_code_4 = 0x34,
	key_code_5 = 0x35,
	key_code_6 = 0x36,
	key_code_7 = 0x37,
	key_code_8 = 0x38,
	key_code_9 = 0x39,
	// 0x3A-40 Undefined
	key_code_a = 0x41,
	key_code_b = 0x42,
	key_code_c = 0x43,
	key_code_d = 0x44,
	key_code_e = 0x45,
	key_code_f = 0x46,
	key_code_g = 0x47,
	key_code_h = 0x48,
	key_code_i = 0x49,
	key_code_j = 0x4A,
	key_code_k = 0x4B,
	key_code_l = 0x4C,
	key_code_m = 0x4D,
	key_code_n = 0x4E,
	key_code_o = 0x4F,
	key_code_p = 0x50,
	key_code_q = 0x51,
	key_code_r = 0x52,
	key_code_s = 0x53,
	key_code_t = 0x54,
	key_code_u = 0x55,
	key_code_v = 0x56,
	key_code_w = 0x57,
	key_code_x = 0x58,
	key_code_y = 0x59,
	key_code_z = 0x5A,
	key_code_lwin = 0x5B,
	key_code_rwin = 0x5C,
	key_code_apps = 0x5D,
	// 0x5E	Reserved
	key_code_sleep = 0x5F,
	key_code_numpad0 = 0x60,
	key_code_numpad1 = 0x61,
	key_code_numpad2 = 0x62,
	key_code_numpad3 = 0x63,
	key_code_numpad4 = 0x64,
	key_code_numpad5 = 0x65,
	key_code_numpad6 = 0x66,
	key_code_numpad7 = 0x67,
	key_code_numpad8 = 0x68,
	key_code_numpad9 = 0x69,
	key_code_multiply = 0x6A,
	key_code_add = 0x6B,
	key_code_separator = 0x6C,
	key_code_subtract = 0x6D,
	key_code_decimal = 0x6E,
	key_code_divide = 0x6F,
	key_code_f1 = 0x70,
	key_code_f2 = 0x71,
	key_code_f3 = 0x72,
	key_code_f4 = 0x73,
	key_code_f5 = 0x74,
	key_code_f6 = 0x75,
	key_code_f7 = 0x76,
	key_code_f8 = 0x77,
	key_code_f9 = 0x78,
	key_code_f10 = 0x79,
	key_code_f11 = 0x7A,
	key_code_f12 = 0x7B,
	key_code_f13 = 0x7C,
	key_code_f14 = 0x7D,
	key_code_f15 = 0x7E,
	key_code_f16 = 0x7F,
	key_code_f17 = 0x80,
	key_code_f18 = 0x81,
	key_code_f19 = 0x82,
	key_code_f20 = 0x83,
	key_code_f21 = 0x84,
	key_code_f22 = 0x85,
	key_code_f23 = 0x86,
	key_code_f24 = 0x87,
	// 0x88-8F Unassigned
	key_code_numlock = 0x90,
	key_code_scroll = 0x91,
	// 0x92-96 OEM specific
	// 0x97-9F Unassigned
	key_code_lshift = 0xA0,
	key_code_rshift = 0xA1,
	key_code_lcontrol = 0xA2,
	key_code_rcontrol = 0xA3,
	key_code_lmenu = 0xA4,
	key_code_rmenu = 0xA5,
	key_code_browser_back = 0xA6,
	key_code_browser_forward = 0xA7,
	key_code_browser_refresh = 0xA8,
	key_code_browser_stop = 0xA9,
	key_code_browser_search = 0xAA,
	key_code_browser_favorites = 0xAB,
	key_code_browser_home = 0xAC,
	key_code_volume_mute = 0xAD,
	key_code_volume_down = 0xAE,
	key_code_volume_up = 0xAF, 
	key_code_media_next_track = 0xB0,
	key_code_media_prev_track = 0xB1,
	key_code_media_stop = 0xB2,
	key_code_media_play_pause = 0xB3,
	key_code_launch_mail = 0xB4,
	key_code_launch_media_select = 0xB5,
	key_code_launch_app1 = 0xB6,
	key_code_launch_app2 = 0xB7,
	// 0xB8-B9 Reserved
	key_code_oem_1 = 0xBA,
	key_code_oem_plus = 0xBB,
	key_code_oem_comma = 0xBC,
	key_code_oem_minus = 0xBD,
	key_code_oem_period = 0xBE,
	key_code_oem_2 = 0xBF,
	key_code_oem_3 = 0xC0,
	// 0xC1-D7 Reserved
	// 0xD8-DA Unassigned
	key_code_oem_4 = 0xDB,
	key_code_oem_5 = 0xDC,
	key_code_oem_6 = 0xDD,
	key_code_oem_7 = 0xDE,
	key_code_oem_8 = 0xDF,
	// 0xE0 Reserved
	// 0xE1 OEM specific
	key_code_oem_102 = 0xE2,
	// 0xE3-E4 OEM specific
	key_code_processkey = 0xE5,
	// 0xE6 OEM specific
	key_code_packet = 0xE7,
	// 0xE8 Unassigned
	// 0xE9-F5 OEM specific
	key_code_attn = 0xF6,
	key_code_crsel = 0xF7,
	key_code_exsel = 0xF8,
	key_code_ereof = 0xF9,
	key_code_play = 0xFA,
	key_code_zoom = 0xFB,
	key_code_noname = 0xFC,
	key_code_pa1 = 0xFD,
	key_code_oem_clear = 0xFE
};

enum mouse_button
{
	mouse_button_left,
	mouse_button_right,
	mouse_button_middle,
	mouse_button_xbutton1,
	mouse_button_xbutton2,
	mouse_button_size
};

struct input_manager_t
{
	bool keyboard_state[256], prev_keyboard_state[256];
	bool mouse_state[5], prev_mouse_state[5];
	vector2i mouse_pos = {}, prev_mouse_pos = {};
	vector2i mouse_delta = {};

	void set_key(key_code key, bool state)
	{
		prev_keyboard_state[key] = keyboard_state[key];
		keyboard_state[key] = state;
	}

	void set_mouse_button(mouse_button button, bool state)
	{
		prev_mouse_state[button] = mouse_state[button];
		mouse_state[button] = state;
	}

	void set_mouse_pos(vector2i pos)
	{
		prev_mouse_pos = mouse_pos;
		mouse_pos = pos;
	}

	bool key_held(key_code key)
	{
		return keyboard_state[key];
		//return GetAsyncKeyState(key);
	}

	bool key_pressed(key_code key)
	{
		return keyboard_state[key] && !prev_keyboard_state[key];
	}

	bool key_released(key_code key)
	{
		return !keyboard_state[key] && prev_keyboard_state[key];
	}

	bool mouse_button_pressed(mouse_button button) 
	{
		return mouse_state[button] && !prev_mouse_state[button];
	}

	bool mouse_button_held(mouse_button button)
	{
		return mouse_state[button];
	}

	bool mouse_button_released(mouse_button button)
	{
		return !mouse_state[button] && prev_mouse_state[button];
	}

	const vector2i get_mouse_pos()
	{
		return mouse_pos;
	}

	const vector2i get_previous_mouse_pos()
	{
		return prev_mouse_pos;
	}

	const vector2i get_mouse_delta()
	{
#if 1
		return mouse_delta;
#else
		return prev_mouse_pos - mouse_pos;
#endif
	}
};

input_manager_t input_manager;

struct water_t
{
	std::shared_ptr<texture_t> texture;
	std::vector<vector3f> vertices;
	std::vector<vector3f> normals;
	std::vector<vector2f> uvs;
	std::vector<unsigned int> indices;

	ID3D11Buffer* pVerticesBuffer, *pNormalsBuffer, *pUvsBuffer;
	ID3D11Buffer* pIBuffer;

	ID3D11VertexShader* pVs;
	ID3D11PixelShader* pPs;
	ID3D11InputLayout* pLayout;
	ID3D11Buffer* pConstantBuffer;

	int water_size = 50;
	float water_height = 0;

	// TODO: Environment mapping for water
	// TODO: Water Alpha
	// TODO: Water ripples.
	// TODO: Water reflections. (maybe screen space reflections????)
	// TODO: Water foam at edges.
	// TODO: Water waves. (i need to look into tesselation)
	// TODO: Water specularity.
	// TODO: Infinite water
	water_t(int size, float height)
		: water_size(size), water_height(height)
	{
		vertices.clear();
		indices.clear();
		uvs.clear();

		vertices.push_back({ 0,water_height,0 });
		vertices.push_back({ water_size,water_height,0 });
		vertices.push_back({ 0,water_height,water_size });
		vertices.push_back({ water_size,water_height,water_size });

		indices.push_back(0);
		indices.push_back(2);
		indices.push_back(1);

		indices.push_back(3);
		indices.push_back(1);
		indices.push_back(2);

		uvs.push_back({ 0,0 });
		uvs.push_back({ 1,0 });
		uvs.push_back({ 0,1 });
		uvs.push_back({ 1,1 });

		normals.push_back({ 0,0,0 });
		normals.push_back({ 0,0,0 });
		normals.push_back({ 0,0,0 });
		normals.push_back({ 0,0,0 });

		setup_buffers();
		setup_shaders();

		texture = asset_manager.load_texture_from_file("../data/diffuse.jpg");
	}

	~water_t()
	{
		vertices.clear();
		normals.clear();
		uvs.clear();
		indices.clear();

		pVerticesBuffer->Release();
		pNormalsBuffer->Release();
		pUvsBuffer->Release();

		pIBuffer->Release();

		pVs->Release();
		pPs->Release();
		pLayout->Release();
		pConstantBuffer->Release();
	}

	void setup_buffers()
	{
		{
			assert(!vertices.empty());

			D3D11_BUFFER_DESC vertex_buffer_desc = {};
			vertex_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			vertex_buffer_desc.ByteWidth = (UINT)(sizeof(vector3f) * vertices.size());
			vertex_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			vertex_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_SUBRESOURCE_DATA vertex_data = {};
			vertex_data.pSysMem = vertices.data();

			HRESULT hr = dev->CreateBuffer(&vertex_buffer_desc, &vertex_data, &pVerticesBuffer);

			assert(SUCCEEDED(hr));
			assert(pVerticesBuffer != NULL);
		}

		{
			assert(!normals.empty());

			D3D11_BUFFER_DESC normal_buffer_desc = {};
			normal_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			normal_buffer_desc.ByteWidth = (UINT)(sizeof(vector3f) * normals.size());
			normal_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			normal_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_SUBRESOURCE_DATA normal_data = {};
			normal_data.pSysMem = normals.data();

			HRESULT hr = dev->CreateBuffer(&normal_buffer_desc, &normal_data, &pNormalsBuffer);

			assert(SUCCEEDED(hr));
			assert(pNormalsBuffer != NULL);
		}

		{
			assert(!uvs.empty());

			D3D11_BUFFER_DESC uvs_buffer_desc = {};
			uvs_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			uvs_buffer_desc.ByteWidth = (UINT)(sizeof(vector2f) * uvs.size());
			uvs_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			uvs_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_SUBRESOURCE_DATA uvs_data = {};
			uvs_data.pSysMem = uvs.data();

			HRESULT hr = dev->CreateBuffer(&uvs_buffer_desc, &uvs_data, &pUvsBuffer);

			assert(SUCCEEDED(hr));
			assert(pUvsBuffer != NULL);
		}
		
		// index buffer
		D3D11_BUFFER_DESC index_buffer_desc = {};
		index_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
		index_buffer_desc.ByteWidth = sizeof(unsigned int) * (UINT)indices.size();
		index_buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
		index_buffer_desc.CPUAccessFlags = 0;

		D3D11_SUBRESOURCE_DATA index_data = {};
		index_data.pSysMem = indices.data();

		HRESULT hr = dev->CreateBuffer(&index_buffer_desc, &index_data, &pIBuffer);

		assert(SUCCEEDED(hr));
		assert(pIBuffer != NULL);
	}

	void setup_shaders()
	{
		ID3DBlob* vs;
		ID3DBlob* ps;
		ID3DBlob* errorBlob;

		static constexpr LPCWSTR grid_shader_path = L"../data/water.hlsl";

		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG;

		{
			HRESULT hr = D3DCompileFromFile(grid_shader_path, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "VSMain", "vs_4_0", flags, 0, &vs, &errorBlob);
			assert(SUCCEEDED(hr) && "Vertex Shader failed to compile");
		}

		{
			HRESULT hr = D3DCompileFromFile(grid_shader_path, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "PSMain", "ps_4_0", flags, 0, &ps, &errorBlob);
			assert(SUCCEEDED(hr) && "Pixel Shader failed to compile");
		}

		assert(errorBlob == NULL);
		assert(vs != NULL);
		assert(ps != NULL);

		dev->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), NULL, &pVs);
		dev->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(), NULL, &pPs);

		devcon->VSSetShader(pVs, 0, 0);
		devcon->PSSetShader(pPs, 0, 0);

		D3D11_INPUT_ELEMENT_DESC ied[] =
		{
			{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, 							  D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    2, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
		};

		dev->CreateInputLayout(ied, 3, vs->GetBufferPointer(), vs->GetBufferSize(), &pLayout);

		devcon->IASetInputLayout(pLayout);

		{
			struct VS_CONSTANT_BUFFER
			{
				matrix4 view_and_projection;
			};

			D3D11_BUFFER_DESC constant_buffer_desc = {};
			constant_buffer_desc.ByteWidth = sizeof(VS_CONSTANT_BUFFER);
			constant_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			constant_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constant_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			constant_buffer_desc.MiscFlags = 0;
			constant_buffer_desc.StructureByteStride = 0;

			HRESULT hr = dev->CreateBuffer(&constant_buffer_desc, NULL, &pConstantBuffer);

			assert(SUCCEEDED(hr));

			devcon->VSSetConstantBuffers(0, 1, &pConstantBuffer);
		}

		assert(pLayout != NULL);
	}

	void draw()
	{
		{
			struct VS_CONSTANT_BUFFER
			{
				matrix4 view_and_projection;
			};

			VS_CONSTANT_BUFFER const_buffer = {};
			const_buffer.view_and_projection = projection_matrix * view_matrix;

			D3D11_MAPPED_SUBRESOURCE ms;
			devcon->Map(pConstantBuffer, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms);
			memcpy(ms.pData, &const_buffer, sizeof(VS_CONSTANT_BUFFER));
			devcon->Unmap(pConstantBuffer, NULL);

			devcon->VSSetConstantBuffers(0, 1, &pConstantBuffer);
		}
		
		// bind shaders and textures
		devcon->VSSetShader(pVs, 0, 0);
		devcon->PSSetShader(pPs, 0, 0);
		devcon->IASetInputLayout(pLayout);

		devcon->PSSetShaderResources(0, 1, &texture->pTexture);
		devcon->PSSetSamplers(0, 1, &texture->pSampleState);

		devcon->IASetIndexBuffer(pIBuffer, DXGI_FORMAT_R32_UINT, 0);

		ID3D11Buffer* buffers[] = { pVerticesBuffer, pNormalsBuffer, pUvsBuffer };
		unsigned int stride[] = { sizeof(vector3f), sizeof(vector3f), sizeof(vector2f) };
		unsigned int offset[] = { 0,0,0 };

		devcon->IASetVertexBuffers(0, 3, buffers, stride, offset);

		devcon->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		devcon->DrawIndexed((UINT)indices.size(), 0, 0);
	}
};

constexpr size_t grid_array_size = 500;

// TODO: ui editor (edit noise, weather, water level etc....)

struct terrain_t
{
	// TODO: Trees, rocks and vegetation
	// TODO: Weather (e.g. rain, snow, wind(maybe????), cloudy, sunny etc..)
	// TODO: Day / night cycle
	// TODO: Grass
	// TODO: Terrain collisions and physics system.
	// TODO: Add chunks
	// TODO: Biomes
	// TODO: Water falls, ponds, streams and other water types.

	std::shared_ptr<texture_t> top_texture;
	std::shared_ptr<texture_t> right_texture;
	std::shared_ptr<texture_t> front_texture;

	std::vector<vector3f> vertices;
	std::vector<vector3f> normals;
	std::vector<vector2f> uvs;
	std::vector<unsigned int> indices;

	ID3D11Buffer* pVerticesBuffer, *pNormalsBuffer, *pUvsBuffer;
	ID3D11Buffer* pIBuffer;

	ID3D11VertexShader* pVs;
	ID3D11PixelShader* pPs;
	ID3D11InputLayout* pLayout;
	ID3D11Buffer* pConstantBuffer;

	std::unique_ptr<std::array<float, grid_array_size*grid_array_size>> grid;

	terrain_t()
	{
		grid = std::make_unique<std::array<float, grid_array_size*grid_array_size>>();

		top_texture = asset_manager.load_texture_from_file("../data/terrain/grass.jpg");
		right_texture = asset_manager.load_texture_from_file("../data/terrain/rock1.jpg");
		front_texture = asset_manager.load_texture_from_file("../data/terrain/rock2.jpg");

		gen_terrain();
		gen_mesh();
	}

	// TODO: Shadow mapping
	// TODO: Smooth normals
	~terrain_t()
	{
		vertices.clear();
		normals.clear();
		uvs.clear();
		indices.clear();

		pVerticesBuffer->Release();
		pNormalsBuffer->Release();
		pUvsBuffer->Release();

		pIBuffer->Release();

		pVs->Release();
		pPs->Release();
		pLayout->Release();
		pConstantBuffer->Release();
	}
	
	void gen_terrain()
	{
		grid->fill(0);
		static constexpr float noise_scale = 50.0f;

		int seed = static_cast<int>(time(NULL));
#if 1
		{
			FastNoiseLite noise;
			noise.SetSeed(seed);//125);

			noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);
			noise.SetFrequency(0.002f);

			noise.SetFractalType(FastNoiseLite::FractalType_FBm);
			noise.SetFractalOctaves(5);
			noise.SetFractalLacunarity(1.0f);
			noise.SetFractalGain(0.3f);
			noise.SetFractalWeightedStrength(1.0f);

			for(unsigned int i = 0; i < grid->size(); i++)
			{
				int x = i % grid_array_size;
				int y = i / grid_array_size;

				(*grid.get())[i] += noise.GetNoise((float)x, (float)y) * 100;
			}
		}
#endif
#if 1
		{
			FastNoiseLite noise;
			noise.SetSeed(seed + 10);

			noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);
			noise.SetFrequency(0.01f);

			noise.SetFractalType(FastNoiseLite::FractalType_FBm);
			noise.SetFractalOctaves(10);
			//noise.SetFractalLacunarity(2.1f);
			noise.SetFractalWeightedStrength(0.2f);

			for(unsigned int i = 0; i < grid->size(); i++)
			{
				int x = i % grid_array_size;
				int y = i / grid_array_size;

				(*grid.get())[i] += noise.GetNoise((float)x, (float)y) * noise_scale;
			}
		}
#endif
#if 1
		{
			FastNoiseLite noise;
			noise.SetSeed(seed + 20);//50);

			noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);
			noise.SetFrequency(0.05f);

			noise.SetFractalType(FastNoiseLite::FractalType_FBm);
			noise.SetFractalOctaves(10);
			//noise.SetFractalLacunarity(2.1f);
			noise.SetFractalWeightedStrength(0.1f);

			for(unsigned int i = 0; i < grid->size(); i++)
			{
				int x = i % grid_array_size;
				int y = i / grid_array_size;

				(*grid.get())[i] += noise.GetNoise((float)x, (float)y) * 2;
			}
		}
#endif
#if 1
		{
			// apply small amount of random noise on top of first noise
			auto gen = std::bind(std::uniform_real_distribution<>(0,0.5f),std::default_random_engine((unsigned int)std::chrono::system_clock::now().time_since_epoch().count()));

			for(auto&& cell : *grid.get())
			{
				cell += static_cast<float>(gen());
			}
		}
#endif
	}

	void gen_mesh()
	{
		vertices.clear();
		indices.clear();
		uvs.clear();

		// TODO: Fix height at end of array
		for(int x = 0; x < grid_array_size; x++)
		{
			for(int z = 0; z < grid_array_size; z++)
			{
				auto grid_pos = vector3f(x,grid.get()->at(grid_array_size * z + x),z);
				int index_offset = (int)vertices.size();
				// FIRST TRIANGLE
				vertices.push_back(vector3f(0,0,0) + vector3f(x,grid.get()->at(grid_array_size * z + x),z));
				vertices.push_back(vector3f(1,0,0) + vector3f(x,grid.get()->at((x < grid_array_size-1) ? grid_array_size * z + (x+1) : grid_array_size * z + x),z));
				vertices.push_back(vector3f(0,0,1) + vector3f(x,grid.get()->at((z < grid_array_size-1) ? grid_array_size * (z+1) + x : grid_array_size * z + x),z));

				// SECOND TRIANGLE
				vertices.push_back(vector3f(1,0,0) + vector3f(x,grid.get()->at((x < grid_array_size-1) ? grid_array_size * z + (x+1) : grid_array_size * z + x),z));
				vertices.push_back(vector3f(0,0,1) + vector3f(x,grid.get()->at((z < grid_array_size-1) ? grid_array_size * (z+1) + x : grid_array_size * z + x),z));
				vertices.push_back(vector3f(1,0,1) + vector3f(x,grid.get()->at((x < grid_array_size-1 && z < grid_array_size-1) ? grid_array_size * (z+1) + (x+1) : grid_array_size * z + x),z));
				
				uvs.push_back({0,0});
				uvs.push_back({1,0});
				uvs.push_back({0,1});

				uvs.push_back({1,0});
				uvs.push_back({0,1});
				uvs.push_back({1,1});

				indices.push_back(index_offset + 0);
				indices.push_back(index_offset + 2);
				indices.push_back(index_offset + 1);

				indices.push_back(index_offset + 3);
				indices.push_back(index_offset + 4);
				indices.push_back(index_offset + 5);

				vector3f A = vertices.at(index_offset + 0);
				vector3f B = vertices.at(index_offset + 1);
				vector3f C = vertices.at(index_offset + 2);

				normals.push_back(math::cross(A-B,A-C));
				normals.push_back(math::cross(A-B,A-C));
				normals.push_back(math::cross(A-B,A-C));

				vector3f E = vertices.at(index_offset + 3);
				vector3f F = vertices.at(index_offset + 5);
				vector3f G = vertices.at(index_offset + 4);

				normals.push_back(math::cross(E-F,E-G));
				normals.push_back(math::cross(E-F,E-G));
				normals.push_back(math::cross(E-F,E-G));
			}
		}

		setup_buffers();
		setup_shaders();
	}

	void setup_buffers()
	{
		{
			assert(!vertices.empty());

			D3D11_BUFFER_DESC vertex_buffer_desc = {};
			vertex_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			vertex_buffer_desc.ByteWidth = (UINT)(sizeof(vector3f) * vertices.size());
			vertex_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			vertex_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_SUBRESOURCE_DATA vertex_data = {};
			vertex_data.pSysMem = vertices.data();

			HRESULT hr = dev->CreateBuffer(&vertex_buffer_desc, &vertex_data, &pVerticesBuffer);

			assert(SUCCEEDED(hr));
			assert(pVerticesBuffer != NULL);
		}

		{
			assert(!normals.empty());

			D3D11_BUFFER_DESC normal_buffer_desc = {};
			normal_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			normal_buffer_desc.ByteWidth = (UINT)(sizeof(vector3f) * normals.size());
			normal_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			normal_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_SUBRESOURCE_DATA normal_data = {};
			normal_data.pSysMem = normals.data();

			HRESULT hr = dev->CreateBuffer(&normal_buffer_desc, &normal_data, &pNormalsBuffer);

			assert(SUCCEEDED(hr));
			assert(pNormalsBuffer != NULL);
		}

		{
			assert(!uvs.empty());

			D3D11_BUFFER_DESC uvs_buffer_desc = {};
			uvs_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			uvs_buffer_desc.ByteWidth = (UINT)(sizeof(vector2f) * uvs.size());
			uvs_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			uvs_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_SUBRESOURCE_DATA uvs_data = {};
			uvs_data.pSysMem = uvs.data();

			HRESULT hr = dev->CreateBuffer(&uvs_buffer_desc, &uvs_data, &pUvsBuffer);

			assert(SUCCEEDED(hr));
			assert(pUvsBuffer != NULL);
		}
		
		// index buffer
		D3D11_BUFFER_DESC index_buffer_desc = {};
		index_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
		index_buffer_desc.ByteWidth = sizeof(unsigned int) * (UINT)indices.size();
		index_buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
		index_buffer_desc.CPUAccessFlags = 0;

		D3D11_SUBRESOURCE_DATA index_data = {};
		index_data.pSysMem = indices.data();

		HRESULT hr = dev->CreateBuffer(&index_buffer_desc, &index_data, &pIBuffer);

		assert(SUCCEEDED(hr));
		assert(pIBuffer != NULL);
	}

	void setup_shaders()
	{
		ID3DBlob* vs;
		ID3DBlob* ps;
		ID3DBlob* errorBlob;

		static constexpr LPCWSTR grid_shader_path = L"../data/terrain.hlsl";

		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG;

		{
			HRESULT hr = D3DCompileFromFile(grid_shader_path, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "VSMain", "vs_4_0", flags, 0, &vs, &errorBlob);
			assert(SUCCEEDED(hr) && "Vertex Shader failed to compile");
		}

		{
			HRESULT hr = D3DCompileFromFile(grid_shader_path, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "PSMain", "ps_4_0", flags, 0, &ps, &errorBlob);
			assert(SUCCEEDED(hr) && "Pixel Shader failed to compile");
		}

		assert(errorBlob == NULL);
		assert(vs != NULL);
		assert(ps != NULL);

		dev->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), NULL, &pVs);
		dev->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(), NULL, &pPs);

		devcon->VSSetShader(pVs, 0, 0);
		devcon->PSSetShader(pPs, 0, 0);

		D3D11_INPUT_ELEMENT_DESC ied[] =
		{
			{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, 							  D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    2, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
		};

		dev->CreateInputLayout(ied, 3, vs->GetBufferPointer(), vs->GetBufferSize(), &pLayout);

		devcon->IASetInputLayout(pLayout);

		{
			struct VS_CONSTANT_BUFFER
			{
				matrix4 view_and_projection;
			};

			D3D11_BUFFER_DESC constant_buffer_desc = {};
			constant_buffer_desc.ByteWidth = sizeof(VS_CONSTANT_BUFFER);
			constant_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			constant_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constant_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			constant_buffer_desc.MiscFlags = 0;
			constant_buffer_desc.StructureByteStride = 0;

			HRESULT hr = dev->CreateBuffer(&constant_buffer_desc, NULL, &pConstantBuffer);

			assert(SUCCEEDED(hr));

			devcon->VSSetConstantBuffers(0, 1, &pConstantBuffer);
		}

		assert(pLayout != NULL);
	}

	void draw()
	{
		{
			struct VS_CONSTANT_BUFFER
			{
				matrix4 view_and_projection;
			};

			VS_CONSTANT_BUFFER const_buffer = {};
			const_buffer.view_and_projection = projection_matrix * view_matrix;

			D3D11_MAPPED_SUBRESOURCE ms;
			devcon->Map(pConstantBuffer, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms);
			memcpy(ms.pData, &const_buffer, sizeof(VS_CONSTANT_BUFFER));
			devcon->Unmap(pConstantBuffer, NULL);

			devcon->VSSetConstantBuffers(0, 1, &pConstantBuffer);
		}
		
		// bind shaders and textures
		devcon->VSSetShader(pVs, 0, 0);
		devcon->PSSetShader(pPs, 0, 0);
		devcon->IASetInputLayout(pLayout);

#if 0
		devcon->PSSetShaderResources(0, 1, &top_texture->pTexture);
		devcon->PSSetSamplers(0, 1, &top_texture->pSampleState);

		devcon->PSSetShaderResources(1, 1, &right_texture->pTexture);
		devcon->PSSetSamplers(1, 1, &right_texture->pSampleState);

		devcon->PSSetShaderResources(2, 1, &front_texture->pTexture);
		devcon->PSSetSamplers(2, 1, &front_texture->pSampleState);
#else
		top_texture->set(0);
		right_texture->set(1);
		front_texture->set(2);
#endif

		devcon->IASetIndexBuffer(pIBuffer, DXGI_FORMAT_R32_UINT, 0);

		ID3D11Buffer* buffers[] = { pVerticesBuffer, pNormalsBuffer, pUvsBuffer };
		unsigned int stride[] = { sizeof(vector3f), sizeof(vector3f), sizeof(vector2f) };
		unsigned int offset[] = { 0,0,0 };

		devcon->IASetVertexBuffers(0, 3, buffers, stride, offset);

		devcon->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		devcon->DrawIndexed((UINT)indices.size(), 0, 0);
	}
};

struct skybox_t
{
	std::vector<vector3f> vertices;
	std::vector<vector2f> uvs;
	std::vector<unsigned int> indices;

	ID3D11ShaderResourceView* pTexture;
	ID3D11SamplerState* pSampleState;

	ID3D11Buffer* pVerticesBuffer, *pNormalsBuffer, *pUvsBuffer;
	ID3D11Buffer* pIBuffer;

	ID3D11VertexShader* pVs;
	ID3D11PixelShader* pPs;
	ID3D11InputLayout* pLayout;
	ID3D11Buffer* pConstantBuffer;

	skybox_t()
	{
		vertices.clear();
		indices.clear();
		uvs.clear();
		
		// set vertices, indices and uvs. (i dont think well need normals)
		{
			// negative y face
			int index_offset = (int)vertices.size();

			vertices.push_back({ -1,-1,-1 });
			vertices.push_back({  1,-1,-1 });
			vertices.push_back({ -1,-1, 1 });
			vertices.push_back({  1,-1, 1 });

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 2);
			indices.push_back(index_offset + 3);

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 3);
			indices.push_back(index_offset + 1);

			uvs.push_back({ 0,0 });
			uvs.push_back({ 1,0 });
			uvs.push_back({ 0,1 });
			uvs.push_back({ 1,1 });
		}

		{
			// positive y face
			int index_offset = (int)vertices.size();

			vertices.push_back({ -1,1,-1 });
			vertices.push_back({  1,1,-1 });
			vertices.push_back({ -1,1, 1 });
			vertices.push_back({  1,1, 1 });

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 3);
			indices.push_back(index_offset + 2);

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 1);
			indices.push_back(index_offset + 3);

			uvs.push_back({ 0,0 });
			uvs.push_back({ 1,0 });
			uvs.push_back({ 0,1 });
			uvs.push_back({ 1,1 });
		}

		{
			// negative x face
			int index_offset = (int)vertices.size();

			vertices.push_back({ -1,-1,-1 });
			vertices.push_back({ -1,-1, 1 });
			vertices.push_back({ -1, 1,-1 });
			vertices.push_back({ -1, 1, 1 });

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 2);
			indices.push_back(index_offset + 3);

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 3);
			indices.push_back(index_offset + 1);

			uvs.push_back({ 0,0 });
			uvs.push_back({ 1,0 });
			uvs.push_back({ 0,1 });
			uvs.push_back({ 1,1 });
		}

		{
			// positive x face
			int index_offset = (int)vertices.size();

			vertices.push_back({ 1,-1,-1 });
			vertices.push_back({ 1,-1, 1 });
			vertices.push_back({ 1, 1,-1 });
			vertices.push_back({ 1, 1, 1 });

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 3);
			indices.push_back(index_offset + 2);

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 1);
			indices.push_back(index_offset + 3);

			uvs.push_back({ 0,0 });
			uvs.push_back({ 1,0 });
			uvs.push_back({ 0,1 });
			uvs.push_back({ 1,1 });
		}

		{
			// negative z face
			int index_offset = (int)vertices.size();

			vertices.push_back({ -1,-1,-1 });
			vertices.push_back({  1,-1,-1 });
			vertices.push_back({ -1, 1,-1 });
			vertices.push_back({  1, 1,-1 });

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 3);
			indices.push_back(index_offset + 2);

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 1);
			indices.push_back(index_offset + 3);

			uvs.push_back({ 0,0 });
			uvs.push_back({ 1,0 });
			uvs.push_back({ 0,1 });
			uvs.push_back({ 1,1 });
		}

		{
			// positive z face
			int index_offset = (int)vertices.size();

			vertices.push_back({ -1,-1,1 });
			vertices.push_back({  1,-1,1 });
			vertices.push_back({ -1, 1,1 });
			vertices.push_back({  1, 1,1 });

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 2);
			indices.push_back(index_offset + 3);

			indices.push_back(index_offset + 0);
			indices.push_back(index_offset + 3);
			indices.push_back(index_offset + 1);

			uvs.push_back({ 0,0 });
			uvs.push_back({ 1,0 });
			uvs.push_back({ 0,1 });
			uvs.push_back({ 1,1 });
		}

		setup_buffers();
		setup_shaders();
		setup_textures();
	}

	~skybox_t()
	{
		vertices.clear();
		uvs.clear();
		indices.clear();

		pTexture->Release();
		pSampleState->Release();

		pVerticesBuffer->Release();
		pNormalsBuffer->Release();
		pUvsBuffer->Release();

		pIBuffer->Release();

		pVs->Release();
		pPs->Release();
		pLayout->Release();
		pConstantBuffer->Release();
	}

	void setup_textures()
	{
		std::array<std::string, 6> images =
		{
			"../data/sky/right.jpg",
			"../data/sky/left.jpg",
			"../data/sky/top.jpg",
			"../data/sky/bottom.jpg",
			"../data/sky/front.jpg",
			"../data/sky/back.jpg"
		};

		std::array<unsigned char*, 6> texture_data;

		int width, height, nrcomponents, desired_channels = 4;
		DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;

		for(int i = 0; i < texture_data.size(); i++)
		{
			unsigned char* data = stbi_load(images[i].c_str(), &width, &height, &nrcomponents, desired_channels);

			assert(data && "Texture failed to load");

			if(data)
			{
				if (nrcomponents == 1)
				{
					format = DXGI_FORMAT_R8_UNORM;
				}
				else if (nrcomponents == 3)
				{
					// if RGB just use RGBA no RGB format;
					format = DXGI_FORMAT_R8G8B8A8_UNORM;
				}
				else if (nrcomponents == 4)
				{
					format = DXGI_FORMAT_R8G8B8A8_UNORM;
				}

				assert(format != DXGI_FORMAT_UNKNOWN);

				texture_data.at(i) = data;
			}
		}

		ID3D11Texture2D* texture = {};

		static constexpr int mip_levels = 1;

		{
			D3D11_TEXTURE2D_DESC texture_desc = {};
			texture_desc.Width = width;
			texture_desc.Height = height;
			texture_desc.MipLevels = mip_levels;
			texture_desc.ArraySize = 6;
			texture_desc.Format = format;
			texture_desc.SampleDesc.Count = 1;
			texture_desc.SampleDesc.Quality = 0;
			texture_desc.Usage = D3D11_USAGE_IMMUTABLE;
			texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			texture_desc.CPUAccessFlags = 0;
			texture_desc.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;
			
			std::array<D3D11_SUBRESOURCE_DATA,6> init_data;
			for(int i = 0; i < init_data.size(); i++)
			{
				init_data.at(i).pSysMem = texture_data.at(i);
				init_data.at(i).SysMemPitch = width * 4;
				init_data.at(i).SysMemSlicePitch = 0;
			}
			
			HRESULT hr = dev->CreateTexture2D(&texture_desc, &init_data.front(), &texture);
			assert(SUCCEEDED(hr) && "Failed Texture Creation");
		}

		{
			D3D11_SHADER_RESOURCE_VIEW_DESC tex_view_desc = {};
			tex_view_desc.Format = format;
			tex_view_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
			tex_view_desc.TextureCube.MipLevels = mip_levels;
			tex_view_desc.TextureCube.MostDetailedMip = 0;

			HRESULT hr = dev->CreateShaderResourceView(texture, &tex_view_desc, &pTexture);
			assert(SUCCEEDED(hr) && "Failed to creater shader resource for texture");
		}

		{
			D3D11_SAMPLER_DESC samplerDesc = {};
			samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
			samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
			samplerDesc.MinLOD = 0;
			samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

			HRESULT hr = dev->CreateSamplerState(&samplerDesc, &pSampleState);
			assert(SUCCEEDED(hr) && "Failed to create sampler");
		}

		for(auto&& tex : texture_data)
		{
			stbi_image_free(tex);
		}
	}
	
	void setup_buffers()
	{
		{
			assert(!vertices.empty());

			D3D11_BUFFER_DESC vertex_buffer_desc = {};
			vertex_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			vertex_buffer_desc.ByteWidth = (UINT)(sizeof(vector3f) * vertices.size());
			vertex_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			vertex_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_SUBRESOURCE_DATA vertex_data = {};
			vertex_data.pSysMem = vertices.data();

			HRESULT hr = dev->CreateBuffer(&vertex_buffer_desc, &vertex_data, &pVerticesBuffer);

			assert(SUCCEEDED(hr));
			assert(pVerticesBuffer != NULL);
		}

		{
			assert(!uvs.empty());

			D3D11_BUFFER_DESC uvs_buffer_desc = {};
			uvs_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			uvs_buffer_desc.ByteWidth = (UINT)(sizeof(vector2f) * uvs.size());
			uvs_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			uvs_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			D3D11_SUBRESOURCE_DATA uvs_data = {};
			uvs_data.pSysMem = uvs.data();

			HRESULT hr = dev->CreateBuffer(&uvs_buffer_desc, &uvs_data, &pUvsBuffer);

			assert(SUCCEEDED(hr));
			assert(pUvsBuffer != NULL);
		}
		
		// index buffer
		D3D11_BUFFER_DESC index_buffer_desc = {};
		index_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
		index_buffer_desc.ByteWidth = sizeof(unsigned int) * (UINT)indices.size();
		index_buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
		index_buffer_desc.CPUAccessFlags = 0;

		D3D11_SUBRESOURCE_DATA index_data = {};
		index_data.pSysMem = indices.data();

		HRESULT hr = dev->CreateBuffer(&index_buffer_desc, &index_data, &pIBuffer);

		assert(SUCCEEDED(hr));
		assert(pIBuffer != NULL);
	}

	void setup_shaders()
	{
		ID3DBlob* vs;
		ID3DBlob* ps;
		ID3DBlob* errorBlob;

		static constexpr LPCWSTR grid_shader_path = L"../data/skybox.hlsl";

		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG;

		{
			HRESULT hr = D3DCompileFromFile(grid_shader_path, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "VSMain", "vs_4_0", flags, 0, &vs, &errorBlob);
			assert(SUCCEEDED(hr) && "Vertex Shader failed to compile");
		}

		{
			HRESULT hr = D3DCompileFromFile(grid_shader_path, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "PSMain", "ps_4_0", flags, 0, &ps, &errorBlob);
			assert(SUCCEEDED(hr) && "Pixel Shader failed to compile");
		}

		assert(errorBlob == NULL);
		assert(vs != NULL);
		assert(ps != NULL);

		dev->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), NULL, &pVs);
		dev->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(), NULL, &pPs);

		devcon->VSSetShader(pVs, 0, 0);
		devcon->PSSetShader(pPs, 0, 0);

		D3D11_INPUT_ELEMENT_DESC ied[] =
		{
			{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, 							  D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
		};

		dev->CreateInputLayout(ied, 2, vs->GetBufferPointer(), vs->GetBufferSize(), &pLayout);

		devcon->IASetInputLayout(pLayout);

		{
			struct VS_CONSTANT_BUFFER
			{
				matrix4 view_and_projection;
			};

			D3D11_BUFFER_DESC constant_buffer_desc = {};
			constant_buffer_desc.ByteWidth = sizeof(VS_CONSTANT_BUFFER);
			constant_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			constant_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constant_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			constant_buffer_desc.MiscFlags = 0;
			constant_buffer_desc.StructureByteStride = 0;

			HRESULT hr = dev->CreateBuffer(&constant_buffer_desc, NULL, &pConstantBuffer);

			assert(SUCCEEDED(hr));

			devcon->VSSetConstantBuffers(0, 1, &pConstantBuffer);
		}

		assert(pLayout != NULL);
	}

	void draw()
	{
		{
			struct VS_CONSTANT_BUFFER
			{
				matrix4 view_and_projection;
			};

			VS_CONSTANT_BUFFER const_buffer = {};
			const_buffer.view_and_projection = projection_matrix * matrix4(glm::mat3(view_matrix));

			D3D11_MAPPED_SUBRESOURCE ms;
			devcon->Map(pConstantBuffer, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms);
			memcpy(ms.pData, &const_buffer, sizeof(VS_CONSTANT_BUFFER));
			devcon->Unmap(pConstantBuffer, NULL);

			devcon->VSSetConstantBuffers(0, 1, &pConstantBuffer);
		}
		
		// bind shaders and textures
		devcon->VSSetShader(pVs, 0, 0);
		devcon->PSSetShader(pPs, 0, 0);
		devcon->IASetInputLayout(pLayout);

		devcon->PSSetShaderResources(0, 1, &pTexture);
		devcon->PSSetSamplers(0, 1, &pSampleState);

		devcon->IASetIndexBuffer(pIBuffer, DXGI_FORMAT_R32_UINT, 0);

		ID3D11Buffer* buffers[] = { pVerticesBuffer, pUvsBuffer };
		unsigned int stride[] = { sizeof(vector3f), sizeof(vector2f) };
		unsigned int offset[] = { 0,0,0 };

		devcon->IASetVertexBuffers(0, 2, buffers, stride, offset);

		devcon->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		devcon->DrawIndexed((UINT)indices.size(), 0, 0);
	}
};

LRESULT CALLBACK WindowProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
	switch(message)
	{
		case WM_DESTROY:
		{
			PostQuitMessage(0);
			return 0;
		} break;
		case WM_SIZE:
		{
			if(wparam == SIZE_MINIMIZED)
			{
				// Pause application
			}
			else
			{
				if(swapchain)
				{
					screen_width = LOWORD(lparam);
					screen_height = HIWORD(lparam);

					devcon->OMSetRenderTargets(0,0,0);

					backbuffer->Release();

					HRESULT hr = swapchain->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0);
					assert(SUCCEEDED(hr) && "Resize Buffer Failed");

					// Recreate Backbuffer
					ID3D11Texture2D* pBackbuffer = {};
					swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackbuffer);
					dev->CreateRenderTargetView(pBackbuffer, NULL, &backbuffer);
					pBackbuffer->Release();
					assert(backbuffer != NULL && "Failed to Recreate the Backbuffer");
					
					// Recreate Depthbuffer 
					depthbuffer->Release();
					D3D11_TEXTURE2D_DESC depth_texture_desc = {};
					depth_texture_desc.Width = screen_width;
					depth_texture_desc.Height = screen_height;
					depth_texture_desc.MipLevels = 1;
					depth_texture_desc.ArraySize = 1;
					depth_texture_desc.SampleDesc.Count = 1;
					depth_texture_desc.SampleDesc.Quality = 0;
					depth_texture_desc.Format = DXGI_FORMAT_D32_FLOAT;

					depth_texture_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

					ID3D11Texture2D* pDepthStencil = {};
					hr = dev->CreateTexture2D(&depth_texture_desc, NULL, &pDepthStencil);
					assert(SUCCEEDED(hr) && "Failed to recreate depth texture");

					hr = dev->CreateDepthStencilView(pDepthStencil, NULL, &depthbuffer);
					assert(SUCCEEDED(hr) && "Failed to recreate depth view");

					pDepthStencil->Release();

					devcon->OMSetRenderTargets(1, &backbuffer, depthbuffer);

					D3D11_VIEWPORT viewport = {};
					viewport.TopLeftX = 0;
					viewport.TopLeftY = 0;
					viewport.Width = (float)screen_width;
					viewport.Height = (float)screen_height;
					viewport.MinDepth = 0.0f;
					viewport.MaxDepth = 1.0f;
					devcon->RSSetViewports(1, &viewport);
				}
			}
		} break;
#ifdef USE_RAW_INPUT
		case WM_INPUT:
		{
			UINT dwSize;

			GetRawInputData((HRAWINPUT)lparam, RID_INPUT, NULL, &dwSize, sizeof(RAWINPUTHEADER));
			LPBYTE lpb = new BYTE[dwSize];
			if(lpb == NULL)
			{
				return 0;
			}

			assert(GetRawInputData((HRAWINPUT)lparam, RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER)) == dwSize);

			RAWINPUT* raw = (RAWINPUT*)lpb;

			if(raw->header.dwType == RIM_TYPEKEYBOARD)
			{
				key_code key = (key_code)raw->data.keyboard.VKey;
				auto flags = raw->data.keyboard.Flags;

#if 1
				input_manager.set_key(key, (flags == RI_KEY_MAKE));

				// TODO: Add alt, ctrl, and other buttons with left right versions.
#if 0
				// ENTER KEY
				if(key == 0x0D)
				{
					if(raw->data.keyboard.MakeCode == 0x1C)
					{
						input_manager.set_key(key_code_r, (flags == (RI_KEY_MAKE|RI_KEY_E0)));
					}

					if(raw->data.keyboard.MakeCode == 0x36)
					{
						input_manager.set_key(key_code_lshift, (flags == (RI_KEY_MAKE|RI_KEY_E0)));
					}
				}
#endif

				auto make_code = raw->data.keyboard.MakeCode;
				// SHIFT KEY
				if(key == 0x10)
				{
					if(make_code == 0x2A)
					{
						input_manager.set_key(key_code_lshift, (flags == RI_KEY_MAKE));
					}

					if(make_code == 0x36)
					{
						input_manager.set_key(key_code_rshift, (flags == RI_KEY_MAKE));
					}
				}
				
				// ALT KEY
				if(key == 0x12)
				{
					if(make_code == 0x1D)
					{
						input_manager.set_key(key_code_rmenu, (flags == (RI_KEY_MAKE|RI_KEY_E0) || flags == (RI_KEY_BREAK|RI_KEY_E0)));
					}

					if(make_code == 0x1D)
					{
						input_manager.set_key(key_code_lmenu, (flags == (RI_KEY_MAKE|RI_KEY_E0) || flags == (RI_KEY_BREAK|RI_KEY_E0)));
					}
				}
#else
				if((raw->data.keyboard.Flags & RI_KEY_MAKE) == RI_KEY_MAKE)
				{
					input_manager.set_key(key, true);

					/*
					if(key == 0x10)
					{
						if(key == 0x2A)
							input_manager.set_key(key_code_lshift,true);
					}
					*/
				}

				if((raw->data.keyboard.Flags & RI_KEY_BREAK) == RI_KEY_BREAK)
				{
					input_manager.set_key(key, false);
				}
#endif
			}
			else if(raw->header.dwType == RIM_TYPEMOUSE)
			{
				if((raw->data.mouse.usFlags & MOUSE_MOVE_RELATIVE) == MOUSE_MOVE_RELATIVE)
				{
					input_manager.mouse_delta = { -raw->data.mouse.lLastX, -raw->data.mouse.lLastY };
				}
				
				if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_DOWN) == RI_MOUSE_BUTTON_1_DOWN)
				{
					input_manager.set_mouse_button(mouse_button_left, true);
				}
				else if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_UP) == RI_MOUSE_BUTTON_1_DOWN)
				{
					input_manager.set_mouse_button(mouse_button_left, false);
				}
				else if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_DOWN) == RI_MOUSE_BUTTON_2_DOWN)
				{
					input_manager.set_mouse_button(mouse_button_right, true);
				}
				else if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_UP) == RI_MOUSE_BUTTON_2_UP)
				{
					input_manager.set_mouse_button(mouse_button_right, false);
				}
				else if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_DOWN) == RI_MOUSE_BUTTON_1_DOWN)
				{
					input_manager.set_mouse_button(mouse_button_middle, true);
				}
				else if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_UP) == RI_MOUSE_BUTTON_1_DOWN)
				{
					input_manager.set_mouse_button(mouse_button_middle, false);
				}
				else if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_4_DOWN) == RI_MOUSE_BUTTON_1_DOWN)
				{
					input_manager.set_mouse_button(mouse_button_xbutton1, true);
				}
				else if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_4_UP) == RI_MOUSE_BUTTON_1_DOWN)
				{
					input_manager.set_mouse_button(mouse_button_xbutton1, false);
				}
				else if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_5_DOWN) == RI_MOUSE_BUTTON_1_DOWN)
				{
					input_manager.set_mouse_button(mouse_button_xbutton1, true);
				}
				else if((raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_5_UP) == RI_MOUSE_BUTTON_1_DOWN)
				{
					input_manager.set_mouse_button(mouse_button_xbutton1, false);
				}
			}
		} break;
#else
		case WM_SYSKEYDOWN:
		case WM_SYSKEYUP:
		case WM_KEYDOWN:
		case WM_KEYUP:
		{
			// lparam & (1 << 30)) == 0;
			// lparam & (1 << 31)) == 0;
			input_manager.prev_keyboard_state[(key_code)wparam] = (lparam & (1 << 30)) != 0; 
			input_manager.keyboard_state[(key_code)wparam] = (lparam & (1 << 31)) == 0; 

			//input_manager.set_key((key_code)wparam, (lparam & (1 << 30)) == 0);
		} break;

		case WM_LBUTTONDOWN:
		{
			input_manager.set_mouse_button(mouse_button_left, true);
		} break;
		case WM_LBUTTONUP:
		{
			input_manager.set_mouse_button(mouse_button_left, false);
		} break;

		case WM_RBUTTONDOWN:
		{
			input_manager.set_mouse_button(mouse_button_right, true);
		} break;
		case WM_RBUTTONUP:
		{
			input_manager.set_mouse_button(mouse_button_right, false);
		} break;

		case WM_MBUTTONDOWN:
		{
			input_manager.set_mouse_button(mouse_button_middle, true);
		} break;
		case WM_MBUTTONUP:
		{
			input_manager.set_mouse_button(mouse_button_middle, false);
		} break;

		case WM_MOUSEMOVE:
		{
			vector2i pos = {};
			pos.x = lparam & 0xffff;
			pos.y = (lparam >> 16) & 0xffff;
			input_manager.set_mouse_pos(pos);
		} break;
#endif
	}

	return DefWindowProc(hwnd, message, wparam, lparam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	// Init Window
	WNDCLASSEX wc = {};
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.lpszClassName = "WindowClass1";
	RegisterClassEx(&wc);

	RECT wr = {0, 0, DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT};
	AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);

	HWND hwnd = CreateWindowEx(NULL, "WindowClass1", "Terrain Generation", WS_OVERLAPPEDWINDOW, 300, 300, wr.right - wr.left, wr.bottom - wr.top, NULL, NULL, hInstance, NULL);

	ShowWindow(hwnd, nShowCmd);

	// init 3d
	DXGI_SWAP_CHAIN_DESC scd = {};

	scd.BufferCount = 1;
	scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	scd.BufferDesc.Width = DEFAULT_SCREEN_WIDTH;
	scd.BufferDesc.Height = DEFAULT_SCREEN_HEIGHT;
	scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scd.OutputWindow = hwnd;
	scd.SampleDesc.Count = 1;
	scd.Windowed = TRUE;
	//scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	scd.Flags = 0;

	D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_DEBUG, NULL, NULL, D3D11_SDK_VERSION, &scd, &swapchain, &dev, NULL, &devcon);

	assert(swapchain != NULL);

	// set viewport
	D3D11_VIEWPORT viewport = {};
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = DEFAULT_SCREEN_WIDTH;
	viewport.Height = DEFAULT_SCREEN_HEIGHT;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;

	devcon->RSSetViewports(1, &viewport);

	// create and set back buffer
	ID3D11Texture2D* pBackbuffer = {};
	swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackbuffer);
	dev->CreateRenderTargetView(pBackbuffer, NULL, &backbuffer);
	pBackbuffer->Release();
	assert(backbuffer != NULL);

	// create depth texture
	D3D11_TEXTURE2D_DESC depth_texture_desc = {};
	depth_texture_desc.Width = DEFAULT_SCREEN_WIDTH;
	depth_texture_desc.Height = DEFAULT_SCREEN_HEIGHT;
	depth_texture_desc.MipLevels = 1;
	depth_texture_desc.ArraySize = 1;
	depth_texture_desc.SampleDesc.Count = 1;
	depth_texture_desc.SampleDesc.Quality = 0;
	depth_texture_desc.Format = DXGI_FORMAT_D32_FLOAT; //DXGI_FORMAT_D24_UNORM_S8_UINT;
	depth_texture_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

	ID3D11Texture2D* pDepthStencil = NULL;
	HRESULT hr = dev->CreateTexture2D(&depth_texture_desc, NULL, &pDepthStencil);
	assert(SUCCEEDED(hr) && "Failed to create depth texture");

	D3D11_DEPTH_STENCIL_DESC depth_stencil_desc = {};

	// Depth test parameters
	depth_stencil_desc.DepthEnable = true;
	depth_stencil_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	depth_stencil_desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;

	// Stencil test parameters
	depth_stencil_desc.StencilEnable = true;
	depth_stencil_desc.StencilReadMask = 0xFF;
	depth_stencil_desc.StencilWriteMask = 0xFF;

	// Stencil operations if pixel is front-facing
	depth_stencil_desc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	depth_stencil_desc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
	depth_stencil_desc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	depth_stencil_desc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;

	// Stencil operations if pixel is back-facing
	depth_stencil_desc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	depth_stencil_desc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR;
	depth_stencil_desc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	depth_stencil_desc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;

	ID3D11DepthStencilState* pDepthStencilState = NULL;
	dev->CreateDepthStencilState(&depth_stencil_desc, &pDepthStencilState);
	devcon->OMSetDepthStencilState(pDepthStencilState, 1);

	D3D11_DEPTH_STENCIL_VIEW_DESC depth_stencil_view_desc = {};
	depth_stencil_view_desc.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
	depth_stencil_view_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depth_stencil_view_desc.Texture2D.MipSlice = 0;

		hr = dev->CreateDepthStencilView(pDepthStencil, NULL, &depthbuffer);
	assert(SUCCEEDED(hr) && "Failed to create depth view");

	pDepthStencil->Release();

	// Set render targets
	devcon->OMSetRenderTargets(1, &backbuffer, depthbuffer);

	D3D11_RASTERIZER_DESC rasterizer_state_desc = {};
	rasterizer_state_desc.FillMode = D3D11_FILL_SOLID;
	rasterizer_state_desc.CullMode = D3D11_CULL_FRONT;
	rasterizer_state_desc.FrontCounterClockwise = false;

	ID3D11RasterizerState* pRasterizerState = {};
	hr = dev->CreateRasterizerState(&rasterizer_state_desc, &pRasterizerState);
	assert(SUCCEEDED(hr) && "Failed to create rasterizer state");

	devcon->RSSetState(pRasterizerState);
	pRasterizerState->Release();

#ifdef USE_RAW_INPUT
	// Init input devices
	RAWINPUTDEVICE devices[2] = {};
	devices[0] = { 0x01, 0x02, 0, 0 };
	devices[1] = { 0x01, 0x06, 0, 0 };

	assert(RegisterRawInputDevices(devices, 2, sizeof(devices[0])));
#endif

	// setup projection and view matrix
	projection_matrix = glm::perspective(glm::radians(70.0f), (float)DEFAULT_SCREEN_WIDTH / (float)DEFAULT_SCREEN_HEIGHT, NEAR_CLIP_PLANE, FAR_CLIP_PLANE);
	view_matrix = matrix4(1.0f);

	skybox_t sky;

	terrain_t terrain;

	auto gen = std::bind(std::uniform_real_distribution<>(-30,10),std::default_random_engine((unsigned int)std::chrono::system_clock::now().time_since_epoch().count()));
	water_t water(grid_array_size, static_cast<float>(gen()));

	camera_t camera;

	MSG msg = {};

	bool is_running = true;
	while(is_running)
	{
		if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);

			switch(msg.message)
			{
				case WM_QUIT:
				{
					is_running = false;
				} break;
			}
		}
		else
		{
			// Update
			if(input_manager.mouse_button_held(mouse_button_right))
			{
				constexpr float default_camera_rotate_speed = 0.5f;
				constexpr float default_camera_move_speed = 0.1f;

				// CAMERA ROTATION
				vector2i mouse_delta = input_manager.get_mouse_delta();
				vector3f camera_rotation = vector3f(mouse_delta.y, -mouse_delta.x, 0);
				camera_rotation *= default_camera_rotate_speed;
				camera.rotation += camera_rotation;

				if(camera.rotation.y > 360.0f)
				{
					camera.rotation.y -= 360.0f;
				}
				if(camera.rotation.y < 0.0f)
				{
					camera.rotation.y += 360.0f;
				}

				if(camera.rotation.x > 89.0f)
				{
					camera.rotation.x = 89.0f;
				}
				if(camera.rotation.x < -89.0f)
				{
					camera.rotation.x = -89.0f;
				}

				// CAMERA MOVEMENT
				float camera_move_speed = default_camera_move_speed;
				if(input_manager.key_held(key_code_lshift))
				{
					camera_move_speed = 1.5f;
				}
				else if(input_manager.key_held(key_code_lmenu))
				{
					camera_move_speed = 0.01f;
				}

				vector3f vel = {};
				if(!(input_manager.key_held(key_code_e) && input_manager.key_held(key_code_q)))
				{
					if(input_manager.key_held(key_code_e))
					{
						vel += camera.up;
					}
					else if(input_manager.key_held(key_code_q))
					{
						vel -= camera.up;
					}
				}

				if(!(input_manager.key_held(key_code_d) && input_manager.key_held(key_code_a)))
				{
					if(input_manager.key_held(key_code_d))
					{
						vel += camera.right;
					}
					else if(input_manager.key_held(key_code_a))
					{
						vel -= camera.right;
					}
				}

				if(!(input_manager.key_held(key_code_s) && input_manager.key_held(key_code_w)))
				{
					if(input_manager.key_held(key_code_s))
					{
						vel -= camera.front;
					}
					else if(input_manager.key_held(key_code_w))
					{
						vel += camera.front;
					}
				}

				vel = sqrt_magnitude(vel);
				vel *= camera_move_speed;

				camera.position += vel;
			}

			// Render
			view_matrix = camera.get_view_matrix();

			float color[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			devcon->ClearRenderTargetView(backbuffer, color);
			devcon->ClearDepthStencilView(depthbuffer, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

			terrain.draw();
			water.draw();

			sky.draw();

			// TODO: Clouds (maybe?????)
			// TODO: Draw sun in the sky.
			// TODO: Screen space AO

			swapchain->Present(1, 0);

			input_manager.mouse_delta = { 0,0 };
		}
	}

	asset_manager.shutdown();

	swapchain->SetFullscreenState(FALSE, NULL);

	swapchain->Release();
	dev->Release();
	devcon->Release();

	backbuffer->Release();
	depthbuffer->Release();

	return (int)msg.wParam;
}
