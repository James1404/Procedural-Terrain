// ---------
// CONSTANTS
// ---------

cbuffer CONSTANT_BUFFER : register(b0)
{
	float4x4 view_and_projection;
	float water_level;
};

float random(float2 p) {
    return frac(sin(dot(p.xy,float2(12.9898,78.233)))*43758.5453123);
}

// -------------
// VERTEX SHADER
// -------------

struct VS_INPUT
{
	float3 position : POSITION;
	float3 normal : NORMAL;
	float2 texcoord: TEXCOORD0;
};

struct VS_OUTPUT
{
	float3 vertex_position: POSITION;
	float4 position: SV_POSITION;
	float3 normal : NORMAL;
	float2 texcoord: TEXCOORD0;
};

VS_OUTPUT VSMain(VS_INPUT input)
{
	VS_OUTPUT output;

	output.position = mul(view_and_projection, float4(input.position, 1.0));
	output.vertex_position = input.position;
	output.normal = input.normal;
	output.texcoord = input.texcoord;
	
	return output;
}

// ------------
// PIXEL SHADER
// ------------

#if 0
struct pbr_material_t
{
	Texture2D albedo_texture : register(t0);
	SamplerState albedo_sampler: register(s0);

	Texture2D normal_texture : register(t1);
	SamplerState normal_sampler: register(s1);

	Texture2D ao_texture : register(t2);
	SamplerState ao_sampler: register(s2);

	Texture2D roughness_texture : register(t3);
	SamplerState roughness_sampler: register(s3);

	float3 specular;
	float shininess;
};
#endif

Texture2D grass_texture : register(t0);
SamplerState grass_sampler : register(s0);

Texture2D cliff_texture : register(t1);
SamplerState cliff_sampler : register(s1);

Texture2D sand_texture : register(t2);
SamplerState sand_sampler : register(s2);

struct PS_INPUT
{
	float3 vertex_position: POSITION;
	float4 position: SV_POSITION;
	float3 normal : NORMAL;
	float2 texcoord: TEXCOORD0;
};

struct PS_OUTPUT
{
	float4 color;
};

#define texture_scale 8
float4 grass_triplanar_mapping(float3 position, float3 normal, float2 texcoord)
{
	float3 blend_sharpness = float3(50, 1, 50);

	float2 uvX = position.zy / texture_scale;
	float2 uvY = position.xz / texture_scale;
	float2 uvZ = position.xy / texture_scale;

	float3 diffX = cliff_texture.Sample(cliff_sampler, uvX).rgb;
	float3 diffY = grass_texture.Sample(grass_sampler, uvY).rgb;
	float3 diffZ = cliff_texture.Sample(cliff_sampler, uvZ).rgb;

	float3 blendWeights = pow(abs(normal), blend_sharpness);
	blendWeights = blendWeights / (blendWeights.x + blendWeights.y + blendWeights.z);

	return float4(diffX * blendWeights.x + diffY * blendWeights.y + diffZ * blendWeights.z, 0);
}

PS_OUTPUT PSMain(PS_INPUT input) : SV_TARGET
{
	PS_OUTPUT output;

	float4 color = grass_triplanar_mapping(input.vertex_position, input.normal, input.texcoord);
	
	float4 underwater_color = sand_texture.Sample(sand_sampler, input.texcoord / texture_scale);
	float water_blend_max = 10 + 
							sin((input.vertex_position.x +
							 	 input.vertex_position.z) * 0.2);
	float water_blend_min = 1;

	// TODO: randomize height so the sand is less uniform
	float blend_value = smoothstep(water_level+water_blend_max,water_level-water_blend_min,input.vertex_position.y);

	color = lerp(color,underwater_color,blend_value); 

	float3 light_dir = normalize(float3(45, 45, 10));
	float3 diffuse = max(dot(-light_dir, normalize(input.normal)), 0);

	output.color = color * float4(diffuse, 1);
	output.color.a = 1;

	return output;
}
