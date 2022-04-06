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

cbuffer VS_CONSTANT_BUFFER : register(b0)
{
	float4x4 view_and_projection;
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
Texture2D top_texture : register(t0);
SamplerState top_sampler : register(s0);

Texture2D right_texture : register(t1);
SamplerState right_sampler : register(s1);

Texture2D front_texture : register(t2);
SamplerState front_sampler : register(s2);

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

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

#define texture_scale 8
float4 triplanar_mapping(float3 position, float3 normal, float2 texcoord)
{
	float3 blend_sharpness = float3(15, 1, 15);

	float2 uvX = position.zy / texture_scale;
	float2 uvY = position.xz / texture_scale;
	float2 uvZ = position.xy / texture_scale;

	float3 diffX = front_texture.Sample(front_sampler, uvX).xyz;
	float3 diffY = top_texture.Sample(top_sampler, uvY).xyz;
	float3 diffZ = right_texture.Sample(right_sampler, uvZ).xyz;

	float3 blendWeights = pow(abs(normal), blend_sharpness);
	blendWeights = blendWeights / (blendWeights.x + blendWeights.y + blendWeights.z);

	return float4(diffX * blendWeights.x + diffY * blendWeights.y + diffZ * blendWeights.z, 0);
}

PS_OUTPUT PSMain(PS_INPUT input) : SV_TARGET
{
	PS_OUTPUT output;

	float4 color = triplanar_mapping(input.vertex_position, input.normal, input.texcoord);
	
	float mapped_value = map(input.vertex_position.y, 20, 100, 0, 1);
	//color += mapped_value;

	float3 light_dir = normalize(float3(45, 45, 10));
	float3 diffuse = max(dot(-light_dir, normalize(input.normal)), 0);

	// TODO: Change texture to sand if low enough
	// And change to snow if high enough.

	output.color = color * float4(diffuse, 1);

	return output;
}
